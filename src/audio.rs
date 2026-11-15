//! Waveform I/O and reference-audio preparation.
//!
//! `prepare_prompt` mirrors `inference.py`: peak-normalise, cut on silence gaps
//! of at least 150 ms, then keep 50 ms of context around each speech run. The
//! reference's exact thresholds matter because the codec sees the result.

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::path::Path;

/// Mono samples plus the file's sample rate.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    if channels == 0 || samples.len() % channels != 0 {
        bail!("{}: {} samples is not a multiple of {} channels", path.display(), samples.len(), channels);
    }
    let mono = if channels == 1 {
        samples
    } else {
        samples.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32).collect()
    };
    Ok((mono, spec.sample_rate))
}

/// `prepare_prompt` from the reference implementation.
pub fn prepare_prompt(path: &Path, sample_rate: usize) -> Result<Vec<f32>> {
    let (samples, rate) = read_wav(path)?;
    prepare_prompt_samples(&samples, rate, sample_rate)
}

/// Band-limited resampling, ported from `torchaudio.functional.resample`.
///
/// The reference resamples a reference recording whose rate is not 24 kHz
/// (`torchaudio.functional.resample(audio, rate, sample_rate)`, torchaudio 2.8:
/// `sinc_interp_hann`, `lowpass_filter_width=6`, `rolloff=0.99`). That is a
/// windowed-sinc FIR bank: one `new_freq`-phase kernel of
/// `2*width + orig_freq` taps, applied with stride `orig_freq`. It is plain
/// arithmetic, so the port carries the formula rather than a dependency; the
/// accumulation order differs from torch's gemm by rounding only.
pub fn resample(samples: &[f32], orig_freq: u32, new_freq: u32) -> Vec<f32> {
    const LOWPASS_FILTER_WIDTH: f32 = 6.0;
    const ROLLOFF: f32 = 0.99;
    if orig_freq == new_freq || samples.is_empty() {
        return samples.to_vec();
    }
    let gcd = {
        fn gcd(mut a: u32, mut b: u32) -> u32 {
            while b != 0 {
                let t = a % b;
                a = b;
                b = t;
            }
            a.max(1)
        }
        gcd(orig_freq, new_freq)
    };
    let orig = (orig_freq / gcd) as f32;
    let new = (new_freq / gcd) as f32;
    let orig_i = orig as usize;
    let new_i = new as usize;
    let base_freq = orig.min(new) * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig / base_freq).ceil() as usize;
    let taps = 2 * width + orig_i;

    // Kernel: one row per output phase.
    let mut kernel = vec![0.0f32; new_i * taps];
    for phase in 0..new_i {
        for tap in 0..taps {
            // t = (0 - phase)/new + (tap - width)/orig, scaled by the low-pass
            // corner frequency and clamped to the window's support.
            let mut t = (-(phase as f32) / new + (tap as f32 - width as f32) / orig) * base_freq;
            t = t.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
            let window = (t * std::f32::consts::PI / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
            let t = t * std::f32::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[phase * taps + tap] = sinc * window * (base_freq / orig);
        }
    }

    // Zero padding `width` on both sides plus `orig` trailing taps, as torch does.
    let padded_len = samples.len() + 2 * width + orig_i;
    let mut padded = vec![0.0f32; padded_len];
    padded[width..width + samples.len()].copy_from_slice(samples);

    let target_length = ((new * samples.len() as f32) / orig).ceil() as usize;
    let mut out = vec![0.0f32; target_length];
    let phases = target_length.div_ceil(new_i);
    out.par_chunks_mut(new_i).enumerate().take(phases).for_each(|(block, chunk)| {
        let base = block * orig_i;
        for (phase, value) in chunk.iter_mut().enumerate() {
            let row = &kernel[phase * taps..(phase + 1) * taps];
            let mut acc = 0.0f32;
            for (tap, weight) in row.iter().enumerate() {
                acc += weight * padded[base + tap];
            }
            *value = acc;
        }
    });
    out
}

/// Same as [`prepare_prompt`] for already-decoded samples.
pub fn prepare_prompt_samples(samples: &[f32], rate: u32, sample_rate: usize) -> Result<Vec<f32>> {
    if samples.is_empty() || !samples.iter().all(|v| v.is_finite()) {
        bail!("reference audio must be nonempty and finite");
    }
    let resampled = resample(samples, rate, sample_rate as u32);
    let samples = resampled.as_slice();
    let peak = samples.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak == 0.0 {
        bail!("reference audio is silent");
    }
    let normalized: Vec<f32> = samples.iter().map(|v| v / (peak + 1e-8)).collect();

    let threshold = if normalized.len() / sample_rate > 10 { 0.02 } else { 0.015 };
    let speech: Vec<usize> = normalized
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > threshold)
        .map(|(index, _)| index)
        .collect();
    if speech.is_empty() {
        bail!("reference audio contains no speech above the silence threshold");
    }

    let gap = (0.15 * sample_rate as f64) as usize;
    let mut starts = vec![speech[0]];
    let mut ends = Vec::new();
    for pair in speech.windows(2) {
        if pair[1] - pair[0] - 1 >= gap {
            ends.push(pair[0] + 1);
            starts.push(pair[1]);
        }
    }
    let tail = normalized.len() - speech[speech.len() - 1] - 1;
    ends.push(if tail >= gap { speech[speech.len() - 1] + 1 } else { normalized.len() });

    let padding = (0.05 * sample_rate as f64) as usize;
    let mut out = Vec::new();
    for (start, end) in starts.iter().zip(&ends) {
        let from = start.saturating_sub(padding);
        let to = (*end + padding).min(normalized.len());
        out.extend_from_slice(&normalized[from..to]);
    }
    Ok(out)
}
