//! Waveform I/O and reference-audio preparation.
//!
//! `prepare_prompt` mirrors `inference.py`: peak-normalise, cut on silence gaps
//! of at least 150 ms, then keep 50 ms of context around each speech run. The
//! reference's exact thresholds matter because the codec sees the result.

use anyhow::{bail, Context, Result};
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

/// Same as [`prepare_prompt`] for already-decoded samples.
pub fn prepare_prompt_samples(samples: &[f32], rate: u32, sample_rate: usize) -> Result<Vec<f32>> {
    if rate as usize != sample_rate {
        // The reference resamples with torchaudio; a wrong rate would silently
        // shift every codec frame, so refuse rather than guess.
        bail!("reference audio is {rate} Hz, expected {sample_rate} Hz (resampling not implemented yet)");
    }
    if samples.is_empty() || !samples.iter().all(|v| v.is_finite()) {
        bail!("reference audio must be nonempty and finite");
    }
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
