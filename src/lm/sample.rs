//! top-k sampler, mirroring `m5/model/model.py::sample_topk` / `_multinomial_sample_one_no_sync`.
//!
//! The reference draws `q ~ Exp(1)` and takes `argmax(probs / q)`, which is
//! multinomial sampling through the inverse-CDF trick. Only the arithmetic is
//! reproduced here: the noise is an argument, so a recorded draw can be replayed
//! without reimplementing torch's MT19937/Philox streams.

use anyhow::{bail, Result};

/// One token from `logits`, given the exponential noise draw `noise`.
pub fn sample_topk(logits: &[f32], temperature: f32, topk: usize, need_finish: bool,
                   noise: &[f32]) -> Result<u32> {
    if logits.len() != noise.len() {
        bail!("sample_topk: {} logits vs {} noise values", logits.len(), noise.len());
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        bail!("sample_topk: bad temperature {temperature}");
    }
    if topk == 0 || topk > logits.len() {
        bail!("sample_topk: topk {topk} out of range 1..={}", logits.len());
    }

    let mut scores = logits.to_vec();
    if !need_finish {
        // The reference forbids the EOS token (id 0) while the utterance is short.
        scores[0] = -1e5;
    }
    for score in scores.iter_mut() {
        *score /= temperature;
    }

    // The k-th largest score is the cutoff; only the top-k survive.
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let cutoff = sorted[topk - 1];

    // `log_softmax` over the masked scores, then `softmax` again (as the reference
    // does; mathematically softmax(log_softmax(x)) == softmax(x)).
    let masked: Vec<f32> = scores.iter().map(|v| if *v < cutoff { f32::NEG_INFINITY } else { *v }).collect();
    let max = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = masked.iter().map(|v| (*v - max).exp()).collect();
    let sum: f64 = exps.iter().map(|v| *v as f64).sum();
    let logp: Vec<f32> = exps.iter().map(|v| ((*v as f64 / sum) as f32).ln()).collect();

    let log_max = logp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logp.iter().map(|v| (*v - log_max).exp()).collect();
    let prob_sum: f64 = probs.iter().map(|v| *v as f64).sum();
    let probs: Vec<f32> = probs.iter().map(|v| (*v as f64 / prob_sum) as f32).collect();

    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, (p, q)) in probs.iter().zip(noise).enumerate() {
        if *p <= 0.0 {
            continue;
        }
        let value = p / q;
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    Ok(best as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_of_unit_noise_follows_probs() {
        // With identical noise, the largest probability wins.
        let logits = vec![0.0f32, 5.0, 1.0, 4.0];
        let noise = vec![1.0f32; 4];
        assert_eq!(sample_topk(&logits, 1.0, 4, true, &noise).unwrap(), 1);
    }

    #[test]
    fn eos_is_masked_until_finish() {
        let logits = vec![100.0f32, 0.0];
        let noise = vec![1.0f32; 2];
        // need_finish=false forbids token 0, so token 1 wins despite the logit.
        assert_eq!(sample_topk(&logits, 1.0, 2, false, &noise).unwrap(), 1);
        assert_eq!(sample_topk(&logits, 1.0, 2, true, &noise).unwrap(), 0);
    }

    #[test]
    fn topk_cuts_the_tail() {
        // Only the top-1 survives; the huge noise on token 2 keeps it alive.
        let logits = vec![5.0f32, 4.0, 1.0];
        let noise = vec![10.0f32, 1e-6, 1e-6];
        assert_eq!(sample_topk(&logits, 1.0, 1, true, &noise).unwrap(), 0);
    }
}
