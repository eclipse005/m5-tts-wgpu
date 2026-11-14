//! Every op through both backends: the CPU twin is the reference and the same
//! numbers must come out of wgpu. This is the gate the codec and LM graphs rely
//! on, so a kernel bug is caught here instead of inside a waveform.

use m5_tts_wgpu::device::{Device, Tensor};
use m5_tts_wgpu::ops::{self, Conv1d};

/// Tolerance for a single op in f32 (accumulation order differs slightly).
const TOL: f32 = 5e-5;

fn gpu() -> Option<Device> {
    match Device::open("gpu") {
        Ok(dev) => {
            eprintln!("gpu: {}", dev.describe());
            Some(dev)
        }
        Err(e) => {
            eprintln!("gpu unavailable, cpu-only run: {e}");
            None
        }
    }
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn assert_close(name: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{name}: length");
    let diff = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    eprintln!("{name}: max|diff|={diff:.3e}");
    assert!(diff < TOL, "{name}: max|diff|={diff} over tolerance {TOL}");
}

fn tensor(dev: &Device, shape: [usize; 2], seed: u64) -> Tensor {
    dev.tensor(shape, &rand_vec(shape[0] * shape[1], seed)).unwrap()
}

#[test]
fn linear_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    // 1-token GEMV, 16-token depth prefill, a text prefill, and an odd shape.
    for (m, n, k) in [(1usize, 2051usize, 512usize), (16, 2051, 512), (89, 2048, 512), (3, 129, 64)] {
        for with_bias in [true, false] {
            let cpu = Device::Cpu;
            let x = tensor(&cpu, [m, k], 1);
            let w = tensor(&cpu, [n, k], 2);
            let bias = with_bias.then(|| tensor(&cpu, [n, 1], 3));
            let mut want = cpu.zeros([m, n]).unwrap();
            ops::linear(&cpu, &x, &w, bias.as_ref(), &mut want).unwrap();
            let want = cpu.read(&want).unwrap();

            let x = tensor(&gpu, [m, k], 1);
            let w = tensor(&gpu, [n, k], 2);
            let bias = with_bias.then(|| tensor(&gpu, [n, 1], 3));
            let mut got = gpu.zeros([m, n]).unwrap();
            ops::linear(&gpu, &x, &w, bias.as_ref(), &mut got).unwrap();
            let got = gpu.read(&got).unwrap();
            assert_close(&format!("linear m={m} n={n} k={k} bias={with_bias}"), &want, &got);
        }
    }
}

#[test]
fn conv1d_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    // Geometry straight out of the codec decoder: k=7 dilated 1/3/9, causal
    // padding, plus the k=1 pointwise taps.
    for (cin, cout, ks, dilation, t_in) in
        [(8usize, 6usize, 7usize, 1usize, 32usize), (6, 4, 7, 3, 33), (4, 4, 1, 1, 16)]
    {
        let p = Conv1d::causal(cin, cout, ks, t_in).dilated(dilation);
        let cpu = Device::Cpu;
        let x = tensor(&cpu, [cin, t_in], 7);
        let w = tensor(&cpu, [cout, cin * ks], 8);
        let b = tensor(&cpu, [cout, 1], 9);
        let mut want = cpu.zeros([cout, p.t_out]).unwrap();
        ops::conv1d(&cpu, &x, &w, &b, &mut want, &p).unwrap();
        let want = cpu.read(&want).unwrap();

        let x = tensor(&gpu, [cin, t_in], 7);
        let w = tensor(&gpu, [cout, cin * ks], 8);
        let b = tensor(&gpu, [cout, 1], 9);
        let mut got = gpu.zeros([cout, p.t_out]).unwrap();
        ops::conv1d(&gpu, &x, &w, &b, &mut got, &p).unwrap();
        let got = gpu.read(&got).unwrap();
        assert_close(&format!("conv1d cin={cin} ks={ks} dil={dilation}"), &want, &got);
    }
}

#[test]
fn snake_beta_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    let (c, t) = (5usize, 11usize);
    let cpu = Device::Cpu;
    let alpha = tensor(&cpu, [c, 1], 11);
    let beta = tensor(&cpu, [c, 1], 12);
    let x = tensor(&cpu, [c, t], 10);
    let mut want = cpu.zeros([c, t]).unwrap();
    ops::write(&mut want, &cpu.read(&x).unwrap()).unwrap();
    ops::snake_beta(&cpu, &mut want, &alpha, &beta).unwrap();
    let want = cpu.read(&want).unwrap();

    let alpha_g = tensor(&gpu, [c, 1], 11);
    let beta_g = tensor(&gpu, [c, 1], 12);
    let mut got = gpu.tensor([c, t], &cpu.read(&x).unwrap()).unwrap();
    ops::snake_beta(&gpu, &mut got, &alpha_g, &beta_g).unwrap();
    let got = gpu.read(&got).unwrap();
    assert_close("snake_beta", &want, &got);
}

#[test]
fn add_and_residual_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    let (rows, cols) = (4usize, 250usize);
    let cpu = Device::Cpu;
    let a = tensor(&cpu, [rows, cols], 13);
    let b = tensor(&cpu, [rows, cols], 14);
    let (ha, hb) = (cpu.read(&a).unwrap(), cpu.read(&b).unwrap());

    let mut want = cpu.zeros([rows, cols]).unwrap();
    ops::add(&cpu, &a, &b, &mut want).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([rows, cols]).unwrap();
    ops::add(&gpu, &gpu.tensor([rows, cols], &ha).unwrap(), &gpu.tensor([rows, cols], &hb).unwrap(), &mut got)
        .unwrap();
    assert_close("add", &want, &gpu.read(&got).unwrap());

    let mut y = gpu.tensor([rows, cols], &ha).unwrap();
    ops::residual(&gpu, &mut y, &gpu.tensor([rows, cols], &hb).unwrap()).unwrap();
    let got = gpu.read(&y).unwrap();
    let want: Vec<f32> = ha.iter().zip(&hb).map(|(x, z)| x + z).collect();
    assert_close("residual", &want, &got);
}

#[test]
fn transpose_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    let (rows, cols) = (7usize, 13usize);
    let cpu = Device::Cpu;
    let x = tensor(&cpu, [rows, cols], 15);
    let mut want = cpu.zeros([cols, rows]).unwrap();
    ops::transpose(&cpu, &x, &mut want).unwrap();
    let want = cpu.read(&want).unwrap();

    let xg = gpu.tensor([rows, cols], &cpu.read(&x).unwrap()).unwrap();
    let mut got = gpu.zeros([cols, rows]).unwrap();
    ops::transpose(&gpu, &xg, &mut got).unwrap();
    assert_close("transpose", &want, &gpu.read(&got).unwrap());
}

#[test]
fn gather_sum_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    // rvq shape: 16 codebooks x 2048 entries x 256 dims, 5 frames.
    let (book, dim, frames, codebooks) = (2048usize, 256usize, 5usize, 16usize);
    let codes: Vec<u32> = (0..frames * codebooks).map(|i| ((i * 37 + 11) % book) as u32).collect();
    let tables = rand_vec(codebooks * book * dim, 16);
    // Real cluster_usage is positive and order ~1e2; a synthetic value near zero
    // would divide by the 1e-5 clamp and turn f32 rounding into a visible diff.
    let usage: Vec<f32> = rand_vec(codebooks * book, 17).iter().map(|v| 1.0 + (v + 0.5) * 99.0).collect();

    let cpu = Device::Cpu;
    let mut want = cpu.zeros([dim, frames]).unwrap();
    ops::gather_sum(&cpu, &codes, &cpu.tensor([codebooks * book, dim], &tables).unwrap(),
                    &cpu.tensor([codebooks * book, 1], &usage).unwrap(), &mut want,
                    dim, frames, 0, codebooks, book, codebooks).unwrap();
    let want = cpu.read(&want).unwrap();

    let mut got = gpu.zeros([dim, frames]).unwrap();
    ops::gather_sum(&gpu, &codes, &gpu.tensor([codebooks * book, dim], &tables).unwrap(),
                    &gpu.tensor([codebooks * book, 1], &usage).unwrap(), &mut got,
                    dim, frames, 0, codebooks, book, codebooks).unwrap();
    assert_close("gather_sum", &want, &gpu.read(&got).unwrap());
}

#[test]
fn rms_norm_layer_norm_and_silu_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    let (rows, cols) = (5usize, 512usize);
    let cpu = Device::Cpu;
    let x = tensor(&cpu, [rows, cols], 21);
    let w = tensor(&cpu, [cols, 1], 22);
    let b = tensor(&cpu, [cols, 1], 23);
    let hx = cpu.read(&x).unwrap();

    let mut want = cpu.zeros([rows, cols]).unwrap();
    ops::rms_norm(&cpu, &x, &w, &mut want, 1e-5).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([rows, cols]).unwrap();
    ops::rms_norm(&gpu, &gpu.tensor([rows, cols], &hx).unwrap(), &gpu.tensor([cols, 1], &cpu.read(&w).unwrap()).unwrap(),
                  &mut got, 1e-5).unwrap();
    assert_close("rms_norm", &want, &gpu.read(&got).unwrap());

    let mut want = cpu.zeros([rows, cols]).unwrap();
    ops::layer_norm(&cpu, &x, &w, &b, &mut want, 1e-6).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([rows, cols]).unwrap();
    ops::layer_norm(&gpu, &gpu.tensor([rows, cols], &hx).unwrap(),
                    &gpu.tensor([cols, 1], &cpu.read(&w).unwrap()).unwrap(),
                    &gpu.tensor([cols, 1], &cpu.read(&b).unwrap()).unwrap(), &mut got, 1e-6).unwrap();
    assert_close("layer_norm", &want, &gpu.read(&got).unwrap());

    let gate = tensor(&cpu, [rows, 1024], 24);
    let up = tensor(&cpu, [rows, 1024], 25);
    let mut want = cpu.zeros([rows, 1024]).unwrap();
    ops::silu_mul(&cpu, &gate, &up, &mut want).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([rows, 1024]).unwrap();
    ops::silu_mul(&gpu, &gpu.tensor([rows, 1024], &cpu.read(&gate).unwrap()).unwrap(),
                  &gpu.tensor([rows, 1024], &cpu.read(&up).unwrap()).unwrap(), &mut got).unwrap();
    assert_close("silu_mul", &want, &gpu.read(&got).unwrap());
}

#[test]
fn rope_and_attention_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    let (frames, heads, dim, window) = (57usize, 16usize, 64usize, 72usize);
    let cols = heads * dim;
    let cpu = Device::Cpu;

    let base = tensor(&cpu, [frames, cols], 31);
    let hb = cpu.read(&base).unwrap();
    let mut want = gpu_less_rope(&hb, frames, heads, dim, 10000.0);
    let mut got = gpu.tensor([frames, cols], &hb).unwrap();
    ops::rope(&gpu, &mut got, heads, dim, 10000.0).unwrap();
    let hg = gpu.read(&got).unwrap();
    assert_close("rope", &want, &hg);
    want.clear();

    // Attention: three independent projections, then compare.
    let q = tensor(&cpu, [frames, cols], 32);
    let k = tensor(&cpu, [frames, cols], 33);
    let v = tensor(&cpu, [frames, cols], 34);
    let mut want = cpu.zeros([frames, cols]).unwrap();
    ops::attention(&cpu, &q, &k, &v, &mut want, heads, dim, window).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([frames, cols]).unwrap();
    ops::attention(&gpu, &gpu.tensor([frames, cols], &cpu.read(&q).unwrap()).unwrap(),
                   &gpu.tensor([frames, cols], &cpu.read(&k).unwrap()).unwrap(),
                   &gpu.tensor([frames, cols], &cpu.read(&v).unwrap()).unwrap(),
                   &mut got, heads, dim, window).unwrap();
    let got = gpu.read(&got).unwrap();
    let diff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    eprintln!("attention: max|diff|={diff:.3e}");
    assert!(diff < 1e-5, "attention diff {diff}");
}

/// Independent rope reference (sin_cos of the exact same angles).
fn gpu_less_rope(x: &[f32], frames: usize, heads: usize, dim: usize, theta: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    let half = dim / 2;
    for t in 0..frames {
        for h in 0..heads {
            for j in 0..half {
                let freq = theta.powf(-2.0 * j as f32 / dim as f32);
                let angle = t as f32 * freq;
                let (s, c) = angle.sin_cos();
                let base = t * heads * dim + h * dim;
                let a = x[base + j];
                let b = x[base + j + half];
                out[base + j] = a * c - b * s;
                out[base + j + half] = b * c + a * s;
            }
        }
    }
    out
}

#[test]
fn conv_transpose_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    // SEANet upsamplers use kernel = 2*stride and trim; the ConvNeXt stages use
    // kernel == stride. Both have to agree with the CPU twin.
    for (cin, cout, rate, one_tap) in
        [(4usize, 2usize, 2usize, false), (6, 3, 8, false), (8, 4, 5, false), (4, 4, 2, true), (6, 6, 3, true)]
    {
        // `kernel == stride` is the latent upsampler (one tap per phase).
        let ks = if one_tap { rate } else { 2 * rate };
        let (t_in, t_out) = (10usize, 10 * rate);
        let cpu = Device::Cpu;
        let x = tensor(&cpu, [cin, t_in], 41);
        let w = tensor(&cpu, [cin, cout * ks], 42);
        let b = tensor(&cpu, [cout, 1], 43);
        let mut want = cpu.zeros([cout, t_out]).unwrap();
        ops::conv_transpose1d(&cpu, &x, &w, &b, &mut want, ks, rate).unwrap();
        let want = cpu.read(&want).unwrap();

        let mut got = gpu.zeros([cout, t_out]).unwrap();
        ops::conv_transpose1d(&gpu, &gpu.tensor([cin, t_in], &cpu.read(&x).unwrap()).unwrap(),
                              &gpu.tensor([cin, cout * ks], &cpu.read(&w).unwrap()).unwrap(),
                              &gpu.tensor([cout, 1], &cpu.read(&b).unwrap()).unwrap(),
                              &mut got, ks, rate).unwrap();
        assert_close(&format!("conv_transpose ks={ks} stride={rate}"), &want, &gpu.read(&got).unwrap());
    }
}

#[test]
fn seanet_shapes_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    // The exact op shapes the SEANet tail uses, at a small time extent so the
    // test stays fast; a shape/layout bug shows up regardless of length.
    let cpu = Device::Cpu;

    // depthwise 96 channels, k=7 causal
    let (c, t) = (96usize, 64usize);
    let x = tensor(&cpu, [c, t], 51);
    let w = tensor(&cpu, [c, 7], 52);
    let b = tensor(&cpu, [c, 1], 53);
    let mut want = cpu.zeros([c, t]).unwrap();
    ops::conv1d_depthwise(&cpu, &x, &w, &b, &mut want, 7, 6).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([c, t]).unwrap();
    ops::conv1d_depthwise(&gpu, &gpu.tensor([c, t], &cpu.read(&x).unwrap()).unwrap(),
                          &gpu.tensor([c, 7], &cpu.read(&w).unwrap()).unwrap(),
                          &gpu.tensor([c, 1], &cpu.read(&b).unwrap()).unwrap(), &mut got, 7, 6).unwrap();
    assert_close("depthwise c=96", &want, &gpu.read(&got).unwrap());

    // dilated conv1d 96 -> 96, k=7, dilation 9
    let dil = 9usize;
    let p = Conv1d::causal(c, c, 7, t).dilated(dil);
    let w = tensor(&cpu, [c, c * 7], 54);
    let b = tensor(&cpu, [c, 1], 55);
    let mut want = cpu.zeros([c, t]).unwrap();
    ops::conv1d(&cpu, &x, &w, &b, &mut want, &p).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([c, t]).unwrap();
    ops::conv1d(&gpu, &gpu.tensor([c, t], &cpu.read(&x).unwrap()).unwrap(),
                &gpu.tensor([c, c * 7], &cpu.read(&w).unwrap()).unwrap(),
                &gpu.tensor([c, 1], &cpu.read(&b).unwrap()).unwrap(), &mut got, &p).unwrap();
    assert_close("conv1d 96x96 dil=9", &want, &gpu.read(&got).unwrap());

    // mono output conv: 96 -> 1
    let p = Conv1d::causal(c, 1, 7, t);
    let w = tensor(&cpu, [1, c * 7], 56);
    let b = tensor(&cpu, [1, 1], 57);
    let mut want = cpu.zeros([1, t]).unwrap();
    ops::conv1d(&cpu, &x, &w, &b, &mut want, &p).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([1, t]).unwrap();
    ops::conv1d(&gpu, &gpu.tensor([c, t], &cpu.read(&x).unwrap()).unwrap(),
                &gpu.tensor([1, c * 7], &cpu.read(&w).unwrap()).unwrap(),
                &gpu.tensor([1, 1], &cpu.read(&b).unwrap()).unwrap(), &mut got, &p).unwrap();
    assert_close("conv1d 96->1", &want, &gpu.read(&got).unwrap());

    // wide conv: 1536 -> 768, k=7
    let (cin, cout, t) = (192usize, 96usize, 32usize);
    let x = tensor(&cpu, [cin, t], 58);
    let p = Conv1d::causal(cin, cout, 7, t);
    let w = tensor(&cpu, [cout, cin * 7], 59);
    let b = tensor(&cpu, [cout, 1], 60);
    let mut want = cpu.zeros([cout, t]).unwrap();
    ops::conv1d(&cpu, &x, &w, &b, &mut want, &p).unwrap();
    let want = cpu.read(&want).unwrap();
    let mut got = gpu.zeros([cout, t]).unwrap();
    ops::conv1d(&gpu, &gpu.tensor([cin, t], &cpu.read(&x).unwrap()).unwrap(),
                &gpu.tensor([cout, cin * 7], &cpu.read(&w).unwrap()).unwrap(),
                &gpu.tensor([cout, 1], &cpu.read(&b).unwrap()).unwrap(), &mut got, &p).unwrap();
    assert_close("conv1d 192->96", &want, &gpu.read(&got).unwrap());
}

#[test]
fn conv1d_matches_at_seanet_prime_shape() {
    let Some(gpu) = gpu() else { return };
    // decoder.decoder.0: 1024 -> 1536, k=7, 32 frames (8 codec frames upsampled x4).
    let (cin, cout, ks, t) = (1024usize, 1536usize, 7usize, 32usize);
    let cpu = Device::Cpu;
    let x = tensor(&cpu, [cin, t], 71);
    let w = tensor(&cpu, [cout, cin * ks], 72);
    let b = tensor(&cpu, [cout, 1], 73);
    let p = Conv1d::causal(cin, cout, ks, t);
    let mut want = cpu.zeros([cout, t]).unwrap();
    ops::conv1d(&cpu, &x, &w, &b, &mut want, &p).unwrap();
    let want = cpu.read(&want).unwrap();

    let mut got = gpu.zeros([cout, t]).unwrap();
    ops::conv1d(&gpu, &gpu.tensor([cin, t], &cpu.read(&x).unwrap()).unwrap(),
                &gpu.tensor([cout, cin * ks], &cpu.read(&w).unwrap()).unwrap(),
                &gpu.tensor([cout, 1], &cpu.read(&b).unwrap()).unwrap(), &mut got, &p).unwrap();
    assert_close("conv1d 1024->1536", &want, &gpu.read(&got).unwrap());
}

#[test]
fn gpu_sampler_matches_cpu_sampler() {
    let Some(gpu) = gpu() else { return };
    use m5_tts_wgpu::device::Tokens;
    use m5_tts_wgpu::lm::sample::sample_topk;
    let vocab = 2051usize;
    let cpu = Device::Cpu;
    for (seed, topk, temperature, allow_eos) in
        [(11u64, 10usize, 0.9f32, true), (12, 10, 0.9, false), (13, 1, 1.0, true), (14, 50, 0.7, true)]
    {
        let logits = rand_vec(vocab, seed);
        // Exp(1) noise, the same distribution the reference draws.
        let noise: Vec<f32> = rand_vec(vocab, seed + 100).iter().map(|v| 0.05 + (v + 0.5)).collect();
        let want = sample_topk(&logits, temperature, topk, allow_eos, &noise).unwrap();

        let l = gpu.tensor([1, vocab], &logits).unwrap();
        let n = gpu.tensor([1, vocab], &noise).unwrap();
        let out = gpu.tokens(&[0u32]).unwrap();
        let buffer = out.buffer.as_ref().unwrap();
        m5_tts_wgpu::gpu_ops::sample_topk(gpu.gpu().unwrap(), l.gpu_buffer().unwrap(), n.gpu_buffer().unwrap(),
                                         buffer, vocab, topk, temperature, allow_eos, 0);
        gpu.gpu().unwrap().flush();
        let got = gpu.read_tokens(&Tokens { buffer: Some(buffer.clone()), data: out.data.clone() }).unwrap();
        eprintln!("sampler seed={seed} topk={topk} temp={temperature}: want {want} got {}", got[0]);
        assert_eq!(got[0], want, "sampler mismatch (seed {seed})");
    }
}

#[test]
fn channels_last_conv_matches_channels_first() {
    // Same math, two layouts: the packed channels-last GEMM must reproduce the
    // reference conv exactly (this is what the codec will run on).
    use m5_tts_wgpu::cpu_ops;
    for (cin, cout, ks, dilation, stride, t_in) in
        [(96usize, 96usize, 7usize, 1usize, 1usize, 37usize),
         (192, 96, 7, 9, 1, 29),
         (64, 128, 8, 1, 4, 64),
         (33, 41, 3, 2, 1, 17)]
    {
        let pad = (ks - 1) * dilation;
        let t_out = if stride == 1 { t_in } else { t_in.div_ceil(stride) };
        let x_cl = rand_vec(t_in * cin, 71);
        let w = rand_vec(cout * cin * ks, 72);
        let b = rand_vec(cout, 73);
        let packed = cpu_ops::pack_conv_weights(&w, cout, cin, ks);
        let mut y_cl = vec![0.0f32; t_out * cout];
        cpu_ops::conv1d_channels_last(&x_cl, &packed, &b, &mut y_cl, cin, cout, ks, dilation,
                                      stride, pad, t_in, t_out);
        // Compare with the channels-first kernel on the same data.
        let mut x_cf = vec![0.0f32; cin * t_in];
        for t in 0..t_in {
            for c in 0..cin {
                x_cf[c * t_in + t] = x_cl[t * cin + c];
            }
        }
        let mut y_cf = vec![0.0f32; cout * t_out];
        cpu_ops::conv1d(&x_cf, &w, &b, &mut y_cf, (cin, cout, ks, dilation, stride, pad, t_in, t_out));
        let mut worst = 0.0f32;
        for t in 0..t_out {
            for c in 0..cout {
                worst = worst.max((y_cl[t * cout + c] - y_cf[c * t_out + t]).abs());
            }
        }
        eprintln!("cl vs cf cin={cin} cout={cout} ks={ks} stride={stride}: max|diff|={worst:.3e}");
        assert!(worst < 2e-3, "channels-last conv diff {worst}");
    }
}

#[test]
fn gpu_channels_last_conv_matches_cpu() {
    let Some(gpu) = gpu() else { return };
    use m5_tts_wgpu::cpu_ops;
    use m5_tts_wgpu::gpu_ops;
    for (cin, cout, ks, stride, t_in) in
        [(96usize, 96usize, 7usize, 1usize, 300usize), (192, 96, 7, 1, 190),
         (32, 64, 3, 2, 128), (16, 128, 4, 4, 64)]
    {
        let pad = ks - stride;
        let t_out = if stride == 1 { t_in } else { t_in.div_ceil(stride) };
        let x = rand_vec(t_in * cin, 81);
        let w = rand_vec(cout * cin * ks, 82);
        let b = rand_vec(cout, 83);
        let packed = cpu_ops::pack_conv_weights(&w, cout, cin, ks);

        let mut want = vec![0.0f32; t_out * cout];
        cpu_ops::conv1d_channels_last(&x, &packed, &b, &mut want, cin, cout, ks, 1, stride, pad,
                                      t_in, t_out);

        let xg = gpu.tensor([t_in, cin], &x).unwrap();
        let wg = gpu.tensor([ks * cin, cout], &packed).unwrap();
        let bg = gpu.tensor([cout, 1], &b).unwrap();
        let yg = gpu.zeros([t_out, cout]).unwrap();
        gpu_ops::conv1d_cl(gpu.gpu().unwrap(), xg.gpu_buffer().unwrap(), wg.gpu_buffer().unwrap(),
                           bg.gpu_buffer().unwrap(), yg.gpu_buffer().unwrap(),
                           cin, cout, ks, 1, stride, pad, t_in, t_out);
        let got = gpu.read(&yg).unwrap();
        let mut worst = 0.0f32;
        for (a, c) in want.iter().zip(&got) {
            worst = worst.max((a - c).abs());
        }
        eprintln!("gpu cl conv cin={cin} cout={cout} ks={ks} stride={stride}: max|diff|={worst:.3e}");
        assert!(worst < 5e-3, "gpu channels-last conv diff {worst}");
    }
}

/// Transposed conv against a naive in-test reference, over shapes that the
/// backend-to-backend test misses: several q-blocks (t_in > BLOCK), channel counts
/// that are not a multiple of four, and every stride the codec uses. This is the
/// coverage that would have caught the silently wrong AVX2 path.
#[test]
fn conv_transpose_matches_naive_reference() {
    use m5_tts_wgpu::cpu_ops;
    for (cin, cout, stride, t_in) in [
        (4usize, 2usize, 2usize, 10usize),
        (6, 3, 8, 40),
        (8, 4, 5, 700),     // multi q-block
        (5, 6, 3, 1100),    // multi block, cout % 4 != 0, cin % 4 != 0
        (192, 96, 3, 1200), // the shape block 4 actually runs
    ] {
        let ks = 2 * stride;
        let t_out = t_in * stride;
        let x = rand_vec(cin * t_in, 91);
        let w = rand_vec(cin * cout * ks, 92);
        let b = rand_vec(cout, 93);
        let mut y = vec![0.0f32; cout * t_out];
        cpu_ops::conv_transpose1d(&x, &w, &b, &mut y, cin, cout, ks, stride, t_in);
        assert_transpose_close(&x, &w, &b, &y, cin, cout, ks, stride, t_in);
    }
}

/// The channel-outer core behind [`conv_transpose1d_phased`]'s big-weight path,
/// checked against the same naive gather at small shapes (the production shape
/// that takes this path would need a 75 MB weight tensor).
#[test]
fn conv_transpose_co_outer_matches_naive_reference() {
    use m5_tts_wgpu::cpu_ops;
    for (cin, cout, stride, t_in) in [
        (4usize, 2usize, 2usize, 1usize), // t_in edge: the [0] tap only
        (4, 2, 2, 10),
        (7, 5, 8, 37),    // stride 8 = block 1, odd channels, partial last chunk
        (5, 6, 3, 129),   // cout % CO != 0, stride 3 = block 4
        (3, 4, 5, 64),    // stride 5 = block 2, cin < CO
        (16, 4, 4, 256),  // stride 4 = block 3
    ] {
        let ks = 2 * stride;
        let t_out = t_in * stride;
        let x = rand_vec(cin * t_in, 94);
        let w = rand_vec(cin * cout * ks, 95);
        let b = rand_vec(cout, 96);
        let rows_total = cout * stride;
        let mut tmp = vec![0.0f32; rows_total * t_in];
        cpu_ops::conv_transpose_phased_co_outer(&x, &w, &b, &mut tmp, cin, cout, ks, stride, t_in);
        // Interleave exactly as conv_transpose1d_phased does.
        let mut y = vec![0.0f32; cout * t_out];
        for co in 0..cout {
            for m in 0..t_in {
                for q in 0..stride {
                    y[co * t_out + m * stride + q] = tmp[(co * stride + q) * t_in + m];
                }
            }
        }
        assert_transpose_close(&x, &w, &b, &y, cin, cout, ks, stride, t_in);
    }
}

/// Shared naive-gather comparison for both transposed-conv tests.
fn assert_transpose_close(x: &[f32], w: &[f32], b: &[f32], y: &[f32],
                          cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize) {
    let t_out = t_in * stride;
    let mut want = vec![0.0f32; cout * t_out];
    for co in 0..cout {
        for j in 0..t_out {
            let mut sum = b[co];
            for k in 0..ks {
                let num = j as isize - k as isize;
                if num < 0 || (num as usize) % stride != 0 {
                    continue;
                }
                let t = num as usize / stride;
                if t >= t_in {
                    continue;
                }
                for ci in 0..cin {
                    sum += x[ci * t_in + t] * w[(ci * cout + co) * ks + k];
                }
            }
            want[co * t_out + j] = sum;
        }
    }
    let mut worst = 0.0f32;
    for (a, c) in want.iter().zip(y) {
        worst = worst.max((a - c).abs());
    }
    eprintln!("transpose naive cin={cin} cout={cout} stride={stride} t_in={t_in}: max|diff|={worst:.3e}");
    assert!(worst < 5e-4, "transposed conv diff {worst}");
}

