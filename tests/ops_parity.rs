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
    for (cin, cout, rate) in [(4usize, 2usize, 2usize), (6, 3, 8), (8, 4, 5)] {
        let ks = 2 * rate;
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
