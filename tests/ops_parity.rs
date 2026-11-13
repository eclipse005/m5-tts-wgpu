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
