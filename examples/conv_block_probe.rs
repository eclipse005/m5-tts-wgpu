//! Conv1d at the four decoder-block shapes the codec actually hits, so a
//! staging/occupancy change can be A/B'd without a full decode.
//!
//!   cargo run --release --example conv_block_probe
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    // (channels, frames) of each decoder block's residual units, in the order
    // the decode runs them.
    let shapes = [(768usize, 2_000usize), (384, 9_000), (192, 36_000), (96, 108_000)];
    let mut total = 0.0f64;
    for (channels, frames) in shapes {
        let x = dev.tensor([frames, channels], &vec![0.01f32; frames * channels])?;
        let b = dev.tensor([channels, 1], &vec![0.001f32; channels])?;
        let y = dev.zeros([frames, channels])?;
        for (ks, dilation) in [(7usize, 1usize), (7, 3), (7, 9), (1, 1)] {
            let w = dev.tensor([ks * channels, channels], &vec![0.01f32; ks * channels * channels])?;
            let macs = (channels * channels * ks * frames) as f64;
            let run = || gpu_ops::conv1d_reg8(
                gpu, x.gpu_buffer().unwrap(), w.gpu_buffer().unwrap(), b.gpu_buffer().unwrap(),
                y.gpu_buffer().unwrap(), channels, channels, ks, dilation, 1,
                (ks - 1) * dilation, frames, frames);
            run();
            gpu.sync();
            let start = Instant::now();
            let iters = 5;
            for _ in 0..iters { run(); }
            gpu.sync();
            let secs = start.elapsed().as_secs_f64() / iters as f64;
            println!("{channels:4}ch x {frames:6}  ks{ks} dil{dilation}  {:7.2} ms  {:5.0} GFLOP/s",
                     secs * 1e3, 2.0 * macs / secs / 1e9);
            if ks == 7 { total += secs * 3.0; } else { total += secs * 3.0; }
        }
    }
    // Three residual units per block, one conv7 + one conv1 each.
    println!("block conv time (3 units/block): {total:.3}s");
    Ok(())
}
