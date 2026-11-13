//! Cost of the SnakeBeta sine at the last block's size (96 x 108000 elements).
use std::time::Instant;

fn fast_sin(x: f32) -> f32 {
    use std::f32::consts::PI;
    let t = x - (x * (1.0 / (2.0 * PI))).round() * (2.0 * PI);
    let t2 = t * t;
    t * (1.0 + t2 * (-1.0 / 6.0 + t2 * (1.0 / 120.0 + t2 * (-1.0 / 5040.0 + t2 / 362880.0))))
}

fn main() {
    let n = 96 * 108_000;
    let src: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.1).collect();
    let calls = 6; // one per fused conv in block 4

    let start = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..calls {
        for v in &src {
            acc += v.sin();
        }
    }
    println!("libm sin x{}: {:.3}s", calls, start.elapsed().as_secs_f64());
    let start = Instant::now();
    let mut acc2 = 0.0f32;
    for _ in 0..calls {
        for v in &src {
            acc2 += fast_sin(*v);
        }
    }
    println!("poly sin x{}: {:.3}s (diff {:.1e})", calls, start.elapsed().as_secs_f64(),
             (acc - acc2).abs());
}
