//! How much of the codec decode is allocation/zeroing?
fn main() {
    let n = 96 * 182400;
    let start = std::time::Instant::now();
    for _ in 0..24 {
        let v = vec![0.0f32; n];
        std::hint::black_box(&v);
    }
    println!("24 x {:.0} MB zeroed allocs: {:.3}s", n as f64 * 4.0 / 1e6,
             start.elapsed().as_secs_f64());
}
