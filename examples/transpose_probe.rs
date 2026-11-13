//! Transposed conv at the real block shapes, one backend path at a time.
use m5_tts_wgpu::cpu_ops;
use std::time::Instant;

fn main() {
    for (cin, cout, stride, t_in) in [(192usize, 96usize, 3usize, 36000usize), (1536, 768, 8, 228)] {
        let ks = 2 * stride;
        let x = vec![0.01f32; cin * t_in];
        let w = vec![0.01f32; cin * cout * ks];
        let b = vec![0.001f32; cout];
        let mut y = vec![0.0f32; cout * t_in * stride];
        cpu_ops::conv_transpose1d(&x, &w, &b, &mut y, cin, cout, ks, stride, t_in);
        let start = Instant::now();
        cpu_ops::conv_transpose1d(&x, &w, &b, &mut y, cin, cout, ks, stride, t_in);
        let s = start.elapsed().as_secs_f64();
        let macs = (cin * cout * ks * t_in) as f64;
        println!("transpose {cin}->{cout} s={stride} t={t_in}: {s:.3}s  {:.1} GFLOP/s", 2.0 * macs / s / 1e9);
    }
}
