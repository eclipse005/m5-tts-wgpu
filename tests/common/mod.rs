//! Golden tensors dumped from the Python reference (`scripts/dump_acoustic.py`).

use std::path::{Path, PathBuf};

pub struct Goldens {
    dir: PathBuf,
    index: serde_json::Value,
}

impl Goldens {
    /// `None` when the dump has not been produced yet, so the tests can skip
    /// instead of failing on a fresh clone.
    pub fn open() -> Option<Self> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens/acoustic");
        let index = std::fs::read_to_string(dir.join("index.json")).ok()?;
        Some(Self { dir, index: serde_json::from_str(&index).ok()? })
    }

    fn entry(&self, name: &str) -> Option<&serde_json::Value> {
        self.index.get("tensors")?.get(name)
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        self.entry(name)
            .and_then(|e| e.get("shape"))
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect())
            .unwrap_or_default()
    }

    pub fn f32(&self, name: &str) -> Vec<f32> {
        let bytes = std::fs::read(self.dir.join(format!("{name}.bin")))
            .unwrap_or_else(|e| panic!("golden {name}: {e}"));
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn i64(&self, name: &str) -> Vec<i64> {
        let bytes = std::fs::read(self.dir.join(format!("{name}.bin")))
            .unwrap_or_else(|e| panic!("golden {name}: {e}"));
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    /// Flatten a golden to `[rows, cols]`, asserting the element count.
    pub fn matrix(&self, name: &str) -> ([usize; 2], Vec<f32>) {
        let shape = self.shape(name);
        let data = self.f32(name);
        let cols = *shape.last().unwrap_or(&1);
        let rows = data.len() / cols.max(1);
        ([rows, cols], data)
    }
}

pub fn max_abs_diff(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len(), "length");
    want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

pub fn codec_dir() -> Option<PathBuf> {
    let dir = std::env::var("M5_CODEC")
        .unwrap_or_else(|_| r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer".to_string());
    let path = PathBuf::from(dir);
    path.join("model.safetensors").is_file().then_some(path)
}
