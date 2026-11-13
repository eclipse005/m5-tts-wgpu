//! M5-TTS 50M in Rust: wgpu GPU inference and a hand-written CPU twin.
//!
//! Reference implementation: `ASLP-lab/M5-TTS` (PyTorch). Every stage here is
//! meant to be numerically traceable to that code — `scripts/dump_golden.py`
//! records the reference activations, `tests/parity_*.rs` compares against them.
//!
//! Layout
//! * [`config`]       — `config.json` / `conditioning.json`
//! * [`conditioning`] — the `<|caption|>…` prompt format
//! * [`tok`]          — `tokenizer.json` plus the eight control tokens
//! * [`tn`]           — the reference's compiled pynini automata (six languages)
//! * [`weights`]      — safetensors (f32 / bf16), preserved exactly
//! * [`device`]       — device-independent tensors
//! * [`ops`]          — the op seam every graph goes through
//! * [`gpu`]          — wgpu device with deferred submission
//! * [`shaders`]      — WGSL kernels
//!
//! Model graphs (`codec`, `lm`) are added on top of [`ops`] and never touch a
//! backend directly.

pub mod audio;
pub mod codec;
pub mod conditioning;
pub mod config;
pub mod cpu_ops;
pub mod device;
pub mod gpu;
pub mod lm;
pub mod gpu_ops;
pub mod ops;
pub mod pipeline;
pub mod shaders;
pub mod tn;
pub mod tok;
pub mod weights;
