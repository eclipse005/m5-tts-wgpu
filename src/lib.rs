//! M5-TTS 50M in Rust: wgpu GPU inference and a hand-written CPU twin.
//!
//! ```no_run
//! use m5_tts_wgpu::pipeline::{Args, Pipeline};
//!
//! let pipeline = Pipeline::load("M5-TTS-50M-CustomVoice".as_ref(),
//!                               "M5-TTS-Audio-Tokenizer".as_ref(), "auto")?;
//! let args = Args { text: "你好，今天的天气很好。".into(), ..Args::default() };
//! let out = pipeline.synthesize(&args)?;
//! m5_tts_wgpu::audio::write_wav("hello.wav".as_ref(), &out.audio, out.sample_rate)?;
//! # anyhow::Ok(())
//! ```
//!
//! Layout
//! * [`pipeline`]    — [`Pipeline`], [`Args`], [`Synthesis`]: load once, synthesise
//! * [`config`]      — `config.json` / `conditioning.json` of a checkpoint bundle
//! * [`conditioning`] — the `<|caption|>…` prompt format
//! * [`tok`]         — `tokenizer.json` plus the eight control tokens
//! * [`tn`]          — the reference's compiled pynini automata (six languages)
//! * [`long_audio`]  — semantic-length text splitting for long scripts
//! * [`weights`]     — safetensors (f32 / bf16), preserved exactly
//! * [`device`]      — device-independent tensors and the `auto|cpu|gpu` choice
//! * [`gpu`]         — wgpu device with deferred submission
//! * [`ops`]         — the op seam every graph goes through, plus [`cpu_ops`]
//! * [`shaders`]     — WGSL kernels
//! * [`codec`], [`lm`] — the speech codec and the language model on top of [`ops`]

// Kernel launchers take one argument per tensor and scalar. Naming them at the call
// site (and mirroring the WGSL bindings) is worth more than a params struct, which
// every caller would have to build anyway.
#![allow(clippy::too_many_arguments)]
// Float literals keep the precision they were measured or compared at.
#![allow(clippy::excessive_precision)]

pub mod audio;
pub mod codec;
pub mod conditioning;
pub mod config;
pub mod cpu_ops;
pub mod device;
pub mod gpu;
pub mod gpu_ops;
pub mod lm;
pub mod long_audio;
pub mod opencc;
pub mod ops;
pub mod pipeline;
pub mod shaders;
pub mod tn;
pub mod tok;
pub mod weights;

pub use device::Device;
pub use pipeline::{Args, Pipeline, Synthesis};
