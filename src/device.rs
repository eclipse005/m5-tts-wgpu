//! Device-independent tensors.
//!
//! The graph code (`codec`, `lm`) never touches wgpu or `Vec<f32>` directly: it
//! allocates through [`Device`] and transforms through [`crate::ops`]. Adding a
//! stage therefore costs one op, not one implementation per backend.

use anyhow::{bail, Result};

use crate::gpu::Gpu;
use std::rc::Rc;

/// Which backend a run uses. `Cpu` is the hand-written f32 path (rayon), `Gpu`
/// is wgpu. Both are always compiled in so the parity tests can compare them.
pub enum Device {
    /// `Rc` so several models (LM + codec) can share one wgpu device: buffers are
    /// only valid on the device that created them.
    Gpu(Rc<Gpu>),
    Cpu,
}

impl Device {
    /// `--device gpu|cpu|cuda` (`cuda` is accepted as an alias for the GPU path
    /// because that is what the reference calls it).
    pub fn open(spec: &str) -> Result<Self> {
        match spec {
            "cpu" => Ok(Device::Cpu),
            "gpu" | "cuda" | "vulkan" | "default" => Ok(Device::Gpu(Rc::new(Gpu::new(None)?))),
            other => bail!("unknown device {other:?}; use cpu or gpu"),
        }
    }

    /// Another handle to the same device (`None`-like for CPU).
    pub fn shared(&self) -> Self {
        match self {
            Device::Gpu(gpu) => Device::Gpu(Rc::clone(gpu)),
            Device::Cpu => Device::Cpu,
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }

    pub fn describe(&self) -> String {
        match self {
            Device::Gpu(gpu) => gpu.describe(),
            Device::Cpu => format!("cpu ({} threads)", rayon::current_num_threads()),
        }
    }

    /// Allocate a tensor with the given 2-D shape and copy `data` in.
    pub fn tensor(&self, shape: [usize; 2], data: &[f32]) -> Result<Tensor> {
        if data.len() != shape[0] * shape[1] {
            bail!("tensor {:?} needs {} values, got {}", shape, shape[0] * shape[1], data.len());
        }
        Ok(match self {
            Device::Gpu(gpu) => Tensor::Gpu(gpu.upload("tensor", data), shape),
            Device::Cpu => Tensor::Cpu(data.to_vec(), shape),
        })
    }

    /// Uninitialised tensor (every op overwrites its output).
    pub fn zeros(&self, shape: [usize; 2]) -> Result<Tensor> {
        self.tensor(shape, &vec![0.0; shape[0] * shape[1]])
    }

    /// Copy a tensor back to the host.
    pub fn read(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        match (self, tensor) {
            (Device::Gpu(gpu), Tensor::Gpu(buffer, shape)) => Ok(gpu.download(buffer, shape[0] * shape[1])),
            (Device::Cpu, Tensor::Cpu(data, _)) => Ok(data.clone()),
            _ => bail!("tensor was created on a different device"),
        }
    }

    /// Copy host data into an existing tensor (used by streaming/caches).
    pub fn write(&self, tensor: &mut Tensor, data: &[f32]) -> Result<()> {
        match (self, tensor) {
            (Device::Gpu(gpu), Tensor::Gpu(buffer, shape)) => {
                if data.len() != shape[0] * shape[1] {
                    bail!("write size {} != tensor {:?}", data.len(), shape);
                }
                gpu.write(buffer, data);
                Ok(())
            }
            (Device::Cpu, Tensor::Cpu(host, shape)) => {
                if data.len() != shape[0] * shape[1] {
                    bail!("write size {} != tensor {:?}", data.len(), shape);
                }
                host.copy_from_slice(data);
                Ok(())
            }
            _ => bail!("tensor was created on a different device"),
        }
    }

    /// Token vector: a u32 storage buffer on the GPU, a plain vec on the CPU.
    pub fn tokens(&self, values: &[u32]) -> Result<Tokens> {
        Ok(match self {
            Device::Gpu(gpu) => Tokens {
                buffer: Some(gpu.upload_u32("tokens", values)),
                data: values.to_vec(),
            },
            Device::Cpu => Tokens { buffer: None, data: values.to_vec() },
        })
    }

    /// Read a token vector back (one synchronisation).
    pub fn read_tokens(&self, tokens: &Tokens) -> Result<Vec<u32>> {
        match (self, &tokens.buffer) {
            (Device::Gpu(gpu), Some(buffer)) => {
                let values = gpu.download_u32(buffer, tokens.data.len());
                Ok(values)
            }
            _ => Ok(tokens.data.clone()),
        }
    }

    /// Explicit tensor copy (device-to-device on the GPU, `Vec` clone on the CPU).
    pub fn copy(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut out = self.zeros(tensor.shape())?;
        match (self, tensor, &mut out) {
            (Device::Gpu(gpu), Tensor::Gpu(src, shape), Tensor::Gpu(dst, _)) => {
                crate::gpu_ops::copy(gpu, src, dst, shape[0] * shape[1]);
            }
            (Device::Cpu, Tensor::Cpu(src, _), Tensor::Cpu(dst, _)) => dst.copy_from_slice(src),
            _ => bail!("copy across devices"),
        }
        Ok(out)
    }

    pub fn gpu(&self) -> Result<&Gpu> {
        match self {
            Device::Gpu(gpu) => Ok(gpu.as_ref()),
            Device::Cpu => bail!("this operation needs the GPU backend"),
        }
    }
}

/// Token ids on the device (u32). Kept apart from [`Tensor`] because the
/// generation loop must sample and embed without a host round-trip.
pub struct Tokens {
    pub buffer: Option<wgpu::Buffer>,
    pub data: Vec<u32>,
}

impl Tokens {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A 2-D activation. Rows/columns mean whatever the stage says (channels x time
/// for the codec, tokens x hidden for the LM).
///
/// Deliberately **not** `Clone`: on the GPU the payload is a buffer handle, so a
/// derived clone would alias device memory and an "in-place" op would silently
/// overwrite its caller's tensor. Use [`Device::copy`] instead.
pub enum Tensor {
    Gpu(wgpu::Buffer, [usize; 2]),
    Cpu(Vec<f32>, [usize; 2]),
}

impl Tensor {
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Tensor::Gpu(_, shape) | Tensor::Cpu(_, shape) => *shape,
        }
    }

    pub fn numel(&self) -> usize {
        let [a, b] = self.shape();
        a * b
    }

    pub fn gpu_buffer(&self) -> Result<&wgpu::Buffer> {
        match self {
            Tensor::Gpu(buffer, _) => Ok(buffer),
            Tensor::Cpu(..) => bail!("expected a GPU tensor"),
        }
    }

    pub fn cpu_slice(&self) -> Result<&[f32]> {
        match self {
            Tensor::Cpu(data, _) => Ok(data),
            Tensor::Gpu(..) => bail!("expected a CPU tensor"),
        }
    }

    pub fn cpu_slice_mut(&mut self) -> Result<&mut [f32]> {
        match self {
            Tensor::Cpu(data, _) => Ok(data),
            Tensor::Gpu(..) => bail!("expected a CPU tensor"),
        }
    }
}
