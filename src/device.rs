//! Device-independent tensors.
//!
//! The graph code (`codec`, `lm`) never touches wgpu or `Vec<f32>` directly: it
//! allocates through [`Device`] and transforms through [`crate::ops`]. Adding a
//! stage therefore costs one op, not one implementation per backend.

use anyhow::{bail, Result};
use rayon::prelude::*;

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
    /// `auto` (the default), `cpu` or `gpu`.
    ///
    /// `auto` takes the GPU when one is usable and falls back to the hand-written
    /// CPU backend otherwise, saying why: a machine without a working adapter should
    /// still synthesise, only slower. `cuda` is accepted as an alias for `gpu`
    /// because that is what the reference implementation calls its devices.
    pub fn open(spec: &str) -> Result<Self> {
        match spec {
            "cpu" => Ok(Device::Cpu),
            "gpu" | "cuda" => Ok(Device::Gpu(Rc::new(Gpu::new(None)?))),
            "auto" => match Gpu::new(None) {
                Ok(gpu) => Ok(Device::Gpu(Rc::new(gpu))),
                Err(err) => {
                    eprintln!("no usable GPU ({err:#}); falling back to the CPU backend");
                    Ok(Device::Cpu)
                }
            },
            other => bail!("unknown device {other:?}; use auto, cpu or gpu"),
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
            Device::Gpu(gpu) => Tensor::Gpu(GpuBuffer {
                raw: gpu.upload_pooled(data),
                gpu: Rc::clone(gpu),
                owned: true,
            }, shape),
            Device::Cpu => Tensor::Cpu(data.to_vec(), shape),
        })
    }

    /// Upload into a FRESH (non-pooled) buffer: for tensors that live for the
    /// whole run (weights, tables) - their buffers never re-enter the pool, so
    /// no later tensor can inherit their read-only usage history.
    pub fn tensor_fresh(&self, shape: [usize; 2], data: &[f32]) -> Result<Tensor> {
        if data.len() != shape[0] * shape[1] {
            bail!("tensor_fresh {:?} needs {} values, got {}", shape, shape[0] * shape[1], data.len());
        }
        Ok(match self {
            Device::Gpu(gpu) => Tensor::Gpu(GpuBuffer {
                raw: gpu.upload("tensor-fresh", data),
                gpu: Rc::clone(gpu),
                owned: false,
            }, shape),
            Device::Cpu => Tensor::Cpu(data.to_vec(), shape),
        })
    }

    /// A tensor whose contents are zero.
    ///
    /// GPU: pooled buffer + an on-device fill dispatch. It used to upload a
    /// host-side zero vec (`queue.write_buffer`), which at codec block-4 sizes
    /// pushed 72 MB per tensor over a x1 PCIe link; the fill kernel moves
    /// nothing. CPU: `vec![0.0; n]` is an `alloc_zeroed`, so the OS hands out
    /// lazily committed zero pages and the page faults land inside whatever
    /// writes them first.
    pub fn zeros(&self, shape: [usize; 2]) -> Result<Tensor> {
        match self {
            Device::Cpu => Ok(Tensor::Cpu(vec![0.0; shape[0] * shape[1]], shape)),
            Device::Gpu(gpu) => {
                let len = shape[0] * shape[1];
                let tensor = Tensor::Gpu(GpuBuffer {
                    raw: gpu.take_pooled(len),
                    gpu: Rc::clone(gpu),
                    owned: true,
                }, shape);
                crate::gpu_ops::fill(gpu, tensor.gpu_buffer()?, len, 0.0);
                Ok(tensor)
            }
        }
    }

    /// Copy a tensor back to the host.
    pub fn read(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        match (self, tensor) {
            (Device::Gpu(gpu), Tensor::Gpu(buffer, shape)) => Ok(gpu.download(&buffer.raw, shape[0] * shape[1])),
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
                // The tensor may have been created by `zeros`, whose fill is a
                // recorded dispatch: submit it first, or the host bytes land
                // before the zeroing kernel and get overwritten.
                gpu.flush();
                gpu.write(&buffer.raw, data);
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
        // GPU: the copy overwrites every element, so skip the zero fill that
        // `zeros` would dispatch first.
        let mut out = match self {
            Device::Gpu(gpu) => Tensor::Gpu(GpuBuffer {
                raw: gpu.take_pooled(tensor.numel()),
                gpu: Rc::clone(gpu),
                owned: true,
            }, tensor.shape()),
            Device::Cpu => Tensor::Cpu(vec![0.0; tensor.numel()], tensor.shape()),
        };
        match (self, tensor, &mut out) {
            (Device::Gpu(gpu), Tensor::Gpu(src, shape), Tensor::Gpu(dst, _)) => {
                crate::gpu_ops::copy(gpu, &src.raw, &dst.raw, shape[0] * shape[1]);
            }
            // Large CPU copies split across threads: the destination's pages are
            // uncommitted, and faulting them in from one thread is what made a
            // 72 MB copy take 36 ms while the memcpy itself needs ~7.
            (Device::Cpu, Tensor::Cpu(src, _), Tensor::Cpu(dst, _)) => {
                if src.len() >= 8192 {
                    dst.par_chunks_mut(8192)
                        .zip(src.par_chunks(8192))
                        .for_each(|(d, s)| d.copy_from_slice(s));
                } else {
                    dst.copy_from_slice(src);
                }
            }
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
/// A GPU buffer bound to its device so a dropped tensor returns the storage to
/// the device's pool instead of paying a Vulkan alloc/free cycle per op.
pub struct GpuBuffer {
    pub raw: wgpu::Buffer,
    pub gpu: Rc<Gpu>,
    /// false for handles sharing a persistent buffer: their drop must NOT
    /// recycle (the owner scratch still uses the buffer, and a pooled
    /// reassignment would create a read/write conflict in a later pass).
    pub owned: bool,
}

impl GpuBuffer {
    /// A fresh Tensor handle sharing this pooled buffer (cheap: handle clone).
    pub fn tensor(&self, shape: [usize; 2]) -> Tensor {
        Tensor::Gpu(GpuBuffer {
            raw: self.raw.clone(),
            gpu: Rc::clone(&self.gpu),
            owned: false,
        }, shape)
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.owned {
            self.gpu.recycle_on_drop(self.raw.size() as usize / 4, self.raw.clone());
        }
    }
}

pub enum Tensor {
    Gpu(GpuBuffer, [usize; 2]),
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

    /// A fresh GPU Tensor handle sharing this tensor's pooled buffer (cheap:
    /// handle clone; the buffer returns to the pool only when ALL handles drop).
    pub fn handle(&self, shape: [usize; 2]) -> Result<Tensor> {
        match self {
            Tensor::Gpu(buf, _) => Ok(buf.tensor(shape)),
            Tensor::Cpu(..) => bail!("handle() is GPU-only"),
        }
    }

    pub fn gpu_buffer(&self) -> Result<&wgpu::Buffer> {
        match self {
            Tensor::Gpu(buffer, _) => Ok(&buffer.raw),
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
