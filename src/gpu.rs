//! wgpu device setup with deferred submission.
//!
//! One dispatch = one encoder + one submit is what kills per-frame decode on
//! WDDM: a single frame is hundreds of small dispatches. Recording into a batch
//! and submitting at readback (or at a size cap) is the discipline the other
//! ports converged on.

use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupEntry, Buffer, BufferUsages, Device, Queue};

/// One recorded dispatch plus everything the command buffer still points at.
///
/// wgpu does not keep a bind group's buffers alive for a *recorded* command
/// buffer, so dropping them before the submit lets Vulkan free them underneath
/// ("Cannot get non-existent resource"). They are held here until after submit.

/// One recorded dispatch, replayed into a single compute pass at flush time.
///
/// Recording a pass (or an encoder) per dispatch makes every kernel pay the full
/// pipeline-drain latency; on this backend that dominated everything (measured
/// 20.5 us per trivial dispatch). WebGPU guarantees each dispatch sees the
/// previous one's writes inside a pass, so a single pass preserves ordering.
struct Pending {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    workgroups: (u32, u32, u32),
}

pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
    pub adapter_name: String,
    pub backend: &'static str,
    max_pending: usize,
    pending: RefCell<Vec<Pending>>,
    pipelines: RefCell<HashMap<String, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>>,
    pub stats: RefCell<Stats>,
    /// Upload bytes per call-site label, so a profiling run can show where the
    /// PCIe traffic actually comes from (`M5_PROFILE=1`).
    pub upload_labels: RefCell<HashMap<String, u64>>,
    /// Parameter-block pool for per-dispatch params: many small buffers bound
    /// whole. A ring of distinct slices on one buffer made wgpu's resource
    /// tracking superlinear (28 us/dispatch); whole bindings on reused buffers
    /// stay on the fast path (~7 us). The pool flushes when it wraps.
    params_pool: RefCell<ParamsPool>,
}

/// A parameter block from the pool: bound whole, offset always 0.
#[derive(Clone)]
pub struct Params {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
}

struct ParamsPool {
    buffers: Vec<Buffer>,
    next: usize,
}



#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    pub dispatches: u64,
    pub submits: u64,
    pub bytes_uploaded: u64,
    pub bytes_readback: u64,
}

impl Gpu {
    /// Snapshot of the counters (dispatches, submits, transfer bytes).
    pub fn snapshot(&self) -> Stats {
        *self.stats.borrow()
    }

    /// The raw wgpu device (diagnostics / probes).
    pub fn raw_device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Block until everything queued so far has executed. `flush` alone only
    /// submits — profiling laps need the wait too, or elapsed measures command
    /// recording, not execution.
    pub fn sync(&self) {
        self.flush();
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    }
}

fn backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "webgpu",
        _ => "other",
    }
}

impl Gpu {
    pub fn new(prefer: Option<&str>) -> Result<Self> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::GL | wgpu::Backends::METAL;
        let backends = desc.backends;
        let instance = wgpu::Instance::new(desc);
        let mut adapters = pollster::block_on(instance.enumerate_adapters(backends));
        if adapters.is_empty() {
            bail!("no wgpu adapter (Vulkan/GL/Metal) found");
        }
        if let Some(want) = prefer {
            let want = want.to_lowercase();
            adapters.sort_by_key(|a| !a.get_info().name.to_lowercase().contains(&want));
        } else {
            // The LM is latency bound, the codec is ~300 GFLOP, so prefer the
            // fastest device rather than the first one enumerated.
            adapters.sort_by_key(|a| match a.get_info().device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                _ => 2,
            });
        }
        let adapter = &adapters[0];
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("m5-tts"),
                    required_features: wgpu::Features::empty(),
                    // Take what the adapter offers: the codec wants large storage
                    // bindings and bigger workgroups than wgpu's baseline.
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
        })
        .context("request wgpu device")?;
        // Windows resets the device (TDR) if a single submit runs longer than
        // ~2 s, and one SEANet block is already hundreds of dispatches on a slow
        // kernel. 8 keeps the worst-case submit well inside the watchdog;
        // `M5_MAX_PENDING` overrides for experiments.
        let max_pending = std::env::var("M5_MAX_PENDING")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        Ok(Self {
            device,
            queue,
            adapter_name: info.name.clone(),
            backend: backend_name(info.backend),
            max_pending,
            pending: RefCell::new(Vec::new()),
            pipelines: RefCell::new(HashMap::new()),
            stats: RefCell::new(Stats::default()),
            upload_labels: RefCell::new(HashMap::new()),
            params_pool: RefCell::new(ParamsPool { buffers: Vec::new(), next: 0 }),
        })
    }

    pub fn describe(&self) -> String {
        format!("{} ({})", self.adapter_name, self.backend)
    }

    fn storage_usage() -> BufferUsages {
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
    }

    /// Allocate a storage buffer and upload `data`.
    pub fn upload(&self, label: &str, data: &[f32]) -> Buffer {
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: Self::storage_usage(),
        });
        self.stats.borrow_mut().bytes_uploaded += (data.len() * 4) as u64;
        *self.upload_labels.borrow_mut().entry(label.to_string()).or_insert(0) += (data.len() * 4) as u64;
        buffer
    }

    /// Allocate an uninitialised storage buffer for `len` f32.
    pub fn empty(&self, label: &str, len: usize) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len.max(1) * 4) as u64,
            usage: Self::storage_usage(),
            mapped_at_creation: false,
        })
    }

    /// Params block (u32 words) for a kernel.
    pub fn upload_u32(&self, label: &str, data: &[u32]) -> Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: Self::storage_usage(),
        })
    }


    /// Carve a parameter block from the pool: small buffers bound whole. A ring
    /// of distinct slices on one buffer made wgpu's resource tracking
    /// superlinear (28 us/dispatch); whole bindings on reused buffers stay on
    /// the fast path. The pool flushes when it wraps so no pending dispatch can
    /// see its params overwritten.
    pub fn params_u32(&self, data: &[u32]) -> Params {
        const POOL: usize = 1024;
        let size = (data.len() * 4) as u64;
        let mut pool = self.params_pool.borrow_mut();
        if pool.buffers.is_empty() {
            for i in 0..POOL {
                pool.buffers.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("params.{i}")),
                    size: 64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
        }
        if pool.next >= POOL {
            self.flush();
            pool.next = 0;
        }
        let buffer = pool.buffers[pool.next].clone();
        pool.next += 1;
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        self.stats.borrow_mut().bytes_uploaded += size;
        Params { buffer, offset: 0, size }
    }

    /// dispatch() with the params block bound as a ring slice.
    pub fn dispatch_params(&self, key: &str, shader: &str, entry: &str,
                           bindings: &[(u32, &Buffer)], params: &Params,
                           workgroups: (u32, u32, u32)) {
        let (pipeline, layout) = self.pipeline_with_layout(key, shader, entry);
        let mut entries: Vec<BindGroupEntry<'_>> = bindings
            .iter()
            .map(|(binding, buffer)| BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(BindGroupEntry {
            binding: entries.len() as u32,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &params.buffer,
                offset: params.offset,
                size: Some(params.size.try_into().expect("non-zero params")),
            }),
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &layout,
            entries: &entries,
        });
        self.reuse(key, &pipeline, &group, workgroups);
    }

    /// Upload into an existing buffer (uniforms / small params).
    pub fn write(&self, buffer: &Buffer, data: &[f32]) {
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        self.stats.borrow_mut().bytes_uploaded += (data.len() * 4) as u64;
    }

    pub fn pipeline(&self, key: &str, shader: &str, entry: &str) -> wgpu::ComputePipeline {
        self.pipeline_with_layout(key, shader, entry).0
    }

    /// Pipeline + its bind group layout, both cached: `get_bind_group_layout`
    /// costs ~15 us per call and sat inside every dispatch (gpu_probe a/b/e).
    pub fn pipeline_with_layout(&self, key: &str, shader: &str, entry: &str)
                                -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        if let Some(pair) = self.pipelines.borrow().get(key) {
            return pair.clone();
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        // `layout: None` derives the bind group layout from the shader, so the
        // kernels do not have to keep a second copy of their binding contract.
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(key),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        let layout = pipeline.get_bind_group_layout(0);
        self.pipelines
            .borrow_mut()
            .insert(key.to_string(), (pipeline.clone(), layout.clone()));
        (pipeline, layout)
    }

    /// dispatch_params with a pre-built pipeline + layout: probe variant.
    pub fn reuse_with_slice(&self, _key: &str, pipeline: &wgpu::ComputePipeline,
                            layout: &wgpu::BindGroupLayout, bindings: &[(u32, &Buffer)],
                            params: &Params, workgroups: (u32, u32, u32)) {
        let mut entries: Vec<BindGroupEntry<'_>> = bindings
            .iter()
            .map(|(binding, buffer)| BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(BindGroupEntry {
            binding: entries.len() as u32,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &params.buffer,
                offset: params.offset,
                size: Some(params.size.try_into().expect("non-zero params")),
            }),
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe-slice"),
            layout,
            entries: &entries,
        });
        self.reuse(_key, pipeline, &group, workgroups);
    }

    /// Record a dispatch from a pre-built pipeline + bind group: the probe path
    /// that separates object-creation cost from plain recording.
    pub fn reuse(&self, _key: &str, pipeline: &wgpu::ComputePipeline,
                 group: &wgpu::BindGroup, workgroups: (u32, u32, u32)) {
        self.pending.borrow_mut().push(Pending {
            pipeline: pipeline.clone(),
            group: group.clone(),
            workgroups,
        });
        self.stats.borrow_mut().dispatches += 1;
        if self.pending.borrow().len() >= self.max_pending {
            self.flush();
        }
    }

    /// Record one dispatch into the current batch.
    pub fn dispatch(&self, key: &str, shader: &str, entry: &str, bindings: &[(u32, &Buffer)],
                    workgroups: (u32, u32, u32)) {
        let pipeline = self.pipeline(key, shader, entry);
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<BindGroupEntry<'_>> = bindings
            .iter()
            .map(|(binding, buffer)| BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &layout,
            entries: &entries,
        });
        self.stats.borrow_mut().dispatches += 1;
        self.pending.borrow_mut().push(Pending { pipeline, group, workgroups });
        if self.pending.borrow().len() >= self.max_pending {
            self.flush();
        }
    }


    /// Record the pending batch into one pass. `None` when nothing is pending.
    fn encode_batch(&self) -> Option<wgpu::CommandBuffer> {
        let batch = std::mem::take(&mut *self.pending.borrow_mut());
        if batch.is_empty() {
            return None;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            for pending in &batch {
                pass.set_pipeline(&pending.pipeline);
                pass.set_bind_group(0, &pending.group, &[]);
                pass.dispatch_workgroups(pending.workgroups.0, pending.workgroups.1,
                                         pending.workgroups.2);
            }
        }
        self.stats.borrow_mut().submits += 1;
        // Keeps the bind groups (and their buffers) alive until the GPU is done.
        std::mem::forget(batch);
        Some(encoder.finish())
    }

    /// Submit everything recorded so far.
    pub fn flush(&self) {
        if let Some(batch) = self.encode_batch() {
            self.queue.submit(std::iter::once(batch));
        }
    }

    /// Read `len` u32 back from `buffer`.
    pub fn download_u32(&self, buffer: &Buffer, len: usize) -> Vec<u32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_u32"),
            size: (len.max(1) * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback_u32") });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (len * 4) as u64);
        let mut commands: Vec<wgpu::CommandBuffer> = Vec::new();
        if let Some(batch) = self.encode_batch() {
            commands.push(batch);
        }
        commands.push(encoder.finish());
        self.queue.submit(commands);
        self.stats.borrow_mut().submits += 1;
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
        let out = {
            let view = staging.slice(..).get_mapped_range().expect("map readback range");
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        staging.unmap();
        self.stats.borrow_mut().bytes_readback += (len * 4) as u64;
        out
    }

    /// Read `len` f32 back from `buffer`.
    pub fn download(&self, buffer: &Buffer, len: usize) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (len.max(1) * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (len * 4) as u64);
        let mut commands: Vec<wgpu::CommandBuffer> = Vec::new();
        if let Some(batch) = self.encode_batch() {
            commands.push(batch);
        }
        commands.push(encoder.finish());
        self.queue.submit(commands);
        self.stats.borrow_mut().submits += 1;
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
        let out = {
            let view = staging.slice(..).get_mapped_range().expect("map readback range");
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        staging.unmap();
        self.stats.borrow_mut().bytes_readback += (len * 4) as u64;
        out
    }
}
