//! wgpu device setup with deferred submission.
//!
//! One dispatch = one encoder + one submit is what kills per-frame decode on
//! WDDM: a single frame is hundreds of small dispatches. Recording into a batch
//! and submitting at readback (or at a size cap) is the discipline the other
//! ports converged on.

use anyhow::{bail, Context, Result};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupEntry, Buffer, BufferUsages, Device, Queue};

/// One recorded dispatch plus everything the command buffer still points at.
///
/// wgpu does not keep a bind group's buffers alive for a *recorded* command
/// buffer, so dropping them before the submit lets Vulkan free them underneath
/// ("Cannot get non-existent resource"). They are held here until after submit.

/// An arena kernel: explicit layout + bind-group cache (ming nn.rs Kernel).
pub struct Kernel {
    pub bgl: wgpu::BindGroupLayout,
    pub pipe: wgpu::ComputePipeline,
    kinds: Vec<bool>,
    name: String,
    /// key -> (bind group, its own params buffer). The params buffer is written
    /// per dispatch (queue.write_buffer, staged with the batch that reads it)
    /// and bound whole at offset 0 - no arena slots, no dynamic offsets, no
    /// cross-pass usage-state reuse.
    cache: RefCell<HashMap<u64, (wgpu::BindGroup, Buffer)>>,
}

impl Kernel {
    /// Dispatch with a cached bind group under `key`. The caller must keep the
    /// same buffer set for a key; only the uniform bytes change per dispatch.
    pub fn run(&self, gpu: &Gpu, key: u64, uni: &[u32], bufs: &[&Buffer],
               workgroups: (u32, u32, u32)) {
        self.run_impl(gpu, key, uni, bufs, workgroups, false);
    }

    /// Same, but the params bytes are uploaded as the LAST binding (a storage
    /// buffer with a dynamic offset) instead of the uniform arena. Lets the
    /// existing storage-params shaders run through cached bind groups with
    /// zero shader edits; the bytes land in a per-params ring buffer whose
    /// write_buffer is queued with the batch that reads it.
    pub fn run_params(&self, gpu: &Gpu, key: u64, uni: &[u32], bufs: &[&Buffer],
                      workgroups: (u32, u32, u32)) {
        self.run_impl(gpu, key, uni, bufs, workgroups, true);
    }

    fn run_impl(&self, gpu: &Gpu, key: u64, uni: &[u32], bufs: &[&Buffer],
                workgroups: (u32, u32, u32), _params_dyn: bool) {
        let hit = self.cache.borrow().get(&key).cloned();
        let (bind, params_buffer) = if let Some(pair) = hit {
            pair
        } else {
            let mut entries = Vec::with_capacity(bufs.len() + 1);
            for (i, b) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                });
            }
            let params_buffer = gpu.empty(&format!("kernel.{}.params", self.name), 16);
            entries.push(wgpu::BindGroupEntry {
                binding: bufs.len() as u32,
                resource: params_buffer.as_entire_binding(),
            });
            let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("kernel.{}", self.name)),
                layout: &self.bgl,
                entries: &entries,
            });
            self.cache.borrow_mut().insert(key, (bind.clone(), params_buffer.clone()));
            (bind, params_buffer)
        };
        // The params bytes land with the batch that reads them (write_buffer is
        // staged per submit), and each (layer, op) owns its buffer exclusively.
        gpu.write_u32(&params_buffer, uni);
        gpu.record(&self.pipe, &bind, workgroups);
    }
}

/// One recorded dispatch, replayed into a single compute pass at flush time./// One recorded dispatch, replayed into a single compute pass at flush time.
///
/// Recording a pass (or an encoder) per dispatch makes every kernel pay the full
/// pipeline-drain latency; on this backend that dominated everything (measured
/// 20.5 us per trivial dispatch). WebGPU guarantees each dispatch sees the
/// previous one's writes inside a pass, so a single pass preserves ordering.
struct Pending {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    /// Some(offset): the bind group has one dynamic uniform binding (arena
    /// params). None: no dynamic bindings.
    dyn_offset: Option<u32>,
    workgroups: (u32, u32, u32),
    label: String,
}

/// Submits kept in flight before a blocking wait (ming-3.0-tts: 4 overlaps
/// CPU recording with GPU execution; deeper queues flake on the P104).
const MAX_IN_FLIGHT: u32 = 4;
/// Uniform slots are 256 B: the guaranteed minUniformBufferOffsetAlignment.
pub const UNIFORM_SLOT: usize = 256;
const ARENA_SLOTS: usize = 4096;
/// Dispatches allowed in the very first submit of the process (P104 faults
/// the device above ~50 in a cold first command buffer).
const FIRST_SUBMIT_MAX: usize = 8;

pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
    pub adapter_name: String,
    pub backend: &'static str,
    max_pending: usize,
    pending: RefCell<Vec<Pending>>,
    pipelines: RefCell<HashMap<String, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>>,
    /// Pipelines not yet exercised by a completed submit: the driver compiles
    /// them lazily at first use, and a batch full of new pipelines + a resident
    /// encoder faults the P104 (ming-3.0-tts gpu.rs cold-buffer notes). Such
    /// dispatches flush and drain immediately.
    unprimed: RefCell<HashMap<String, ()>>,
    pub stats: RefCell<Stats>,
    /// Upload bytes per call-site label, so a profiling run can show where the
    /// PCIe traffic actually comes from (`M5_PROFILE=1`).
    pub upload_labels: RefCell<HashMap<String, u64>>,
    /// Parameter arena: ONE storage buffer, 256 B slots selected per dispatch
    /// via a dynamic bind-group offset. Cached bind groups bind it whole, so
    /// every dispatch reads its own slot (ming-3.0-tts uniform-arena pattern,
    /// in storage address space so the existing shaders need no edits).
    params_arena: Buffer,
    params_host: RefCell<Vec<u8>>,
    params_used: Cell<usize>,
    /// Submits in flight without a wait: a few queued submits let the CPU
    /// record the next batch while the GPU drains the previous one
    /// (ming-3.0-tts proved 4 is the sweet spot on this exact card).
    in_flight: Cell<u32>,
    /// The first submit of the process is capped (FIRST_SUBMIT_MAX) and
    /// drained: the P104 faults the whole device when the first submit
    /// carries more than ~50 dispatches (nvlddmkm 153 -> device lost).
    primed: Cell<bool>,
    /// Scratch-buffer pool keyed by element count: codec ops allocate 36-72 MB
    /// outputs per call, and a fresh Vulkan buffer per op cost ~70 ms each once
    /// the allocator fragmented. Dropped tensors return here.
    pool: RefCell<HashMap<usize, Vec<Buffer>>>,
    pool_bytes: RefCell<usize>,
    pool_counter: Cell<usize>,
    submit_count: Cell<usize>,
    /// Buffers dropped while a submit batch was still open: they may be
    /// referenced by recorded dispatches, so they only join the pool after the
    /// next flush (write_buffer semantics make post-submit rewrites safe).
    deferred_pool: RefCell<Vec<(usize, Buffer)>>,
    /// Uniform arena: one buffer, 256 B slots selected per dispatch via a
    /// dynamic bind-group offset. Params cost one host memcpy per dispatch
    /// instead of a buffer allocation (ming-3.0-tts gpu.rs:267/398).
    arena: Buffer,
    arena_host: RefCell<Vec<u8>>,
    arena_used: Cell<usize>,
}

/// A parameter block from the pool: bound whole, offset always 0.
#[derive(Clone)]
pub struct Params {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
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

    /// The uniform arena buffer (bound whole by arena kernels).
    pub fn arena_buffer(&self) -> &Buffer {
        &self.arena
    }

    /// Block until everything queued so far has executed. `flush` alone only
    /// submits — profiling laps need the wait too, or elapsed measures command
    /// recording, not execution.
    pub fn sync(&self) {
        self.flush();
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    }

    fn drain_deferred(&self) {
        let mut deferred = self.deferred_pool.borrow_mut();
        if deferred.is_empty() { return; }
        for (len, buf) in deferred.drain(..) {
            self.recycle(len, buf);
        }
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
        // 64: large enough that submit overhead amortises, small enough that a
        // pass ends before pooled activations are read-then-written inside it
        // (wgpu forbids RO->RW on one buffer within a pass) and before VRAM
        // fills with deferred recycling.
        let max_pending = std::env::var("M5_MAX_PENDING")
            .ok()
            .and_then(|v| v.parse().ok())
            // 256 (ming-3.0-tts, same card): >=1024-record submits lose the
            // device to TDR during lazy pipeline compilation; 256 amortises
            // submit overhead without that.
            .unwrap_or(64);
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform-arena"),
            size: (ARENA_SLOTS * UNIFORM_SLOT) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params-arena"),
            size: (ARENA_SLOTS * UNIFORM_SLOT) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            device,
            queue,
            adapter_name: info.name.clone(),
            backend: backend_name(info.backend),
            max_pending,
            pending: RefCell::new(Vec::new()),
            pipelines: RefCell::new(HashMap::new()),
            unprimed: RefCell::new(HashMap::new()),
            stats: RefCell::new(Stats::default()),
            upload_labels: RefCell::new(HashMap::new()),
            params_arena,
            params_host: RefCell::new(Vec::new()),
            params_used: Cell::new(0),
            in_flight: Cell::new(0),
            primed: Cell::new(false),
            pool: RefCell::new(HashMap::new()),
            pool_bytes: RefCell::new(0),
            pool_counter: Cell::new(0),
            submit_count: Cell::new(0),
            deferred_pool: RefCell::new(Vec::new()),
            arena,
            arena_host: RefCell::new(Vec::new()),
            arena_used: Cell::new(0),
        })
    }

    pub fn describe(&self) -> String {
        format!("{} ({})", self.adapter_name, self.backend)
    }

    fn storage_usage() -> BufferUsages {
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
    }

    /// Take a pooled buffer of exactly `len` f32, or allocate one.
    pub fn take_pooled(&self, len: usize) -> Buffer {
        // Small buffers skip the pool: the LM decode re-reads buffers (RO) in a
        // later pass after writing them (RW) in an earlier one, and wgpu-core's
        // per-buffer usage state across submits flags the recycled buffer's
        // new RW binding against its stale RO one. Fresh small buffers cost
        // ~2 us; the codec's multi-MB buffers keep pooling (their flow is
        // write->read only).
        if len <= 4096 || std::env::var_os("M5_NO_POOL_REUSE").is_some() {
            let n = self.pool_counter.get();
            self.pool_counter.set(n + 1);
            return self.empty(&format!("fresh.{n}.{len}"), len);
        }
        if let Some(buf) = self.pool.borrow_mut().get_mut(&len).and_then(Vec::pop) {
            *self.pool_bytes.borrow_mut() -= len * 4;
            return buf;
        }
        let n = self.pool_counter.get();
        self.pool_counter.set(n + 1);
        self.empty(&format!("pooled.{n}.{len}"), len)
    }

    /// Return a dropped buffer to the pool (capped: ~1.5 GB of scratch).
    pub fn recycle(&self, len: usize, buf: Buffer) {
        const CAP_BYTES: usize = 512_000_000;
        let mut pool = self.pool.borrow_mut();
        let slot = pool.entry(len).or_default();
        if slot.len() < 4 && *self.pool_bytes.borrow() + len * 4 <= CAP_BYTES {
            *self.pool_bytes.borrow_mut() += len * 4;
            slot.push(buf);
        }
    }

    /// Recycle on tensor drop: a buffer recorded in the open submit batch is
    /// deferred to after the flush (rewrites before submit would corrupt it).
    pub fn recycle_on_drop(&self, len: usize, buf: Buffer) {
        if std::env::var_os("M5_NO_POOL").is_some() { return; }
        if self.pending.borrow().is_empty() {
            self.recycle(len, buf);
        } else {
            self.deferred_pool.borrow_mut().push((len, buf));
        }
    }

    /// Allocate-or-pool a storage buffer and upload `data`. Weights hit this
    /// once and stay cached; pooled buffers only help per-op scratch tensors.
    pub fn upload_pooled(&self, data: &[f32]) -> Buffer {
        let buffer = self.take_pooled(data.len());
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        self.stats.borrow_mut().bytes_uploaded += (data.len() * 4) as u64;
        buffer
    }

    /// Allocate a storage buffer and upload `data`.
    ///
    /// create + queue.write_buffer, NOT mapped_at_creation: the P104's driver
    /// fails the creation-time map under memory pressure/fragmentation
    /// ("Buffer is invalid" at get_mapped_range); the staged write_buffer path
    /// is what ming-3.0-tts uses on the same card.
    pub fn upload(&self, label: &str, data: &[f32]) -> Buffer {
        let buffer = self.storage((data.len() * 4) as u64, label);
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        self.stats.borrow_mut().bytes_uploaded += (data.len() * 4) as u64;
        *self.upload_labels.borrow_mut().entry(label.to_string()).or_insert(0) += (data.len() * 4) as u64;
        buffer
    }

    /// The raw storage allocation (create_buffer, no mapping).
    pub fn storage(&self, size: u64, label: &str) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: Self::storage_usage(),
            mapped_at_creation: false,
        })
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


    /// Reserve a uniform slot, copying the bytes into the arena's host staging
    /// area. Returns the byte offset for the dynamic bind-group offset. The
    /// arena uploads once per flush (one write_buffer for all slots).
    pub fn alloc_uniform(&self, data: &[u8]) -> u32 {
        debug_assert!(data.len() <= UNIFORM_SLOT);
        let mut used = self.arena_used.get();
        if used + UNIFORM_SLOT > ARENA_SLOTS * UNIFORM_SLOT {
            self.flush();
            used = 0;
        }
        let mut host = self.arena_host.borrow_mut();
        if host.len() < used + UNIFORM_SLOT {
            host.resize(used + UNIFORM_SLOT, 0);
        }
        host[used..used + data.len()].copy_from_slice(data);
        self.arena_used.set(used + UNIFORM_SLOT);
        used as u32
    }

    /// Params slot as a standalone Params (static binding, offset/size baked
    /// in) - for the one-shot dispatch_params path.
    pub fn params_u32(&self, data: &[u32]) -> Params {
        let (buffer, offset, size) = self.alloc_params(data);
        Params { buffer, offset, size }
    }

    /// Carve a parameter slot from the shared storage arena. Returns the byte
    /// offset for the dynamic bind-group offset; the arena uploads once per
    /// flush (one write_buffer for all slots).
    pub fn alloc_params(&self, data: &[u32]) -> (Buffer, u64, u64) {
        const ALIGN: u64 = 32;  // min_storage_buffer_offset_alignment
        let size = (data.len() * 4) as u64;
        let cap = (ARENA_SLOTS * UNIFORM_SLOT) as u64;
        let mut used = self.params_used.get();
        if used as u64 + size + ALIGN > cap {
            self.flush();
            used = 0;
        }
        let offset = (used as u64 + ALIGN - 1) / ALIGN * ALIGN;
        let mut host = self.params_host.borrow_mut();
        if host.len() < offset as usize + size as usize {
            host.resize(offset as usize + size as usize, 0);
        }
        host[offset as usize..offset as usize + size as usize]
            .copy_from_slice(bytemuck::cast_slice(data));
        self.params_used.set(offset as usize + size as usize);
        (self.params_arena.clone(), offset, size)
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

    /// Upload u32 params into an existing buffer.
    pub fn write_u32(&self, buffer: &Buffer, data: &[u32]) {
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        self.stats.borrow_mut().bytes_uploaded += (data.len() * 4) as u64;
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
        self.unprimed.borrow_mut().insert(key.to_string(), ());
        (pipeline, layout)
    }

    /// Drain now if the given pipeline has never been through a submit: its
    /// first dispatch triggers the driver's lazy compilation, which must not
    /// pile up with other new pipelines in one batch on the P104.
    fn prime_if_needed(&self, key: &str) {
        if self.unprimed.borrow_mut().remove(key).is_some() {
            self.flush();
            self.device.poll(wgpu::PollType::wait_indefinitely()).expect("prime poll");
            self.in_flight.set(0);
        }
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

    /// Flush policy: the P104's cold-start window faults the device when the
    /// first submits carry too many dispatches (the threshold ~50, dropping to
    /// ~16 with the encoder resident - ming-3.0-tts gpu.rs). The submit size
    /// ramps 8 -> 16 -> 32 -> 64 -> 128 -> max_pending over the first six
    /// submits, each drained, before steady state.
    fn maybe_flush(&self) {
        let n = self.pending.borrow().len();
        let s = self.submit_count.get();
        let ramp = [FIRST_SUBMIT_MAX as usize, 16, 32, 64, 128];
        let cap = if s < ramp.len() { ramp[s] } else { self.max_pending };
        if !self.primed.get() {
            if n >= cap {
                self.primed.set(true);
                self.flush();
                self.device.poll(wgpu::PollType::wait_indefinitely()).expect("prime poll");
                self.in_flight.set(0);
            }
        } else if n >= cap {
            self.flush();
        }
    }

    /// Record a dispatch from a pre-built pipeline + bind group: the probe path
    /// that separates object-creation cost from plain recording.
    pub fn reuse(&self, _key: &str, pipeline: &wgpu::ComputePipeline,
                 group: &wgpu::BindGroup, workgroups: (u32, u32, u32)) {
        self.pending.borrow_mut().push(Pending {
            pipeline: pipeline.clone(),
            group: group.clone(),
            dyn_offset: None,
            workgroups,
            label: _key.to_string(),
        });
        self.stats.borrow_mut().dispatches += 1;
        self.maybe_flush();
    }

    /// Explicit bind group layout for the arena pattern: uniform (dynamic
    /// offset) at binding 0, `storage_ro` read-only buffers, one read-write
    /// buffer last. The shader must declare `var<uniform> p` at binding 0.
    pub fn arena_layout(&self, key: &str, shader: &str, kinds: &str)
                        -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        self.arena_layout_impl(key, shader, kinds, true)
    }

    /// Same, without the leading uniform binding: `kinds` starts at binding 0
    /// (the existing storage-params shaders keep their original binding order,
    /// with the params binding marked `dyn` for the ring's dynamic offset).
    pub fn arena_layout_plain(&self, key: &str, shader: &str, kinds: &str)
                              -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        self.arena_layout_impl(key, shader, kinds, false)
    }

    fn arena_layout_impl(&self, key: &str, shader: &str, kinds: &str, leading_uniform: bool)
                         -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let mut entries = Vec::new();
        if leading_uniform {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(UNIFORM_SLOT as u64),
                },
                count: None,
            });
        }
        let base = if leading_uniform { 1 } else { 0 };
        for (i, kind) in kinds.split(',').enumerate() {
            let k = kind.trim();
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i as u32) + base,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    // the dyn params binding stays read-only: the ring's bytes
                    // are written via queue.write_buffer before the submit
                    ty: wgpu::BufferBindingType::Storage { read_only: k != "rw" },
                    has_dynamic_offset: k == "dyn",
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(key),
            entries: &entries,
        });
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(key),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (bgl, pipeline)
    }

    /// Record one dispatch into the current batch.    /// An arena-kernel: explicit layout (uniform params at binding 0 with a
    /// dynamic offset) plus a bind-group cache keyed by the caller's buffer
    /// set. Dispatches against a cached key cost one host memcpy + record.
    pub fn kernel(&self, key: &str, shader: &str, kinds: &str) -> Kernel {
        let (bgl, pipe) = self.arena_layout(key, shader, kinds);
        let kinds = kinds.split(',').map(|k| k.trim() == "ro").collect();
        Kernel {
            bgl,
            pipe,
            kinds,
            name: key.to_string(),
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Kernel for the existing storage-params shaders: the params binding (the
    /// LAST one) gets has_dynamic_offset, fed from the params ring per dispatch.
    pub fn kernel_params(&self, key: &str, shader: &str, kinds: &str) -> Kernel {
        let kinds_full = format!("{kinds},dyn");
        let (bgl, pipe) = self.arena_layout_plain(key, shader, &kinds_full);
        let kinds = kinds_full.split(',').map(|k| k.trim() == "ro").collect();
        Kernel {
            bgl,
            pipe,
            kinds,
            name: key.to_string(),
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Record a dispatch (no dynamic offsets) into the current batch.
    pub fn record(&self, pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup,
                  workgroups: (u32, u32, u32)) {
        self.pending.borrow_mut().push(Pending {
            pipeline: pipeline.clone(),
            group: group.clone(),
            dyn_offset: None,
            workgroups,
            label: String::new(),
        });
        self.stats.borrow_mut().dispatches += 1;
        self.maybe_flush();
    }

    /// Record a dispatch whose bind group has one dynamic offset.
    pub fn record_dyn_off(&self, pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup,
                          offset: u32, workgroups: (u32, u32, u32)) {
        self.pending.borrow_mut().push(Pending {
            pipeline: pipeline.clone(),
            group: group.clone(),
            dyn_offset: Some(offset),
            workgroups,
            label: String::new(),
        });
        self.stats.borrow_mut().dispatches += 1;
        self.maybe_flush();
    }

    /// Record a dispatch with a dynamic uniform offset into the current batch.
    pub fn record_dyn(&self, label: &str, pipeline: &wgpu::ComputePipeline,
                      group: &wgpu::BindGroup, offset: u32, workgroups: (u32, u32, u32)) {
        self.pending.borrow_mut().push(Pending {
            pipeline: pipeline.clone(),
            group: group.clone(),
            dyn_offset: Some(offset),
            workgroups,
            label: label.to_string(),
        });
        self.stats.borrow_mut().dispatches += 1;
        self.maybe_flush();
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
        self.pending.borrow_mut().push(Pending { pipeline, group, dyn_offset: None, workgroups, label: key.to_string() });
        self.maybe_flush();
    }


    /// Record the pending batch into one pass. `None` when nothing is pending.
    fn encode_batch(&self) -> Option<wgpu::CommandBuffer> {
        let batch = std::mem::take(&mut *self.pending.borrow_mut());
        if batch.is_empty() {
            return None;
        }
        let n = self.submit_count.get();
        self.submit_count.set(n + 1);
        if std::env::var_os("M5_DISPATCH_TRACE").is_some() {
            let comps: Vec<&str> = batch.iter().map(|p| p.label.as_str()).collect();
            eprintln!("[batch.{n}] {} dispatches: {}", batch.len(), comps.join(", "));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("batch.{n}")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            for pending in &batch {
                pass.set_pipeline(&pending.pipeline);
                match pending.dyn_offset {
                    Some(off) => pass.set_bind_group(0, &pending.group, &[off]),
                    None => pass.set_bind_group(0, &pending.group, &[]),
                }
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
        let used = self.arena_used.get();
        if used > 0 {
            let host = self.arena_host.borrow();
            self.queue.write_buffer(&self.arena, 0, &host[..used]);
        }
        self.arena_used.set(0);
        let pused = self.params_used.get();
        if pused > 0 {
            let host = self.params_host.borrow();
            self.queue.write_buffer(&self.params_arena, 0, &host[..pused]);
        }
        self.params_used.set(0);
        if let Some(batch) = self.encode_batch() {
            self.queue.submit(std::iter::once(batch));
            self.drain_deferred();
            let n = self.in_flight.get() + 1;
            self.in_flight.set(n);
            if n >= MAX_IN_FLIGHT {
                // block: the pipeline is deep enough
                self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
                self.in_flight.set(0);
            } else {
                // non-blocking: let the driver start the batch while the CPU
                // records the next one
                let _ = self.device.poll(wgpu::PollType::Poll);
            }
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
