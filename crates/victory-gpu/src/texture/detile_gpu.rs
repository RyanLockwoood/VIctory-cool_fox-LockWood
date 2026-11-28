//! Detiling title textures on the card instead of the processor: one
//! compute shader for every mode [`Layout::detile`] can address, driven by
//! a [`DetilePlan`] (the in-block offsets as three small tables, so no mode
//! needs code of its own).
//!
//! `VICTORY_GPU_DETILE=1` (off by default): a title texture all of whose
//! staged levels the shader can detile keeps its staging buffer in the
//! title's own tiled layout (one plain copy at refresh instead of the
//! processor's detiling). A list that uploads it records, into a side list
//! of its own ([`Work`]), a copy of those bytes into the card's memory and
//! the detiling there into rows laid out as the upload's footprints; its
//! own copy into the texture then reads those rows. The side list runs in
//! a queue call of its own just before the submission's lists, so it is
//! done before they copy, and the list's own pipeline and root signatures
//! are never touched. Every other texture goes the processor's way, as
//! before.
//!
//! [`Layout::detile`]: victory_formats::texture::Layout::detile

use std::sync::{Mutex, OnceLock};

use victory_formats::texture::{DetilePlan, Layout};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

/// Whether textures are detiled on the card (`VICTORY_GPU_DETILE=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_GPU_DETILE").as_deref() == Ok("1"))
}

/// The shader on the host's device, compiled once.
fn detiler(device: &ID3D12Device) -> Option<&'static GpuDetiler> {
    static DETILER: OnceLock<Option<GpuDetiler>> = OnceLock::new();
    DETILER
        .get_or_init(|| match GpuDetiler::new(device) {
            Ok(d) => {
                tracing::info!("VICTORY_GPU_DETILE=1: title textures the shader can detile are detiled on the card");
                Some(d)
            }
            Err(e) => {
                tracing::warn!(error = %e, "VICTORY_GPU_DETILE=1, but the detile shader did not compile: the processor detiles");
                None
            }
        })
        .as_ref()
}

/// The plans of a staging buffer's parts (`parts`: the level and array
/// slice of each, in footprint order), if the card can detile every one of
/// them from `read` bytes of title memory; else `None` (the processor
/// detiles the texture).
pub(crate) fn plans(layout: &Layout, parts: impl Iterator<Item = (u32, u32)>, read: u64) -> Option<Vec<DetilePlan>> {
    if !on() {
        return None;
    }
    parts
        .map(|(mip, slice)| layout.detile_plan(mip, slice).filter(|p| supported(p, read)))
        .collect()
}

/// Commands that run in a queue call of their own just ahead of a host
/// list's submission (the detiling of the textures it uploads), and the
/// card memory they write, both held until the list is reset (every
/// submission of it has run by then).
pub(crate) struct Work {
    pair: Option<crate::host::Prelude>,
    closed: bool,
    held: Vec<Chunk>,
}

// SAFETY: D3D12 objects (free-threaded), used under the host list's lock.
unsafe impl Send for Work {}

impl Work {
    /// An open side list, from those kept or new.
    pub(crate) fn new(host: &crate::host::Host) -> Option<Self> {
        let pair = lock(&LISTS).pop().or_else(|| host.make_prelude())?;
        // SAFETY: a kept pair's last execution finished before it was given
        // back (its list's reset); a new one has none.
        let reset = unsafe { pair.0.Reset().and_then(|()| pair.1.Reset(&pair.0, None)) };
        if let Err(e) = reset {
            tracing::warn!(error = %e, "no side list for detiling on the card");
            return None;
        }
        Some(Self { pair: Some(pair), closed: false, held: Vec::new() })
    }

    /// Closes the side list (once).
    pub(crate) fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if let Some((_, list)) = &self.pair
            // SAFETY: our open list.
            && let Err(e) = unsafe { list.Close() }
        {
            tracing::warn!(error = %e, "a detiling side list did not close: its textures stay as they were");
            self.pair = None;
        }
    }

    /// The closed side list, to execute.
    pub(crate) fn list(&self) -> Option<ID3D12CommandList> {
        if !self.closed {
            return None;
        }
        self.pair.as_ref().and_then(|(_, list)| list.cast().ok())
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        // Kept for the next list that detiles (a few; the rest are let go).
        if let Some(pair) = self.pair.take() {
            let mut kept = lock(&LISTS);
            if kept.len() < 64 {
                kept.push(pair);
            }
        }
    }
}

/// Side lists kept for reuse: only ones whose execution has finished.
static LISTS: Mutex<Vec<crate::host::Prelude>> = Mutex::new(Vec::new());

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A buffer in the card's memory, a power of two of at least 64 KiB, given
/// back for reuse when dropped while few bytes wait.
pub(crate) struct Chunk {
    buffer: ID3D12Resource,
    class: u32,
}

/// Bytes of chunks kept for reuse at most.
const KEPT_MOST: u64 = 256 << 20;
static KEPT: Mutex<(Vec<Chunk>, u64)> = Mutex::new((Vec::new(), 0));

impl Drop for Chunk {
    fn drop(&mut self) {
        let size = 1u64 << self.class;
        let mut kept = lock(&KEPT);
        if kept.1 + size <= KEPT_MOST {
            kept.1 += size;
            kept.0.push(Chunk { buffer: self.buffer.clone(), class: self.class });
        }
    }
}

/// A chunk of at least `bytes`: a kept one of that size, or a new one.
fn chunk(device: &ID3D12Device, bytes: u64) -> Option<Chunk> {
    let class = bytes.max(64 << 10).next_power_of_two().trailing_zeros();
    {
        let mut kept = lock(&KEPT);
        if let Some(i) = kept.0.iter().position(|c| c.class == class) {
            let c = kept.0.swap_remove(i);
            kept.1 -= 1 << class;
            return Some(c);
        }
    }
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 1 << class,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a plain buffer.
    let made = unsafe {
        device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, bytes, "no card memory for detiling a texture");
        return None;
    }
    CHUNKS_MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(Chunk { buffer: buffer?, class })
}

/// Chunks made since the start (for the tests and the memory counters).
pub(crate) static CHUNKS_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The tables of a plan in an upload buffer, made once per set of tables
/// (one per mode and element size) and kept.
fn tables_buffer(host: &crate::host::Host, plan: &DetilePlan) -> Option<ID3D12Resource> {
    struct Kept(std::collections::HashMap<Vec<u32>, ID3D12Resource>);
    // SAFETY: D3D12 resources are free-threaded.
    unsafe impl Send for Kept {}
    static TABLES: Mutex<Option<Kept>> = Mutex::new(None);
    let table = tables(plan);
    let mut kept = lock(&TABLES);
    let kept = &mut kept.get_or_insert_with(|| Kept(Default::default())).0;
    if let Some(buffer) = kept.get(&table) {
        return Some(buffer.clone());
    }
    let bytes = (table.len() as u64 * 4).max(4);
    let (buffer, data) = host.upload_buffer(bytes)?;
    // SAFETY: a mapped upload buffer of `bytes`; nothing reads it yet.
    unsafe { std::ptr::copy_nonoverlapping(table.as_ptr().cast::<u8>(), data, table.len() * 4) };
    kept.insert(table, buffer.clone());
    Some(buffer)
}

/// Records into `work` the detiling of a staging buffer that holds `read`
/// bytes of a texture's tiled memory (from `staging` at `base`), into card
/// memory laid out as `footprints` (`rows` rows each, `total` bytes).
/// Returns the buffer the list's copies into the texture read instead of
/// the staging buffer (offsets from 0); `None` if nothing could be recorded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    host: &crate::host::Host,
    work: &mut Work,
    plans: &[DetilePlan],
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
    (staging, base, read): (&ID3D12Resource, u64, u64),
    total: u64,
) -> Option<ID3D12Resource> {
    // (The shader's addresses and the rows' offsets are 32-bit.)
    if work.closed || plans.len() != footprints.len() || total > u64::from(u32::MAX) || read > u64::from(u32::MAX) {
        return None;
    }
    let detiler = detiler(&host.device)?;
    let list = work.pair.as_ref()?.1.clone();
    let tiled = chunk(&host.device, read)?;
    let linear = chunk(&host.device, total)?;
    let tables = tables_buffer(host, &plans[0])?;
    let barrier = crate::host::transition;
    // SAFETY: an open side list; live buffers large enough for every
    // plan (`read` and `total`), held by `work` until the list is reset.
    unsafe {
        list.CopyBufferRegion(&tiled.buffer, 0, staging, base, read);
        list.ResourceBarrier(&[barrier(
            &tiled.buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )]);
        let (tiled_at, tables_at, linear_at) =
            (tiled.buffer.GetGPUVirtualAddress(), tables.GetGPUVirtualAddress(), linear.buffer.GetGPUVirtualAddress());
        for ((plan, footprint), &n) in plans.iter().zip(footprints).zip(rows) {
            let pitch = footprint.Footprint.RowPitch;
            for z in 0..plan.bases.len() as u32 {
                let at = footprint.Offset + u64::from(z) * u64::from(pitch) * u64::from(n);
                let k = constants(plan, z, at as u32, pitch);
                detiler.record(&list, plan, &k, (tiled_at, tables_at, linear_at));
            }
        }
        // The rows are written before the list's copies read them; left
        // readable as a copy's source (buffers decay to common between
        // queue calls in any case).
        list.ResourceBarrier(&[barrier(
            &linear.buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
    }
    let rows_buffer = linear.buffer.clone();
    work.held.push(tiled);
    work.held.push(linear);
    DETILED.fetch_add(plans.len() as u64, std::sync::atomic::Ordering::Relaxed);
    Some(rows_buffer)
}

/// Subresources detiled on the card since the start.
pub(crate) static DETILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The shader. Root constants (16): width, height, dwords per element,
/// linear, pitch, block width, block height, log2 block bytes, blocks per
/// row, tail x, tail y, base, depth bits, destination base, destination row
/// pitch, unused. t0: the tiled texture from its start; t1: the column
/// table then the row table (u32 each); u0: the linear rows.
const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=16, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer C : register(b0) {
    uint width; uint height; uint dwords; uint is_linear;
    uint pitch; uint block_w; uint block_h; uint block_log2;
    uint blocks_per_row; uint tail_x; uint tail_y; uint base;
    uint depth_bits; uint dst_base; uint dst_row_pitch; uint unused;
};
ByteAddressBuffer tiled : register(t0);
ByteAddressBuffer tables : register(t1);
RWByteAddressBuffer dst : register(u0);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) {
        return;
    }
    uint bytes = dwords * 4;
    uint src;
    if (is_linear != 0) {
        src = base + (id.y * pitch + id.x) * bytes;
    } else {
        uint x = id.x + tail_x;
        uint y = id.y + tail_y;
        uint row = (((y / block_h) * blocks_per_row) << block_log2)
            + (tables.Load((block_w + y % block_h) * 4) | depth_bits);
        uint column = ((x / block_w) << block_log2) + tables.Load((x % block_w) * 4);
        src = base + row + column;
    }
    uint to = dst_base + id.y * dst_row_pitch + id.x * bytes;
    for (uint i = 0; i < dwords; i++) {
        dst.Store(to + i * 4, tiled.Load(src + i * 4));
    }
}
"#;

/// The compiled shader and its root signature.
pub(crate) struct GpuDetiler {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

/// Threads per group along each side ([numthreads(8, 8, 1)]).
const GROUP: u32 = 8;

impl GpuDetiler {
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("detile"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(e) = compiled {
            if let Some(errors) = errors {
                // SAFETY: the blob's own bytes.
                let text = unsafe {
                    std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
                };
                tracing::warn!("detile shader: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: the blob's own bytes; the root signature is inside the shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            Ok(Self { root, pso })
        }
    }

    /// Records the detiling of depth slice `z` of `plan`'s subresource:
    /// from `tiled` (the texture from its start), with `tables`
    /// ([`tables`]), into `dst` at `constants`' destination base. The
    /// caller makes the resources readable and writable as the shader
    /// needs, and keeps them alive until the list has run.
    ///
    /// # Safety
    /// An open list; addresses of live buffers large enough for the plan.
    pub(crate) unsafe fn record(
        &self,
        list: &ID3D12GraphicsCommandList,
        plan: &DetilePlan,
        constants: &[u32; 16],
        (tiled, tables, dst): (u64, u64, u64),
    ) {
        // SAFETY: the caller's promise.
        unsafe {
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRoot32BitConstants(0, 16, constants.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, tiled);
            list.SetComputeRootShaderResourceView(2, tables);
            list.SetComputeRootUnorderedAccessView(3, dst);
            list.Dispatch(plan.width.div_ceil(GROUP), plan.height.div_ceil(GROUP), 1);
        }
    }
}

/// Whether the shader can detile `plan`'s subresource: elements of 4, 8 or
/// 16 bytes (dword loads), every address within 32 bits. Others stay with
/// the processor.
pub(crate) fn supported(plan: &DetilePlan, tiled_bytes: u64) -> bool {
    matches!(plan.element_bytes, 4 | 8 | 16) && tiled_bytes <= u64::from(u32::MAX)
}

/// The shader's t1: the column table, then the row table.
pub(crate) fn tables(plan: &DetilePlan) -> Vec<u32> {
    plan.columns.iter().chain(&plan.rows).copied().collect()
}

/// The root constants for depth slice `z`, its rows written from
/// `dst_base` on, `dst_row_pitch` bytes apart.
pub(crate) fn constants(plan: &DetilePlan, z: u32, dst_base: u32, dst_row_pitch: u32) -> [u32; 16] {
    [
        plan.width,
        plan.height,
        plan.element_bytes / 4,
        u32::from(plan.linear),
        plan.pitch,
        plan.block_width.max(1),
        plan.block_height.max(1),
        plan.block_log2,
        plan.blocks_per_row,
        plan.tail_x,
        plan.tail_y,
        plan.bases[z as usize] as u32,
        plan.depth_bits[z as usize],
        dst_base,
        dst_row_pitch,
        0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    /// A buffer the processor writes and reads and the shader may write.
    fn buffer(device: &ID3D12Device, size: u64) -> (ID3D12Resource, *mut u8) {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(4),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain buffer creation and persistent mapping.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut resource)
                .unwrap();
            let resource = resource.unwrap();
            resource.Map(0, None, Some(&mut data)).unwrap();
            (resource, data.cast())
        }
    }

    /// The shader's linear rows are the processor detiler's, byte for byte:
    /// each mode with tables, elements of 4, 8 and 16 bytes, odd sizes, mips
    /// into the tail, arrays and 3D textures. On the software adapter.
    #[test]
    fn the_card_detiles_as_the_processor_does() {
        let _turn = crate::indirect::tests::warp_turn();
        // SAFETY: test harness setup on the software adapter.
        let (device, queue, allocator, list, fence, event) = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
                return;
            };
            let Ok(adapter) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                return;
            };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            (device, queue, allocator, list, fence, event)
        };
        let detiler = GpuDetiler::new(&device).expect("the detile shader must compile on WARP");
        let mut submitted = 0u64;
        let mut compared = 0usize;
        let modes = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        for swizzle in modes {
            for bpe in [4u32, 8, 16] {
                for (dimension, w, h, depth, mips) in [
                    (Dimension::Tex2d, 256, 256, 1, 9),
                    (Dimension::Tex2d, 300, 77, 1, 9),
                    (Dimension::Tex2d, 33, 200, 3, 8),
                    (Dimension::Tex3d, 40, 24, 12, 6),
                ] {
                    let desc = Desc { dimension, width: w, height: h, depth, mips, element_bytes: bpe, block: 1, samples: 1, swizzle };
                    let Ok(layout) = Layout::new(&desc) else {
                        continue;
                    };
                    if !layout.addressable() {
                        continue;
                    }
                    // The tiled texture: bytes that differ everywhere.
                    let (tiled, tiled_ptr) = buffer(&device, layout.size);
                    let mut seed = 0x9e37_79b9_u32 ^ (bpe << 8) ^ swizzle.0;
                    // SAFETY: the mapped buffer's own bytes.
                    let tiled_bytes = unsafe { std::slice::from_raw_parts_mut(tiled_ptr, layout.size as usize) };
                    for b in tiled_bytes.iter_mut() {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        *b = seed as u8;
                    }
                    let slices = if dimension == Dimension::Tex3d { 1 } else { depth };
                    for mip in 0..layout.mips.len() as u32 {
                        for slice in 0..slices {
                            let plan = layout.detile_plan(mip, slice).unwrap();
                            assert!(supported(&plan, layout.size));
                            let m = &layout.mips[mip as usize];
                            let row_pitch = (m.width * bpe).next_multiple_of(256) as usize;
                            let depth_pitch = row_pitch * m.height as usize;
                            let total = depth_pitch * plan.bases.len();
                            let mut expected = vec![0u8; total];
                            layout.detile(tiled_bytes, mip, slice, &mut expected, row_pitch, depth_pitch).unwrap();
                            let table = tables(&plan);
                            let (tables_buf, tables_ptr) = buffer(&device, (table.len() as u64 * 4).max(4));
                            let (dst, dst_ptr) = buffer(&device, total as u64);
                            // SAFETY: mapped buffers of these sizes; nothing runs yet; an open list.
                            unsafe {
                                std::ptr::copy_nonoverlapping(table.as_ptr().cast::<u8>(), tables_ptr, table.len() * 4);
                                std::ptr::write_bytes(dst_ptr, 0xcd, total);
                                for z in 0..plan.bases.len() as u32 {
                                    let k = constants(&plan, z, z * depth_pitch as u32, row_pitch as u32);
                                    detiler.record(
                                        &list,
                                        &plan,
                                        &k,
                                        (tiled.GetGPUVirtualAddress(), tables_buf.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()),
                                    );
                                }
                                list.Close().unwrap();
                                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                                submitted += 1;
                                queue.Signal(&fence, submitted).unwrap();
                                fence.SetEventOnCompletion(submitted, event).unwrap();
                                WaitForSingleObject(event, INFINITE);
                                allocator.Reset().unwrap();
                                list.Reset(&allocator, None).unwrap();
                            }
                            // SAFETY: the list has run; the mapped buffer's bytes.
                            let got = unsafe { std::slice::from_raw_parts(dst_ptr, total) };
                            // Only the rows' element bytes are compared: the
                            // padding past each row is neither side's.
                            let bytes = (m.width * bpe) as usize;
                            for z in 0..plan.bases.len() {
                                for y in 0..m.height as usize {
                                    let at = z * depth_pitch + y * row_pitch;
                                    assert_eq!(
                                        &got[at..at + bytes],
                                        &expected[at..at + bytes],
                                        "{swizzle:?} {bpe} B {dimension:?} {w}x{h}x{depth} mip {mip} slice {slice} z {z} row {y}"
                                    );
                                }
                            }
                            compared += 1;
                        }
                    }
                }
            }
        }
        // SAFETY: our own event.
        unsafe {
            let _ = CloseHandle(event);
        }
        assert!(compared > 300, "{compared} subresources compared");
    }

    /// The processor's detiler against the card's on this PC's first
    /// adapter: one 2048 x 2048 texture of 16-byte blocks (64 MiB) in
    /// SW_64KB_D, its tiled bytes in the PC's memory as staging would hold
    /// them. Prints both rates and checks the results agree. Run on its own:
    /// `cargo test --release -p victory-gpu --lib detile_rates -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement on the real adapter"]
    fn detile_rates_on_this_pc() {
        let _turn = crate::indirect::tests::warp_turn();
        // SAFETY: test harness setup on the first hardware adapter.
        let (device, queue, allocator, list, fence, event, name) = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter1 = factory.EnumAdapters1(0).unwrap();
            let desc = adapter.GetDesc1().unwrap();
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                return;
            }
            let name = String::from_utf16_lossy(&desc.Description).trim_end_matches('\0').to_owned();
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).unwrap();
            let device = device.unwrap();
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            (device, queue, allocator, list, fence, event, name)
        };
        let detiler = GpuDetiler::new(&device).unwrap();
        let desc = Desc {
            dimension: Dimension::Tex2d,
            width: 2048,
            height: 2048,
            depth: 1,
            mips: 1,
            element_bytes: 16,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_D,
        };
        let layout = Layout::new(&desc).unwrap();
        let plan = layout.detile_plan(0, 0).unwrap();
        let (tiled, tiled_ptr) = buffer(&device, layout.size);
        // SAFETY: the mapped buffer's own bytes.
        let tiled_bytes = unsafe { std::slice::from_raw_parts_mut(tiled_ptr, layout.size as usize) };
        for (i, b) in tiled_bytes.iter_mut().enumerate() {
            *b = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
        }
        let row_pitch = (desc.width * 16) as usize;
        let total = row_pitch * desc.height as usize;
        let mut cpu = vec![0u8; total];
        let runs = 10;
        let started = std::time::Instant::now();
        for _ in 0..runs {
            layout.detile(tiled_bytes, 0, 0, &mut cpu, row_pitch, total).unwrap();
        }
        let cpu_s = started.elapsed().as_secs_f64() / f64::from(runs);
        let table = tables(&plan);
        let (tables_buf, tables_ptr) = buffer(&device, table.len() as u64 * 4);
        let (dst, dst_ptr) = buffer(&device, total as u64);
        let k = constants(&plan, 0, 0, row_pitch as u32);
        let mut submitted = 0;
        let mut card_s = f64::MAX;
        // SAFETY: mapped buffers of these sizes; an open list each time.
        unsafe {
            std::ptr::copy_nonoverlapping(table.as_ptr().cast::<u8>(), tables_ptr, table.len() * 4);
            for _ in 0..=runs {
                detiler.record(&list, &plan, &k, (tiled.GetGPUVirtualAddress(), tables_buf.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()));
                list.Close().unwrap();
                let started = std::time::Instant::now();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                submitted += 1;
                queue.Signal(&fence, submitted).unwrap();
                fence.SetEventOnCompletion(submitted, event).unwrap();
                WaitForSingleObject(event, INFINITE);
                // The first submission also pays for the pipeline's first use.
                if submitted > 1 {
                    card_s = card_s.min(started.elapsed().as_secs_f64());
                }
                allocator.Reset().unwrap();
                list.Reset(&allocator, None).unwrap();
            }
            let _ = CloseHandle(event);
        }
        // SAFETY: the list has run; the mapped buffer's bytes.
        let got = unsafe { std::slice::from_raw_parts(dst_ptr, total) };
        assert!(got == cpu.as_slice(), "the card's rows differ from the processor's");
        // As the layer would do it: the tiled bytes copied to the card's own
        // memory in one transfer, detiled there, card memory to card memory.
        let vram = |size: u64| -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: a plain buffer.
            unsafe {
                device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r)
                    .unwrap();
            }
            r.unwrap()
        };
        let (card_tiled, card_dst) = (vram(layout.size), vram(total as u64));
        let event = unsafe { CreateEventW(None, false, false, None).unwrap() };
        let mut in_card_s = f64::MAX;
        let barrier = |r: &ID3D12Resource, from, to| crate::host::transition(r, from, to);
        // SAFETY: live buffers of these sizes; an open list each time.
        unsafe {
            std::ptr::write_bytes(dst_ptr, 0, total);
            for run in 0..=runs {
                list.CopyBufferRegion(&card_tiled, 0, &tiled, 0, layout.size);
                list.ResourceBarrier(&[barrier(&card_tiled, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                detiler.record(&list, &plan, &k, (card_tiled.GetGPUVirtualAddress(), tables_buf.GetGPUVirtualAddress(), card_dst.GetGPUVirtualAddress()));
                if run == runs {
                    // Last run: bring the rows back to check them.
                    list.ResourceBarrier(&[barrier(&card_dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    list.CopyBufferRegion(&dst, 0, &card_dst, 0, total as u64);
                }
                list.Close().unwrap();
                let started = std::time::Instant::now();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                submitted += 1;
                queue.Signal(&fence, submitted).unwrap();
                fence.SetEventOnCompletion(submitted, event).unwrap();
                WaitForSingleObject(event, INFINITE);
                if run > 0 && run < runs {
                    in_card_s = in_card_s.min(started.elapsed().as_secs_f64());
                }
                allocator.Reset().unwrap();
                list.Reset(&allocator, None).unwrap();
            }
            let _ = CloseHandle(event);
        }
        // SAFETY: the list has run; the mapped buffer's bytes.
        let got = unsafe { std::slice::from_raw_parts(dst_ptr, total) };
        assert!(got == cpu.as_slice(), "the card's rows (in its own memory) differ from the processor's");
        // What the processor would still do: copy the tiled bytes as they are.
        let mut plain = vec![0u8; total];
        let started = std::time::Instant::now();
        for _ in 0..runs {
            plain.copy_from_slice(&tiled_bytes[..total]);
            std::hint::black_box(&plain);
        }
        let copy_s = started.elapsed().as_secs_f64() / f64::from(runs);
        let mb = total as f64 / 1e6;
        eprintln!(
            "detile 2048x2048 x 16 B (SW_64KB_D, {mb:.0} MB) on {name}:\n  processor detile {:.2} ms ({:.0} MB/s, one thread)\n  \
             processor plain copy {:.2} ms ({:.0} MB/s): what it would still do with the card detiling\n  \
             card, reading and writing the PC's memory: {:.2} ms\n  \
             card, one copy into its memory then detile there: {:.2} ms (submission and wait included)",
            cpu_s * 1e3,
            mb / cpu_s,
            copy_s * 1e3,
            mb / copy_s,
            card_s * 1e3,
            in_card_s * 1e3,
        );
    }

    /// The whole path on the host's own device and queue, as a run takes
    /// it: a real texture uploaded once the processor's way (detiled rows
    /// from staging) and once the card's way (tiled staging, the host list's
    /// side list detiling into card memory, the list copying those rows),
    /// through `Host::execute` (its submission runs the side list first).
    /// Both textures are read back and must be equal, for 4-, 8- and
    /// 16-byte elements, block-compressed and not, with mips, arrays and a
    /// 3D texture, in several modes.
    #[test]
    fn a_texture_detiled_on_the_card_equals_the_processors() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let _turn = crate::indirect::tests::warp_turn();
        let Some(host) = crate::host::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let readback = |size: u64| -> (ID3D12Resource, *mut u8) {
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: a plain buffer, mapped for reading.
            unsafe {
                host.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)
                    .unwrap();
                let r = r.unwrap();
                r.Map(0, None, Some(&mut data)).unwrap();
                (r, data.cast())
            }
        };
        let location = |resource: &ID3D12Resource, footprint: Option<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, index: u32| {
            D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Type: if footprint.is_some() { D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT } else { D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX },
                Anonymous: match footprint {
                    Some(f) => D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
                    None => D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
                },
            }
        };
        let cases: [(DXGI_FORMAT, u32, u32, Dimension, u32, u32, u32, u32, Swizzle); 6] = [
            (DXGI_FORMAT_R8G8B8A8_UINT, 4, 1, Dimension::Tex2d, 300, 77, 1, 9, Swizzle::SW_64KB_D),
            (DXGI_FORMAT_R8G8B8A8_UINT, 4, 1, Dimension::Tex2d, 256, 256, 2, 9, Swizzle::SW_4KB_S),
            (DXGI_FORMAT_BC1_UNORM, 8, 4, Dimension::Tex2d, 512, 256, 1, 10, Swizzle::SW_64KB_S),
            (DXGI_FORMAT_BC7_UNORM, 16, 4, Dimension::Tex2d, 256, 128, 3, 8, Swizzle::SW_64KB_D),
            (DXGI_FORMAT_BC7_UNORM, 16, 4, Dimension::Tex2d, 64, 64, 1, 7, Swizzle::SW_256B_D),
            (DXGI_FORMAT_R32G32B32A32_UINT, 16, 1, Dimension::Tex3d, 40, 24, 12, 4, Swizzle::SW_64KB_S),
        ];
        let mut compared = 0;
        for (format, bpe, block, dimension, w, h, depth, mips, swizzle) in cases {
            let desc = Desc { dimension, width: w, height: h, depth, mips, element_bytes: bpe, block, samples: 1, swizzle };
            let layout = Layout::new(&desc).unwrap();
            assert!(layout.addressable(), "{desc:?}");
            let texture_desc = D3D12_RESOURCE_DESC {
                Dimension: if dimension == Dimension::Tex3d { D3D12_RESOURCE_DIMENSION_TEXTURE3D } else { D3D12_RESOURCE_DIMENSION_TEXTURE2D },
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: depth as u16,
                MipLevels: mips as u16,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let count = if dimension == Dimension::Tex3d { mips } else { mips * depth };
            let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
            let mut rows = vec![0u32; count as usize];
            let mut total = 0;
            // SAFETY: outputs for `count` subresources.
            unsafe {
                host.device.GetCopyableFootprints(
                    &texture_desc,
                    0,
                    count,
                    0,
                    Some(footprints.as_mut_ptr()),
                    Some(rows.as_mut_ptr()),
                    None,
                    Some(&mut total),
                );
            }
            let read = layout.size;
            // The title's tiled bytes, different everywhere.
            let mut tiled = vec![0u8; read as usize];
            let mut seed = 0x2545_f491_u32 ^ swizzle.0 ^ (bpe << 12);
            for b in &mut tiled {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *b = seed as u8;
            }
            // The processor's way: rows in staging.
            let (cpu_staging, cpu_data) = host.upload_buffer(total).unwrap();
            let part = |i: usize| -> (u32, u32) { ((i as u32) % mips, (i as u32) / mips) };
            for (i, f) in footprints.iter().enumerate() {
                let (mip, slice) = part(i);
                let pitch = f.Footprint.RowPitch as usize;
                let len = pitch * rows[i] as usize * f.Footprint.Depth as usize;
                // SAFETY: the subresource's part of the mapped staging buffer.
                let linear = unsafe { std::slice::from_raw_parts_mut(cpu_data.add(f.Offset as usize), len) };
                layout.detile(&tiled, mip, slice, linear, pitch, pitch * rows[i] as usize).unwrap();
            }
            // The card's way: the tiled bytes as they are.
            let (gpu_staging, gpu_data) = host.upload_buffer(read).unwrap();
            // SAFETY: a mapped upload buffer of `read` bytes.
            unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), gpu_data, tiled.len()) };
            let plans: Vec<DetilePlan> = (0..count as usize)
                .map(|i| {
                    let (mip, slice) = part(i);
                    layout.detile_plan(mip, if dimension == Dimension::Tex3d { 0 } else { slice }).unwrap()
                })
                .collect();
            let made = |_| -> ID3D12Resource {
                let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
                let mut r: Option<ID3D12Resource> = None;
                // SAFETY: a valid texture description.
                unsafe {
                    host.device
                        .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)
                        .unwrap();
                }
                r.unwrap()
            };
            let [by_cpu, by_card] = [0, 1].map(made);
            let backs = [readback(total), readback(total)];
            let card_rows = list
                .detile_work(|host, work| record(host, work, &plans, &footprints, &rows, (&gpu_staging, 0, read), total))
                .expect("the detiling records");
            // SAFETY: an open list; live resources, held below until it has run.
            unsafe {
                for (i, f) in footprints.iter().enumerate() {
                    list.list.CopyTextureRegion(&location(&by_cpu, None, i as u32), 0, 0, 0, &location(&cpu_staging, Some(*f), 0), None);
                    list.list.CopyTextureRegion(&location(&by_card, None, i as u32), 0, 0, 0, &location(&card_rows, Some(*f), 0), None);
                }
                for t in [&by_cpu, &by_card] {
                    list.list.ResourceBarrier(&[crate::host::transition(t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                }
                for (t, (back, _)) in [&by_cpu, &by_card].into_iter().zip(&backs) {
                    for (i, f) in footprints.iter().enumerate() {
                        list.list.CopyTextureRegion(&location(back, Some(*f), 0), 0, 0, 0, &location(t, None, i as u32), None);
                    }
                }
            }
            list.close();
            host.execute(&[&list]);
            list.reset();
            for (i, f) in footprints.iter().enumerate() {
                let pitch = f.Footprint.RowPitch as usize;
                let bytes = (f.Footprint.Width.div_ceil(block) * bpe) as usize;
                for z in 0..f.Footprint.Depth as usize {
                    for y in 0..rows[i] as usize {
                        let at = f.Offset as usize + (z * rows[i] as usize + y) * pitch;
                        // SAFETY: mapped read-back buffers of `total` bytes; the list has run.
                        let (a, b) = unsafe {
                            (
                                std::slice::from_raw_parts(backs[0].1.add(at), bytes),
                                std::slice::from_raw_parts(backs[1].1.add(at), bytes),
                            )
                        };
                        assert_eq!(a, b, "{format:?} {swizzle:?} {dimension:?} {w}x{h}x{depth} subresource {i} z {z} row {y}");
                    }
                }
                compared += 1;
            }
            drop((cpu_staging, gpu_staging, by_cpu, by_card, backs, card_rows));
        }
        assert!(compared >= 40, "{compared} subresources compared");
    }

    /// Elements of 1 or 2 bytes stay with the processor.
    #[test]
    fn small_elements_are_not_for_the_card() {
        let desc = Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 1,
            element_bytes: 2,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_D,
        };
        let layout = Layout::new(&desc).unwrap();
        let plan = layout.detile_plan(0, 0).unwrap();
        assert!(!supported(&plan, layout.size));
        assert!(!supported(&DetilePlan { element_bytes: 4, ..plan }, u64::from(u32::MAX) + 1));
    }
}
