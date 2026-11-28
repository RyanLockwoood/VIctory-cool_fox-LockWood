//! Detiling title textures on the card instead of the processor: one
//! compute shader for every mode [`Layout::detile`] can address, driven by
//! a [`DetilePlan`] (the in-block offsets as three small tables, so no mode
//! needs code of its own).
//!
//! Step 1 of the plan in `docs/bug-journal.md` (2026-11-28): the shader and
//! its proof against the processor's detiler. Nothing records it yet; a
//! later step stages the tiled bytes as they are and detiles them here
//! behind `VICTORY_GPU_DETILE=1`, with the processor's path as fallback.
//!
//! [`Layout::detile`]: victory_formats::texture::Layout::detile

use victory_formats::texture::DetilePlan;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

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
