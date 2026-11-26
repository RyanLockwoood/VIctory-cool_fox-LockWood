//! Host-memory admission for speculative texture work and reusable upload storage.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
static PRESSURED: AtomicBool = AtomicBool::new(false);
static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);

fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: the API initializes the fields of a correctly sized structure.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

// The texture sweep may run for seconds inside a single submitted frame.
static NEXT_COMMIT_SAMPLE: AtomicU64 = AtomicU64::new(0);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn allocation_error(error: &windows::core::Error) {
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        RETRY_UPLOAD_AT.fetch_max(crate::host::coarse_ms().saturating_add(500), Ordering::Relaxed);
    }
}

thread_local! {
    static STAGING_DEFERRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn note_staging_deferred() {
    STAGING_DEFERRED.set(true);
}

pub(crate) fn take_staging_deferred() -> bool {
    STAGING_DEFERRED.replace(false)
}

pub(crate) fn upload_refused() -> bool {
    crate::host::coarse_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

fn sample_commit(now: u64) -> u64 {
    let next = NEXT_COMMIT_SAMPLE.load(Ordering::Relaxed);
    if now >= next
        && NEXT_COMMIT_SAMPLE
            .compare_exchange(next, now.saturating_add(32), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && let Some(status) = memory_status()
    {
        AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    }
    AVAILABLE_COMMIT.load(Ordering::Relaxed)
}

fn has_upload_reserve(commit: u64, bake: bool) -> bool {
    // Keep a protective floor against Rust/game OOM aborts without starving
    // bindless world textures or one-shot ped decoration bakes when steady-state
    // commit headroom sits in the 1-4 GiB range on 32 GiB hosts.
    commit >= if bake { 256 * MIB } else { 512 * MIB }
}

fn has_growth_reserve(commit: u64, bake: bool) -> bool {
    // Growing a fresh 64 MiB upload block for speculative bindless prefetch
    // requires more headroom than reusing slices already committed inside the
    // staging pool.
    commit >= if bake { 256 * MIB } else { 1024 * MIB }
}

/// Leave commit for game allocations and submissions already in flight while
/// still letting bindless textures and one-shot material bakes upload.
pub(crate) fn admit_texture_pass(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_upload_reserve(sample_commit(now), bake)
}

/// Whether the staging allocator may commit a brand-new OS upload buffer
/// (reusing existing free ranges in already-committed pool blocks stays open
/// whenever `admit_texture_pass` is true).
pub(crate) fn allow_staging_growth(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_growth_reserve(sample_commit(now), bake)
}

pub(crate) fn defer_speculative_buffer_growth() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    crate::texture::PICKING.get().is_some()
        && !crate::texture::BAKING.get()
        && !crate::texture::BAKE_UPLOAD_CONTEXT.get()
        && *ENABLED.get_or_init(|| std::env::var("VICTORY_SPECULATIVE_COMMIT_GUARD").as_deref() == Ok("1"))
        && sample_commit(crate::host::coarse_ms()) < 1536 * MIB
}

pub(crate) static SPECULATIVE_BUFFER_DEFERRALS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn limited_host_ram() -> bool {
    static LIMITED: OnceLock<bool> = OnceLock::new();
    *LIMITED.get_or_init(|| memory_status().is_some_and(|s| s.ullTotalPhys <= 32 * GIB))
}

fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    if was {
        physical < 8 * GIB || commit < 8 * GIB
    } else {
        physical <= 6 * GIB || commit <= 6 * GIB
    }
}

/// Called by the once-per-second submission sweep. A failed sample keeps the last good state.
pub(crate) fn refresh() {
    let Some(status) = memory_status() else { return };
    let (physical, commit) = (status.ullAvailPhys, status.ullAvailPageFile);
    HEADROOM.store(physical.min(commit), Ordering::Relaxed);
    let was = pressured();
    let pressure = next_pressure(was, physical, commit);
    if pressure != was {
        PRESSURED.store(pressure, Ordering::Relaxed);
        if pressure {
            tracing::warn!(
                pid = std::process::id(),
                frame = crate::frame::current(),
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                buffer_mirror_span_mib = crate::shadow::live_span_bytes() >> 20,
                texture_staging_mib = crate::texture::live_staging_bytes() >> 20,
                spare_scratch_mib = crate::host::ALIVE[4].load(Ordering::Relaxed) >> 20,
                game_filled_gpu_texture_mib = crate::resource::KINDS[1].load(Ordering::Relaxed) >> 20,
                "host memory pressure: limiting texture prefetch and idle upload storage"
            );
        } else {
            tracing::info!(physical_free_mib = physical >> 20, available_commit_mib = commit >> 20, "host memory pressure cleared");
        }
    }
}

pub(crate) fn headroom() -> Option<u64> {
    let bytes = HEADROOM.load(Ordering::Relaxed);
    (bytes != u64::MAX).then_some(bytes)
}

pub(crate) fn pressured() -> bool {
    PRESSURED.load(Ordering::Relaxed)
}

pub(crate) fn sampling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VICTORY_MEMORY_SAMPLE").as_deref() == Ok("1"))
}

/// Records a snapshot before the periodic texture and staging pool eviction.
pub(crate) fn log_sample(video: Option<[u64; 4]>) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    if !tracing::enabled!(target: "victory_memory", tracing::Level::INFO) {
        return;
    }
    let mut process = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: the process handle is valid and cb describes the extended output buffer.
    let process_ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut process).cast(), process.cb) } != 0;
    let (arena, idle_arena) = crate::upload::arena_bytes();
    let [texture_pool, staging_pool] = crate::pool::memory();
    let (texture_occupied, texture_free, texture_empty, texture_resting) = crate::pool::texture_memory();
    let (retired, oldest_ms, retired_any, retired_resources) = crate::host::retirement_sample();
    let (started, finished) = crate::host::submissions();
    let queued = crate::queue::queued();
    let os = memory_status();
    tracing::info!(
        target: "victory_memory",
        frame = crate::frame::current(),
        elapsed_ms = crate::host::now_ms(),
        process_private_mib = process_ok.then_some(process.PrivateUsage as u64 / MIB),
        process_working_set_mib = process_ok.then_some(process.WorkingSetSize as u64 / MIB),
        physical_free_mib = os.as_ref().map(|s| s.ullAvailPhys / MIB),
        available_commit_mib = os.as_ref().map(|s| s.ullAvailPageFile / MIB),
        dxgi_local_budget_mib = video.map(|v| v[0] / MIB),
        dxgi_local_used_mib = video.map(|v| v[1] / MIB),
        dxgi_nonlocal_budget_mib = video.map(|v| v[2] / MIB),
        dxgi_nonlocal_used_mib = video.map(|v| v[3] / MIB),
        upload_arena_mib = arena / MIB,
        upload_arena_idle_mib = idle_arena / MIB,
        texture_pool_mib = texture_pool,
        texture_pool_occupied_mib = texture_occupied / MIB,
        texture_pool_free_mib = texture_free / MIB,
        texture_pool_empty_heaps_mib = texture_empty / MIB,
        texture_pool_resting_mib = texture_resting / MIB,
        staging_pool_mib = staging_pool,
        staging_idle_limit_blocks = crate::pool::STAGING_IDLE_LIMIT.load(Ordering::Relaxed),
        live_staging_mib = crate::texture::live_staging_bytes() / MIB,
        texture_recorded_upload_bytes = crate::texture::UPLOAD_TRAFFIC[0].load(Ordering::Relaxed),
        texture_prefetch_upload_bytes = crate::texture::UPLOAD_PATH_BYTES[crate::texture::UploadPath::Prefetch as usize].load(Ordering::Relaxed),
        texture_draw_upload_bytes = crate::texture::UPLOAD_PATH_BYTES[crate::texture::UploadPath::Draw as usize].load(Ordering::Relaxed),
        texture_bake_upload_bytes = crate::texture::UPLOAD_PATH_BYTES[crate::texture::UploadPath::Bake as usize].load(Ordering::Relaxed),
        texture_ahead_upload_bytes = crate::texture::UPLOAD_PATH_BYTES[crate::texture::UploadPath::Ahead as usize].load(Ordering::Relaxed),
        prefetch_pressure_time_stops = crate::draw::PREFETCH_PRESSURE_TIME_STOPS.load(Ordering::Relaxed),
        texture_heap_pass_us = crate::draw::TEXTURE_HEAP_PASS_US.load(Ordering::Relaxed),
        texture_heap_descriptors_us = crate::draw::TEXTURE_HEAP_DESCRIPTORS_US.load(Ordering::Relaxed),
        texture_heap_create_us = crate::draw::TEXTURE_HEAP_CREATE_US.load(Ordering::Relaxed),
        texture_heap_staging_us = crate::draw::TEXTURE_HEAP_STAGING_US.load(Ordering::Relaxed),
        texture_staging_lock_wait_us = crate::texture::STAGING_LOCK_WAIT_US.load(Ordering::Relaxed),
        texture_staging_footprint_us = crate::texture::STAGING_FOOTPRINT_US.load(Ordering::Relaxed),
        texture_staging_buffer_us = crate::texture::STAGING_BUFFER_US.load(Ordering::Relaxed),
        texture_staging_pool_lock_us = crate::pool::STAGING_POOL_LOCK_US.load(Ordering::Relaxed),
        texture_staging_pool_growth_us = crate::pool::STAGING_POOL_GROWTH_US.load(Ordering::Relaxed),
        texture_staging_pool_clear_us = crate::pool::STAGING_POOL_CLEAR_US.load(Ordering::Relaxed),
        texture_staging_dedicated_us = crate::texture::STAGING_DEDICATED_US.load(Ordering::Relaxed),
        staging_speculative_pool_alloc_us = crate::texture::STAGING_POOL_ALLOCATIONS_US[0].load(Ordering::Relaxed),
        staging_demand_pool_alloc_us = crate::texture::STAGING_POOL_ALLOCATIONS_US[1].load(Ordering::Relaxed),
        staging_speculative_dedicated_alloc_us = crate::texture::STAGING_DEDICATED_ALLOCATIONS_US[0].load(Ordering::Relaxed),
        staging_demand_dedicated_alloc_us = crate::texture::STAGING_DEDICATED_ALLOCATIONS_US[1].load(Ordering::Relaxed),
        speculative_buffer_deferrals = SPECULATIVE_BUFFER_DEFERRALS.load(Ordering::Relaxed),
        submission_prepare_us = crate::host::STALL_TIMING_US[0].load(Ordering::Relaxed),
        submitter_wait_us = crate::host::STALL_TIMING_US[1].load(Ordering::Relaxed),
        relay_fence_wait_us = crate::host::STALL_TIMING_US[2].load(Ordering::Relaxed),
        texture_first_uploads_after_eviction = crate::texture::UPLOAD_TRAFFIC[1].load(Ordering::Relaxed),
        texture_first_upload_bytes_after_eviction = crate::texture::UPLOAD_TRAFFIC[2].load(Ordering::Relaxed),
        buffer_mirror_span_mib = crate::shadow::live_span_bytes() / MIB,
        spare_scratch_mib = crate::host::ALIVE[4].load(Ordering::Relaxed) / MIB,
        retired,
        retired_any,
        retired_resources,
        oldest_retired_ms = oldest_ms,
        queued,
        started,
        finished,
        "memory sample"
    );
}

/// True when OS commit headroom is critically low (< 1.5 GiB) or an allocation
/// just failed with `E_OUTOFMEMORY`, requiring emergency throttling.
pub(crate) fn critical_commit() -> bool {
    let now = crate::host::coarse_ms();
    now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) || sample_commit(now) < 1536 * MIB
}

/// A single warm staging block can be kept through moderate commit pressure.
pub(crate) fn keep_staging_reserve() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VICTORY_STAGING_RESERVE").as_deref() == Ok("1"))
        && !upload_refused()
        && sample_commit(crate::host::coarse_ms()) >= 512 * MIB
}

/// Enables extra texture eviction after three seconds below the commit reserve.
pub(crate) fn trim_textures_for_commit() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    static LOW_SINCE: AtomicU64 = AtomicU64::new(0);
    if !*ENABLED.get_or_init(|| std::env::var("VICTORY_TEXTURE_COMMIT_TRIM").as_deref() == Ok("1")) {
        return false;
    }
    let now = crate::host::coarse_ms().saturating_add(1);
    if sample_commit(now - 1) >= 1536 * MIB {
        LOW_SINCE.store(0, Ordering::Relaxed);
        return false;
    }
    match LOW_SINCE.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => false,
        Err(since) => now.saturating_sub(since) >= 3000,
    }
}

/// A soft per-frame admission limit. A single texture can cross it. The
/// queue retains unserved slots so they can be retried on later frames.
pub(crate) fn prefetch_limit(bytes: u64, pressure: bool, critical: bool, limited: bool, staging: u64) -> u64 {
    if critical {
        bytes.min(16 << 20)
    } else if pressure || (limited && staging >= GIB) {
        bytes.min(64 << 20)
    } else {
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bake_and_lookahead_leave_room_for_live_game_allocations() {
        for (commit, bake, lookahead, growth) in [
            (128 * MIB, false, false, false),
            (256 * MIB, true, false, false),
            (512 * MIB, true, true, false),
            (768 * MIB, true, true, false),
            (1024 * MIB, true, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit, true), bake);
            assert_eq!(has_upload_reserve(commit, false), lookahead);
            assert_eq!(has_growth_reserve(commit, false), growth);
        }
    }

    #[test]
    fn either_constraint_enters_pressure_and_both_must_recover() {
        assert!(next_pressure(false, 12 * GIB, 4 * GIB));
        assert!(next_pressure(false, 3 * GIB, 16 * GIB));
        assert!(next_pressure(true, 4 * GIB, 7 * GIB));
        assert!(next_pressure(true, 6 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 8 * GIB, 8 * GIB));
        assert!(!next_pressure(false, 7 * GIB, 7 * GIB));
    }

    #[test]
    fn prefetch_respects_existing_budgets_and_resumes_after_recovery() {
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB - 1), 512 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, false, 2 * GIB), 512 << 20);
        assert_eq!(prefetch_limit(0, true, true, false, 0), 0);
        assert_eq!(prefetch_limit(8 << 20, true, true, false, 0), 8 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, false, false, 0), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, true, false, 0), 16 << 20);
        assert_eq!(prefetch_limit(192 << 20, false, false, false, 0), 192 << 20);
    }
}
