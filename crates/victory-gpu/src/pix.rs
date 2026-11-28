//! PIX on Windows: the layer's memory, as counters on a timing capture's
//! timeline (`PIXReportCounter` from Microsoft's `WinPixEventRuntime.dll`,
//! MIT licence, NuGet package `WinPixEventRuntime`).
//!
//! `VICTORY_PIX=1`: the DLL is loaded from beside `victory.exe`;
//! `VICTORY_PIX=<path to the DLL>`: from there. Never from the search
//! path. Unset (the default): nothing is loaded and every call here is one
//! relaxed check. With it, the once-a-second sweep reports the counters
//! below, so a PIX timing capture shows when and where RAM climbs (the
//! working set, what is free, and the layer's own pools beside them).

use std::sync::OnceLock;

type ReportCounter = unsafe extern "system" fn(name: *const u16, value: f32);
/// pix3_win.h: `void WINAPI PIXRecordMemoryAllocationEvent(USHORT
/// allocatorId, void* baseAddress, size_t size, UINT64 metadata)`, and the
/// free event with the same arguments.
type MemoryEvent = unsafe extern "system" fn(allocator: u16, base: *const u8, size: usize, metadata: u64);

struct Pix {
    report_counter: ReportCounter,
    allocated: Option<MemoryEvent>,
    freed: Option<MemoryEvent>,
    woke: Option<Woke>,
}

/// pix3_win.h: `void WINAPI PIXNotifyWakeFromFenceSignal(HANDLE event)`.
type Woke = unsafe extern "system" fn(event: *mut core::ffi::c_void);

/// The thread woke from waiting on `event`, which a fence's
/// `SetEventOnCompletion` signals (with `VICTORY_PIX`): a PIX timing capture
/// then joins the wait to the work on the card that ended it.
pub(crate) fn woke(event: windows::Win32::Foundation::HANDLE) {
    if let Some(f) = pix().and_then(|p| p.woke)
        && !event.is_invalid()
    {
        // SAFETY: the export's signature; the handle is only recorded.
        unsafe { f(event.0) };
    }
}

/// Allocator ids in PIX's memory view: staging blocks of textures (64 MiB
/// each, in the PC's memory) and host copies of title memory.
pub(crate) const STAGING_BLOCKS: u16 = 1;
pub(crate) const HOST_COPIES: u16 = 2;

/// `size` bytes at `base` are the layer's, from now on, as `allocator`'s
/// (with `VICTORY_PIX`): a PIX timing capture's memory view shows each with
/// the call stack that made it.
pub(crate) fn allocated(allocator: u16, base: *const u8, size: u64) {
    if let Some(f) = pix().and_then(|p| p.allocated)
        && !base.is_null()
    {
        // SAFETY: the export's signature; the address is only recorded.
        unsafe { f(allocator, base, size as usize, 0) };
    }
}

/// The matching free of [`allocated`].
pub(crate) fn freed(allocator: u16, base: *const u8, size: u64) {
    if let Some(f) = pix().and_then(|p| p.freed)
        && !base.is_null()
    {
        // SAFETY: as above.
        unsafe { f(allocator, base, size as usize, 0) };
    }
}

fn pix() -> Option<&'static Pix> {
    static PIX: OnceLock<Option<Pix>> = OnceLock::new();
    PIX.get_or_init(load).as_ref()
}

/// Where the DLL is, from `VICTORY_PIX`: `None` if unset, `0` or empty.
fn wanted(value: Option<std::ffi::OsString>) -> Option<std::path::PathBuf> {
    let value = value?;
    if value.is_empty() || value == "0" {
        return None;
    }
    if value == "1" {
        return Some(std::env::current_exe().ok()?.with_file_name("WinPixEventRuntime.dll"));
    }
    Some(value.into())
}

fn load() -> Option<Pix> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
    };
    let path = wanted(std::env::var_os("VICTORY_PIX"))?;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // SAFETY: a NUL-terminated full path; only its own folder
    // and System32 for what it loads in turn.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32) };
    if module.is_null() {
        tracing::warn!(path = %path.display(), "VICTORY_PIX: WinPixEventRuntime.dll did not load; no PIX counters");
        return None;
    }
    // SAFETY: a loaded module and a NUL-terminated name.
    let Some(report) = (unsafe { GetProcAddress(module, c"PIXReportCounter".as_ptr().cast()) }) else {
        tracing::warn!(path = %path.display(), "VICTORY_PIX: the DLL has no PIXReportCounter; no PIX counters");
        return None;
    };
    tracing::info!(path = %path.display(), "VICTORY_PIX: memory counters and allocations go to PIX timing captures");
    let event = |name: &std::ffi::CStr| {
        // SAFETY: a loaded module and a NUL-terminated name; the exports'
        // documented signature (pix3_win.h).
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, MemoryEvent>(f) })
    };
    Some(Pix {
        // SAFETY: the export's documented signature (pix3.h:
        // `void WINAPI PIXReportCounter(PCWSTR name, float value)`).
        report_counter: unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, ReportCounter>(report) },
        allocated: event(c"PIXRecordMemoryAllocationEvent"),
        freed: event(c"PIXRecordMemoryFreeEvent"),
        // SAFETY: a loaded module and a NUL-terminated name; the export's
        // documented signature (pix3_win.h).
        woke: unsafe { GetProcAddress(module, c"PIXNotifyWakeFromFenceSignal".as_ptr().cast()) }
            .map(|f| unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, Woke>(f) }),
    })
}

/// Whether counters go to PIX.
pub(crate) fn on() -> bool {
    pix().is_some()
}

/// One counter's value now, under `name` (UTF-16, NUL-terminated).
fn counter(name: &[u16], value: f64) {
    debug_assert_eq!(name.last(), Some(&0));
    if let Some(p) = pix() {
        // SAFETY: a NUL-terminated name that outlives the call.
        unsafe { (p.report_counter)(name.as_ptr(), value as f32) };
    }
}

/// A NUL-terminated UTF-16 name, made once.
macro_rules! name {
    ($text:literal) => {{
        static NAME: OnceLock<Vec<u16>> = OnceLock::new();
        NAME.get_or_init(|| $text.encode_utf16().chain([0]).collect()).as_slice()
    }};
}

/// The memory counters, in MiB (from the once-a-second sweep).
pub(crate) fn report_memory() {
    if !on() {
        return;
    }
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const MIB: f64 = (1u64 << 20) as f64;
    let mut process = PROCESS_MEMORY_COUNTERS_EX { cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32, ..Default::default() };
    // SAFETY: our own process and a sized structure.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut process).cast(), process.cb) } != 0 {
        counter(name!("Victory: private (MiB)"), process.PrivateUsage as f64 / MIB);
        counter(name!("Victory: working set (MiB)"), process.WorkingSetSize as f64 / MIB);
    }
    // SAFETY: plain data the call fills in.
    let mut system: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    system.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    if unsafe { GlobalMemoryStatusEx(&mut system) } != 0 {
        counter(name!("PC: RAM free (MiB)"), system.ullAvailPhys as f64 / MIB);
        counter(name!("PC: commit left (MiB)"), system.ullAvailPageFile as f64 / MIB);
    }
    let [texture_blocks, staging_blocks] = crate::pool::memory();
    counter(name!("Victory: texture pool (MiB)"), texture_blocks as f64);
    counter(name!("Victory: staging pool (MiB)"), staging_blocks as f64);
    let (arena, idle_arena) = crate::upload::arena_bytes();
    counter(name!("Victory: upload arena (MiB)"), arena as f64 / MIB);
    counter(name!("Victory: upload arena idle (MiB)"), idle_arena as f64 / MIB);
    counter(name!("Victory: texture staging alive (MiB)"), crate::texture::live_staging_bytes() as f64 / MIB);
    counter(name!("Victory: host copies of title memory, span (MiB)"), crate::shadow::live_span_bytes() as f64 / MIB);
    let (retired, ..) = crate::host::retirement_sample();
    counter(name!("Victory: retired objects waiting"), retired as f64);
}

/// Stutters caught in PIX timing captures by themselves
/// (`VICTORY_PIX_STUTTER=<ms>`, off by default): a capture runs from the
/// first frame on into a ring in a file (PIX's `FileCircular` storage, the
/// newest `VICTORY_PIX_STUTTER_MB` MiB, 2048 unless set); when a frame took
/// longer than `<ms>`, it is ended two seconds later on a thread of its own,
/// which writes `stutter-<n>-<seconds>.wpix` into `VICTORY_PIX_STUTTER_DIR`
/// (`pix-stutters` beside `victory.exe` unless set), and the next one
/// begins. At most `VICTORY_PIX_STUTTER_MAX` (5) a run. Needs PIX on
/// Windows installed (`WinPixTimingCapturer.dll` from the newest
/// `%ProgramFiles%\Microsoft PIX\*`) and `victory.exe` run as administrator
/// (timing captures trace the kernel). With `VICTORY_PIX` too, the captures
/// have the layer's counters and allocation events.
pub(crate) mod stutter {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    /// pix3.h `PIXCaptureParameters::TimingCaptureParameters`, field for
    /// field (the Xbox-only words included: they are in the layout).
    #[repr(C)]
    #[derive(Default)]
    struct TimingParameters {
        file_name: usize,
        maximum_tooling_memory_mb: u32,
        capture_storage: i32,
        capture_gpu_timing: i32,
        capture_callstacks: i32,
        capture_cpu_samples: i32,
        cpu_samples_per_second: u32,
        capture_file_io: i32,
        capture_virtual_alloc_events: i32,
        capture_heap_alloc_events: i32,
        capture_xmem_events: i32,
        capture_pix_mem_events: i32,
        capture_page_fault_events: i32,
        capture_video_frames: i32,
    }
    /// `PIXCaptureStorage::FileCircular` (PC only).
    const FILE_CIRCULAR: i32 = 2;

    type Begin = unsafe extern "system" fn(parameters: *const TimingParameters, size: u64) -> i32;
    type End = unsafe extern "system" fn(discard: i32) -> i32;

    struct Capturer {
        begin: Begin,
        end: End,
    }

    /// What the frames do to the capture ([`step`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum Action {
        Nothing,
        Begin,
        /// A frame of this many milliseconds: end the capture in two seconds.
        Mark(u64),
        End,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum State {
        Idle,
        Capturing { since: Instant },
        Marked { since: Instant, at: Instant },
        Saving,
        Done,
    }

    /// How long a capture runs before a stutter counts (its ring has the
    /// time before it), and how long it goes on after one.
    const WARM: Duration = Duration::from_secs(3);
    const AFTER: Duration = Duration::from_secs(2);

    /// The decision for a frame that took `frame` at `now`.
    pub(super) fn step(state: State, now: Instant, frame: Duration, threshold: Duration, taken: u32, most: u32) -> Action {
        match state {
            State::Idle if taken < most => Action::Begin,
            State::Capturing { since } if frame > threshold && now.duration_since(since) >= WARM => {
                Action::Mark(frame.as_millis() as u64)
            }
            State::Marked { at, .. } if now.duration_since(at) >= AFTER => Action::End,
            _ => Action::Nothing,
        }
    }

    fn threshold() -> Option<Duration> {
        static MS: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
        MS.get_or_init(|| std::env::var("VICTORY_PIX_STUTTER").ok()?.trim().parse::<u64>().ok().filter(|&ms| ms >= 34))
            .map(Duration::from_millis)
    }

    fn number(var: &str, default: u32) -> u32 {
        std::env::var(var).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
    }

    /// The newest `WinPixTimingCapturer.dll` under `%ProgramFiles%\Microsoft PIX`.
    pub(super) fn newest_capturer(root: &std::path::Path) -> Option<std::path::PathBuf> {
        let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(root)
            .ok()?
            .flatten()
            .map(|e| e.path().join("WinPixTimingCapturer.dll"))
            .filter(|p| p.is_file())
            .collect();
        found.sort();
        found.pop()
    }

    fn capturer() -> Option<&'static Capturer> {
        static CAPTURER: std::sync::OnceLock<Option<Capturer>> = std::sync::OnceLock::new();
        CAPTURER
            .get_or_init(|| {
                use std::os::windows::ffi::OsStrExt;
                use windows_sys::Win32::System::LibraryLoader::{
                    GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LoadLibraryExW,
                };
                let root = std::path::PathBuf::from(std::env::var_os("ProgramFiles")?).join("Microsoft PIX");
                let Some(path) = newest_capturer(&root) else {
                    tracing::warn!("VICTORY_PIX_STUTTER: no WinPixTimingCapturer.dll under {}; no captures", root.display());
                    return None;
                };
                let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
                // SAFETY: a NUL-terminated full path; the flags pix3_win.h uses.
                let module = unsafe {
                    LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
                };
                if module.is_null() {
                    tracing::warn!(path = %path.display(), "VICTORY_PIX_STUTTER: the timing capturer did not load");
                    return None;
                }
                // SAFETY: a loaded module and NUL-terminated names.
                let (begin, end) = unsafe {
                    (
                        GetProcAddress(module, c"BeginProgrammaticTimingCapture".as_ptr().cast())?,
                        GetProcAddress(module, c"EndProgrammaticTimingCapture".as_ptr().cast())?,
                    )
                };
                tracing::info!(path = %path.display(), "VICTORY_PIX_STUTTER: timing captures of slow frames");
                // SAFETY: the exports' signatures as pix3_win.h calls them.
                Some(unsafe {
                    Capturer {
                        begin: std::mem::transmute::<unsafe extern "system" fn() -> isize, Begin>(begin),
                        end: std::mem::transmute::<unsafe extern "system" fn() -> isize, End>(end),
                    }
                })
            })
            .as_ref()
    }

    struct Run {
        state: State,
        last: Option<Instant>,
        taken: u32,
        /// The running capture's file (UTF-16, NUL-terminated: PIX keeps the
        /// pointer while it runs) and the slowest frame seen in it.
        file: Vec<u16>,
        path: std::path::PathBuf,
        slowest: u64,
    }

    static RUN: Mutex<Run> =
        Mutex::new(Run { state: State::Idle, last: None, taken: 0, file: Vec::new(), path: std::path::PathBuf::new(), slowest: 0 });

    /// Called once a frame, when it is presented.
    pub(crate) fn presented() {
        let Some(threshold) = threshold() else {
            return;
        };
        let now = Instant::now();
        let mut run = RUN.lock().unwrap_or_else(|e| e.into_inner());
        let frame = run.last.map_or(Duration::ZERO, |last| now.duration_since(last));
        run.last = Some(now);
        let most = number("VICTORY_PIX_STUTTER_MAX", 5);
        match step(run.state, now, frame, threshold, run.taken, most) {
            Action::Nothing => {}
            Action::Begin => begin(&mut run, now),
            Action::Mark(ms) => {
                if let State::Capturing { since } = run.state {
                    run.state = State::Marked { since, at: now };
                }
                run.slowest = run.slowest.max(ms);
                tracing::warn!(ms, frame = crate::frame::current(), "VICTORY_PIX_STUTTER: a slow frame; the capture ends in two seconds");
            }
            Action::End => end(&mut run),
        }
        if let State::Marked { .. } = run.state {
            run.slowest = run.slowest.max(frame.as_millis() as u64);
        }
    }

    fn begin(run: &mut Run, now: Instant) {
        let Some(capturer) = capturer() else {
            run.state = State::Done;
            return;
        };
        let dir = std::env::var_os("VICTORY_PIX_STUTTER_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| {
            std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.join("pix-stutters"))).unwrap_or_default()
        });
        let _ = std::fs::create_dir_all(&dir);
        let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        run.path = dir.join(format!("stutter-{}-{seconds}.wpix", run.taken + 1));
        {
            use std::os::windows::ffi::OsStrExt;
            run.file = run.path.as_os_str().encode_wide().chain([0]).collect();
        }
        let parameters = TimingParameters {
            file_name: run.file.as_ptr() as usize,
            maximum_tooling_memory_mb: number("VICTORY_PIX_STUTTER_MB", 2048),
            capture_storage: FILE_CIRCULAR,
            capture_gpu_timing: 1,
            capture_callstacks: 1,
            capture_cpu_samples: 1,
            cpu_samples_per_second: 4000,
            capture_file_io: 1,
            capture_pix_mem_events: 1,
            ..Default::default()
        };
        // SAFETY: the export's signature; the parameters and the file name
        // (kept in `run` until the capture ends) outlive what PIX needs.
        let hr = unsafe { (capturer.begin)(&parameters, size_of::<TimingParameters>() as u64) };
        if hr < 0 {
            // E_ACCESSDENIED: timing captures trace the kernel, which needs
            // the process to run as administrator.
            let why = if hr as u32 == 0x8007_0005 { " (access denied: start victory.exe as administrator)" } else { "" };
            tracing::warn!(hr = format_args!("{hr:#x}"), "VICTORY_PIX_STUTTER: the timing capture did not begin{why}; no more");
            run.state = State::Done;
            return;
        }
        run.slowest = 0;
        run.state = State::Capturing { since: now };
    }

    fn end(run: &mut Run) {
        let Some(capturer) = capturer() else {
            return;
        };
        run.state = State::Saving;
        run.taken += 1;
        let (path, slowest, end) = (run.path.clone(), run.slowest, capturer.end);
        // The file name stays alive in `run` until the next capture begins,
        // which waits for this one to be saved.
        let saved = std::thread::Builder::new().name("victory-pix-stutter".into()).spawn(move || {
            // SAFETY: the export's signature; a capture is running.
            let hr = unsafe { end(0) };
            if hr < 0 {
                tracing::warn!(hr = format_args!("{hr:#x}"), path = %path.display(), "VICTORY_PIX_STUTTER: the capture did not save");
            } else {
                tracing::warn!(path = %path.display(), slowest_ms = slowest, "VICTORY_PIX_STUTTER: a slow frame's timing capture saved");
            }
            let mut run = RUN.lock().unwrap_or_else(|e| e.into_inner());
            run.state = State::Idle;
        });
        if saved.is_err() {
            run.state = State::Done;
        }
    }

    /// For the ignored test: one capture into `path`, ended after `for_ms`.
    #[cfg(test)]
    pub(super) fn capture_once(path: &std::path::Path, for_ms: u64) -> Result<(), String> {
        use std::os::windows::ffi::OsStrExt;
        let capturer = capturer().ok_or("no timing capturer")?;
        let file: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let parameters = TimingParameters {
            file_name: file.as_ptr() as usize,
            maximum_tooling_memory_mb: 256,
            capture_storage: FILE_CIRCULAR,
            capture_gpu_timing: 1,
            capture_callstacks: 1,
            capture_cpu_samples: 1,
            cpu_samples_per_second: 4000,
            ..Default::default()
        };
        // SAFETY: as in `begin`.
        let hr = unsafe { (capturer.begin)(&parameters, size_of::<TimingParameters>() as u64) };
        if hr < 0 {
            return Err(format!("begin {hr:#x}"));
        }
        let spin = Instant::now();
        while spin.elapsed() < Duration::from_millis(for_ms) {
            std::hint::black_box((0..1000).sum::<u64>());
        }
        // SAFETY: as in `end`.
        let hr = unsafe { (capturer.end)(0) };
        if hr < 0 {
            return Err(format!("end {hr:#x}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// A capture begins at once, counts a slow frame only once it is warm,
    /// ends two seconds after it, and no more begin than allowed.
    #[test]
    fn stutters_are_marked_once_warm_and_saved_after() {
        use super::stutter::{Action, State, step};
        use std::time::{Duration, Instant};
        let ms = Duration::from_millis;
        let t0 = Instant::now();
        let limit = ms(100);
        assert_eq!(step(State::Idle, t0, ms(16), limit, 0, 5), Action::Begin);
        assert_eq!(step(State::Idle, t0, ms(16), limit, 5, 5), Action::Nothing);
        let capturing = State::Capturing { since: t0 };
        assert_eq!(step(capturing, t0 + ms(1000), ms(500), limit, 0, 5), Action::Nothing, "not warm yet");
        assert_eq!(step(capturing, t0 + ms(4000), ms(33), limit, 0, 5), Action::Nothing, "a normal frame");
        assert_eq!(step(capturing, t0 + ms(4000), ms(250), limit, 0, 5), Action::Mark(250));
        let marked = State::Marked { since: t0, at: t0 + ms(4000) };
        assert_eq!(step(marked, t0 + ms(5000), ms(900), limit, 0, 5), Action::Nothing, "still the two seconds after");
        assert_eq!(step(marked, t0 + ms(6000), ms(16), limit, 0, 5), Action::End);
        assert_eq!(step(State::Saving, t0 + ms(7000), ms(500), limit, 0, 5), Action::Nothing);
        assert_eq!(step(State::Done, t0 + ms(7000), ms(500), limit, 0, 5), Action::Nothing);
    }

    #[test]
    fn the_newest_capturer_is_found() {
        let root = std::env::temp_dir().join(format!("victory-pix-test-{}", std::process::id()));
        for v in ["2408.09", "2611.19", "2503.01"] {
            std::fs::create_dir_all(root.join(v)).unwrap();
            std::fs::write(root.join(v).join("WinPixTimingCapturer.dll"), b"").unwrap();
        }
        std::fs::create_dir_all(root.join("2700.01")).unwrap(); // no DLL in it
        assert_eq!(super::stutter::newest_capturer(&root).unwrap(), root.join("2611.19").join("WinPixTimingCapturer.dll"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A real timing capture with the installed PIX, written to a file:
    /// the parameters' layout is the one the capturer reads. Run on its own:
    /// `cargo test --release -p victory-gpu --lib a_real_timing_capture -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs PIX on Windows installed and an elevated test process; writes a capture file"]
    fn a_real_timing_capture_is_written() {
        let path = std::env::temp_dir().join(format!("victory-pix-capture-{}.wpix", std::process::id()));
        super::stutter::capture_once(&path, 1500).unwrap();
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        eprintln!("timing capture {} bytes at {}", size, path.display());
        assert!(size > 0, "the capture file is written");
        let _ = std::fs::remove_file(&path);
    }
    use super::*;

    #[test]
    fn the_dll_is_looked_for_only_where_asked() {
        assert_eq!(wanted(None), None);
        assert_eq!(wanted(Some("".into())), None);
        assert_eq!(wanted(Some("0".into())), None);
        let beside = wanted(Some("1".into())).unwrap();
        assert_eq!(beside.file_name().unwrap(), "WinPixEventRuntime.dll");
        assert_eq!(beside.parent(), std::env::current_exe().unwrap().parent());
        assert_eq!(wanted(Some(r"D:\pix\WinPixEventRuntime.dll".into())).unwrap(), std::path::PathBuf::from(r"D:\pix\WinPixEventRuntime.dll"));
    }

    /// The real DLL, if the test is pointed at it
    /// (`VICTORY_PIX_TEST_DLL=<path>`): it loads, its counter export is
    /// found, and reporting with no capture running does nothing harmful.
    #[test]
    fn the_runtime_loads_and_takes_counters() {
        let Some(path) = std::env::var_os("VICTORY_PIX_TEST_DLL") else {
            return;
        };
        // SAFETY: the test sets the variable for this process before the
        // first use of `pix()`; no other test reads it.
        unsafe { std::env::set_var("VICTORY_PIX", &path) };
        assert!(on(), "WinPixEventRuntime.dll at {path:?} must load");
        report_memory();
        counter(name!("Victory: test counter"), 1.5);
        assert!(pix().is_some_and(|p| p.allocated.is_some() && p.freed.is_some()), "the memory event exports are found");
        assert!(pix().is_some_and(|p| p.woke.is_some()), "the fence wake export is found");
        // SAFETY: a plain event of our own.
        let event = unsafe { windows::Win32::System::Threading::CreateEventW(None, false, true, None) }.unwrap();
        woke(event);
        woke(windows::Win32::Foundation::HANDLE::default());
        // SAFETY: our own handle.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(event);
        }
        let block = vec![0u8; 1 << 20];
        allocated(STAGING_BLOCKS, block.as_ptr(), block.len() as u64);
        freed(STAGING_BLOCKS, block.as_ptr(), block.len() as u64);
        allocated(HOST_COPIES, std::ptr::null(), 0);
    }
}
