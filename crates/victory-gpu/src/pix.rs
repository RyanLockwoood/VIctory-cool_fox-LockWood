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

struct Pix {
    report_counter: ReportCounter,
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
    tracing::info!(path = %path.display(), "VICTORY_PIX: memory counters go to PIX timing captures once a second");
    // SAFETY: the export's documented signature (pix3.h:
    // `void WINAPI PIXReportCounter(PCWSTR name, float value)`).
    Some(Pix { report_counter: unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, ReportCounter>(report) } })
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

#[cfg(test)]
mod tests {
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
    }
}
