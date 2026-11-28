//! The threads that submit to the card in Windows' multimedia class
//! scheduler, task "Games" (MMCSS): Windows raises them above the rest of
//! the PC's work, so other programs interrupt them less. Off by default.
//! `VICTORY_MMCSS=1` starts with it on; **Scroll Lock** turns it on and off
//! while the game runs (with one of our windows in front), and the log says
//! which. Each submitting thread joins or leaves at its next submission
//! (the call must come from the thread itself).
//!
//! It can make things worse as well as better: a raised thread that waits
//! for one of ours below it (the title's, refreshes, pipeline workers) gets
//! nothing from the raise, and Windows holds such threads back now and then
//! for the rest of the system (20 % by default). Judge it by A/B runs.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use windows_sys::Win32::System::Threading::{AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_SCROLL};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

/// Whether submitting threads should be in MMCSS now.
fn wanted() -> &'static AtomicBool {
    static WANTED: OnceLock<AtomicBool> = OnceLock::new();
    WANTED.get_or_init(|| AtomicBool::new(from_setting(std::env::var("VICTORY_MMCSS").ok().as_deref())))
}

fn from_setting(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// Threads in MMCSS now.
static JOINED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// This thread's MMCSS handle while it is in.
    static IN: Cell<usize> = const { Cell::new(0) };
}

/// Once a frame: Scroll Lock pressed (since the last frame, one of our
/// windows in front) turns it on or off.
pub(crate) fn poll_key() {
    static DOWN: AtomicBool = AtomicBool::new(false);
    // SAFETY: a plain query.
    let down = unsafe { GetAsyncKeyState(i32::from(VK_SCROLL)) } < 0;
    if DOWN.swap(down, Ordering::Relaxed) || !down {
        return;
    }
    let mut process = 0;
    // SAFETY: plain queries; room for the answer.
    unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut process) };
    if process != std::process::id() {
        return;
    }
    let now = !wanted().fetch_xor(true, Ordering::Relaxed);
    tracing::warn!(
        on = now,
        threads_in = JOINED.load(Ordering::Relaxed),
        "Scroll Lock: MMCSS (\"Games\") for the submitting threads {} (from their next submission)",
        if now { "ON" } else { "OFF" }
    );
}

/// At each submission, on the submitting thread: in or out of MMCSS as
/// wanted. One relaxed load when nothing changes.
pub(crate) fn submitting() {
    let want = wanted().load(Ordering::Relaxed);
    IN.with(|handle| apply(handle, want));
}

/// Joins (`want`) or leaves MMCSS on this thread, if not already so.
fn apply(handle: &Cell<usize>, want: bool) {
    match (handle.get(), want) {
        (0, true) => {
            let task: Vec<u16> = "Games".encode_utf16().chain([0]).collect();
            let mut index = 0u32;
            // SAFETY: a NUL-terminated task name and room for the index.
            let h = unsafe { AvSetMmThreadCharacteristicsW(task.as_ptr(), &mut index) };
            if h.is_null() {
                static SAID: AtomicBool = AtomicBool::new(false);
                if !SAID.swap(true, Ordering::Relaxed) {
                    tracing::warn!(error = %std::io::Error::last_os_error(), "MMCSS: this thread could not join \"Games\"");
                }
                return;
            }
            handle.set(h as usize);
            let n = JOINED.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(threads_in = n, "MMCSS: a submitting thread joined \"Games\"");
        }
        (h, false) if h != 0 => {
            // SAFETY: the handle this thread got from `AvSetMmThreadCharacteristicsW`.
            unsafe { AvRevertMmThreadCharacteristics(h as _) };
            handle.set(0);
            let n = JOINED.fetch_sub(1, Ordering::Relaxed) - 1;
            tracing::info!(threads_in = n, "MMCSS: a submitting thread left \"Games\"");
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_is_off_unless_asked() {
        assert!(!from_setting(None));
        assert!(!from_setting(Some("0")));
        assert!(!from_setting(Some("")));
        assert!(!from_setting(Some("yes")));
        assert!(from_setting(Some("1")));
        assert!(from_setting(Some(" 1 ")));
    }

    /// A real thread joins "Games", stays in when asked again, and leaves.
    #[test]
    fn a_thread_joins_and_leaves() {
        std::thread::spawn(|| {
            let handle = Cell::new(0);
            apply(&handle, true);
            let first = handle.get();
            assert_ne!(first, 0, "the \"Games\" task exists on this PC: {}", std::io::Error::last_os_error());
            apply(&handle, true);
            assert_eq!(handle.get(), first, "joined once");
            apply(&handle, false);
            assert_eq!(handle.get(), 0);
            apply(&handle, false);
            assert_eq!(handle.get(), 0);
        })
        .join()
        .unwrap();
    }
}
