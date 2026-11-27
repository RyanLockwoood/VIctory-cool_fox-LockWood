//! Diagnostics: why texture data is read, detiled and uploaded again.
//!
//! In the opening cutscene (`VIctory-bench`, batch base-20261127-051611)
//! 202 GB of texture uploads were recorded in 380 s, about nine times the
//! texture pool, and the frames lost to stalls followed the time of the
//! texture refreshes (r = 0.8). These counters split that work:
//!
//! - refreshes by the way a change was found: a write the write watch saw
//!   (the texture is read and detiled whole, without a hash), a partly
//!   mapped texture read again (every 30 frames at most, hashed), a guarded
//!   texture in view memory (hashed), or any other (hashed); with the bytes
//!   read, the bytes found changed, and for watched writes the bytes of the
//!   64 KiB pages actually written;
//! - uploads by reason: the first of a texture (by its memory address),
//!   its contents changed since the last upload, its levels judged anew,
//!   a level left without tiles asked again, a one-shot bake, or a new
//!   staging buffer for a texture uploaded before;
//! - the bytes of uploads by how often that texture had been uploaded
//!   before (1st, 2nd-3rd, 4th-9th, 10th and later).
//!
//! Logged per frame with the copy statistics. Diagnostics builds only.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Clone, Copy)]
pub(crate) enum T {
    /// Watched path: refreshes, bytes read, bytes of pages written.
    WatchedN,
    WatchedRead,
    WatchedWritten,
    /// Partly mapped path: refreshes, bytes hashed, bytes changed, reads
    /// that found nothing but zeros.
    PartialN,
    PartialRead,
    PartialChanged,
    PartialZero,
    /// Guarded (view memory) path: refreshes, bytes hashed, bytes changed.
    GuardedN,
    GuardedRead,
    GuardedChanged,
    /// Any other hashed refresh: refreshes, bytes hashed, bytes changed.
    OtherN,
    OtherRead,
    OtherChanged,
    /// Uploads by reason, count and bytes.
    UpFirst,
    UpFirstBytes,
    UpChanged,
    UpChangedBytes,
    UpLevels,
    UpLevelsBytes,
    UpRetry,
    UpRetryBytes,
    UpBake,
    UpBakeBytes,
    UpAgain,
    UpAgainBytes,
    /// Upload bytes by the texture's upload count at that upload.
    Rep1Bytes,
    Rep2To3Bytes,
    Rep4To9Bytes,
    Rep10Bytes,
    /// Guarded textures read to be hashed, by why (bytes): the guard saw a
    /// write (or it was the first asking), the eighth-frame turn, the
    /// texture is hot (new, or changed unseen lately), a bake's exact read;
    /// and of all of them, those whose staging buffer had never been hashed.
    GuardWhyWritten,
    GuardWhyTurn,
    GuardWhyHot,
    GuardWhyExact,
    GuardFresh,
    /// Refreshes that found the memory changed against the staging buffer
    /// and detiled it, while it hashes the same as what the host copy holds
    /// (the last upload's): read, detiled and uploaded for nothing.
    SameAsCardN,
    SameAsCardBytes,
    Len,
}

static COUNT: [AtomicU64; T::Len as usize] = [const { AtomicU64::new(0) }; T::Len as usize];

pub(crate) fn add(t: T, n: u64) {
    COUNT[t as usize].fetch_add(n, Relaxed);
}

/// Why an upload was recorded (see the module's notes).
#[derive(Clone, Copy)]
pub(crate) enum Why {
    Changed,
    Levels,
    Retry,
    Bake,
    NewStaging,
}

/// Upload counts by texture memory address since the start.
static SEEN: Mutex<Option<HashMap<u64, u32>>> = Mutex::new(None);

/// Notes one recorded upload of `bytes` of the texture at `memory`.
pub(crate) fn upload(memory: u64, bytes: u64, why: Why) {
    let n = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let count = seen.get_or_insert_with(HashMap::new).entry(memory).or_insert(0);
        *count = count.saturating_add(1);
        *count
    };
    let (c, b) = if n == 1 {
        (T::UpFirst, T::UpFirstBytes)
    } else {
        match why {
            Why::Changed => (T::UpChanged, T::UpChangedBytes),
            Why::Levels => (T::UpLevels, T::UpLevelsBytes),
            Why::Retry => (T::UpRetry, T::UpRetryBytes),
            Why::Bake => (T::UpBake, T::UpBakeBytes),
            Why::NewStaging => (T::UpAgain, T::UpAgainBytes),
        }
    };
    add(c, 1);
    add(b, bytes);
    add(
        match n {
            1 => T::Rep1Bytes,
            2..=3 => T::Rep2To3Bytes,
            4..=9 => T::Rep4To9Bytes,
            _ => T::Rep10Bytes,
        },
        bytes,
    );
}

/// The per-frame line; counters go back to 0.
pub(crate) fn log(frames: f64) {
    if frames <= 0.0 {
        return;
    }
    let c: Vec<f64> = COUNT.iter().map(|a| a.swap(0, Relaxed) as f64 / frames).collect();
    let mb = |t: T| c[t as usize] / 1e6;
    let n = |t: T| c[t as usize];
    let textures = SEEN.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |m| m.len());
    tracing::info!(
        "texture work per frame: refreshes by how a change was found: watched write {:.1}x {:.1} MB read \
         ({:.1} MB of written pages); partly mapped {:.1}x {:.1} MB hashed, {:.1} MB changed, {:.1} all zeros; \
         guarded {:.1}x {:.1} MB hashed, {:.1} MB changed; other {:.1}x {:.1} MB hashed, {:.1} MB changed. \
         uploads by reason: first {:.1}x {:.1} MB, changed {:.1}x {:.1} MB, levels judged anew {:.1}x {:.1} MB, \
         level asked again {:.1}x {:.1} MB, bake {:.1}x {:.1} MB, new staging for an uploaded texture {:.1}x {:.1} MB. \
         upload MB by the texture's upload count: 1st {:.1}, 2nd-3rd {:.1}, 4th-9th {:.1}, 10th+ {:.1}; \
         {} textures uploaded since the start. guarded MB hashed because: written {:.1}, turn {:.1}, hot {:.1}, \
         exact {:.1} ({:.1} MB of never-hashed staging buffers). detiled although the same as the host copy holds: \
         {:.1}x {:.1} MB",
        n(T::WatchedN),
        mb(T::WatchedRead),
        mb(T::WatchedWritten),
        n(T::PartialN),
        mb(T::PartialRead),
        mb(T::PartialChanged),
        n(T::PartialZero),
        n(T::GuardedN),
        mb(T::GuardedRead),
        mb(T::GuardedChanged),
        n(T::OtherN),
        mb(T::OtherRead),
        mb(T::OtherChanged),
        n(T::UpFirst),
        mb(T::UpFirstBytes),
        n(T::UpChanged),
        mb(T::UpChangedBytes),
        n(T::UpLevels),
        mb(T::UpLevelsBytes),
        n(T::UpRetry),
        mb(T::UpRetryBytes),
        n(T::UpBake),
        mb(T::UpBakeBytes),
        n(T::UpAgain),
        mb(T::UpAgainBytes),
        mb(T::Rep1Bytes),
        mb(T::Rep2To3Bytes),
        mb(T::Rep4To9Bytes),
        mb(T::Rep10Bytes),
        textures,
        mb(T::GuardWhyWritten),
        mb(T::GuardWhyTurn),
        mb(T::GuardWhyHot),
        mb(T::GuardWhyExact),
        mb(T::GuardFresh),
        n(T::SameAsCardN),
        mb(T::SameAsCardBytes),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uploads_are_classed_by_count_and_reason() {
        let before: Vec<u64> = COUNT.iter().map(|a| a.load(Relaxed)).collect();
        let delta = |t: T, before: &[u64]| COUNT[t as usize].load(Relaxed) - before[t as usize];
        let at = 0xdead_0000_u64;
        upload(at, 100, Why::Changed); // first ever: counted as first
        upload(at, 100, Why::Changed); // 2nd
        upload(at, 100, Why::Levels); // 3rd
        upload(at, 100, Why::NewStaging); // 4th
        assert_eq!(delta(T::UpFirst, &before), 1);
        assert_eq!(delta(T::UpChanged, &before), 1);
        assert_eq!(delta(T::UpLevels, &before), 1);
        assert_eq!(delta(T::UpAgain, &before), 1);
        assert_eq!(delta(T::Rep1Bytes, &before), 100);
        assert_eq!(delta(T::Rep2To3Bytes, &before), 200);
        assert_eq!(delta(T::Rep4To9Bytes, &before), 100);
    }
}
