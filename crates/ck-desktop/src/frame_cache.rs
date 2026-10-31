//! RAM cache for rendered/displayed result frames, with resident-frame bookkeeping
//! so the timeline can show what is actually cached (v1 result cache lane).

use crate::ram_cache::{MemoryCost, RamCache};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

/// One cached preview/result frame (the pixels the viewer presents).
#[derive(Clone, Debug)]
pub struct CachedPreview {
    pub width: usize,
    pub height: usize,
    pub rgba: Arc<[u8]>,
    pub path: Option<PathBuf>,
    pub background_applied: bool,
    pub despill_applied: bool,
    pub despill_unavailable: bool,
    pub source_rgba: Option<Arc<[u8]>>,
}

impl MemoryCost for CachedPreview {
    fn memory_cost(&self) -> usize {
        self.rgba
            .len()
            .saturating_add(self.source_rgba.as_ref().map_or(0, |bytes| bytes.len()))
            .saturating_add(256)
    }
}

/// `<composition>` of a `<composition>:<frame>:<masks>` preview key.
fn composition_prefix(key: &str) -> Option<&str> {
    let mut parts = key.rsplitn(3, ':');
    parts.next()?;
    parts.next()?;
    parts.next()
}

/// Bounded LRU of presented result frames keyed by the preview request key.
pub struct ResultRamCache {
    cache: RamCache<String, CachedPreview>,
    /// Source frame shown by each retained key.
    frame_by_key: HashMap<String, u64>,
    /// Number of retained keys currently showing each source frame.
    frame_counts: HashMap<u64, usize>,
    /// Frames whose last retained key was evicted since the last drain.
    evicted_frames: BTreeSet<u64>,
}

impl ResultRamCache {
    pub fn new(byte_limit: usize, item_limit: usize) -> Self {
        Self {
            cache: RamCache::new(byte_limit, item_limit),
            frame_by_key: HashMap::new(),
            frame_counts: HashMap::new(),
            evicted_frames: BTreeSet::new(),
        }
    }

    /// Cached frame for a preview request key, refreshing its LRU recency.
    pub fn get(&mut self, key: &str) -> Option<CachedPreview> {
        self.cache.get_cloned(&key.to_string())
    }

    /// Whether a key is retained (does not touch LRU recency).
    pub fn contains(&self, key: &str) -> bool {
        self.frame_by_key.contains_key(key)
    }

    /// Retain a presented frame; returns whether it fitted the budget. Older versions of the
    /// same frame in the same composition (keys `<composition>:<frame>:<masks>` that differ only
    /// in the mask part) are dropped first, so live mask edits do not fill the budget.
    pub fn insert(&mut self, key: String, frame: u64, value: CachedPreview) -> bool {
        if let Some(prefix) = composition_prefix(&key) {
            let stale = self
                .frame_by_key
                .iter()
                .filter(|(other, shown)| {
                    **shown == frame
                        && other.as_str() != key
                        && composition_prefix(other) == Some(prefix)
                })
                .map(|(other, _)| other.clone())
                .collect::<Vec<_>>();
            for other in stale {
                self.remove(&other);
            }
        }
        let frame_key = key.clone();
        let outcome = self.cache.insert(key, value);
        if !outcome.retained {
            return false;
        }
        if let Some(previous) = self.frame_by_key.insert(frame_key, frame) {
            self.decrement_frame(previous);
        }
        *self.frame_counts.entry(frame).or_insert(0) += 1;
        for (evicted_key, _) in &outcome.evicted {
            if let Some(previous) = self.frame_by_key.remove(evicted_key) {
                self.decrement_frame(previous);
                self.evicted_frames.insert(previous);
            }
        }
        true
    }

    /// Drop one retained composition key (e.g. a fallback frame rendered while
    /// the original plate was briefly unavailable). Returns whether it existed.
    pub fn remove(&mut self, key: &str) -> bool {
        if self.cache.remove(&key.to_string()).is_none() {
            return false;
        }
        if let Some(frame) = self.frame_by_key.remove(key) {
            self.decrement_frame(frame);
        }
        true
    }

    /// Drop every retained key showing `frame` (its artifact was recalculated).
    /// Returns how many keys were removed.
    pub fn remove_frame(&mut self, frame: u64) -> usize {
        let keys = self
            .frame_by_key
            .iter()
            .filter(|(_, shown)| **shown == frame)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in &keys {
            self.remove(key);
        }
        keys.len()
    }

    /// Frames evicted since the last call (used to keep the auto-cache view of
    /// resident frames accurate).
    pub fn take_evicted_frames(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.evicted_frames)
            .into_iter()
            .collect()
    }

    fn decrement_frame(&mut self, frame: u64) {
        if let Some(count) = self.frame_counts.get_mut(&frame) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.frame_counts.remove(&frame);
            }
        }
    }

    /// Frames with at least one retained key (tests/diagnostics).
    #[cfg(test)]
    pub fn resident_frames(&self) -> BTreeSet<u64> {
        self.frame_counts.keys().copied().collect()
    }

    /// Resident frames belonging to one composition signature (`prefix` without
    /// the trailing frame part). Used so the timeline lane only shows frames for
    /// the current despill/mask/background composition.
    pub fn resident_frames_with_prefix(&self, prefix: &str) -> BTreeSet<u64> {
        let needle = format!("{prefix}:");
        self.frame_by_key
            .iter()
            .filter(|(key, _)| key.starts_with(&needle))
            .map(|(_, frame)| *frame)
            .collect()
    }

    /// Drop every retained key outside one composition, freeing budget for the
    /// new composition after a despill/mask/background change.
    pub fn purge_except_prefix(&mut self, prefix: &str) -> usize {
        let needle = format!("{prefix}:");
        let stale = self
            .frame_by_key
            .keys()
            .filter(|key| !key.starts_with(&needle))
            .cloned()
            .collect::<Vec<_>>();
        for key in &stale {
            self.cache.remove(&key.clone());
            if let Some(frame) = self.frame_by_key.remove(key) {
                self.decrement_frame(frame);
            }
        }
        stale.len()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn bytes(&self) -> usize {
        self.cache.bytes()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.frame_by_key.clear();
        self.frame_counts.clear();
        self.evicted_frames.clear();
    }

    /// Apply a new budget; evicted keys update the resident frame set.
    pub fn retune(&mut self, byte_limit: usize, item_limit: usize) {
        let evicted = self.cache.retune(byte_limit, item_limit);
        for (key, _) in evicted {
            if let Some(frame) = self.frame_by_key.remove(&key) {
                self.decrement_frame(frame);
                self.evicted_frames.insert(frame);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preview(luma: u8) -> CachedPreview {
        CachedPreview {
            width: 2,
            height: 1,
            rgba: Arc::from([luma, 0, 0, 255, 0, 0, 0, 255]),
            path: None,
            background_applied: false,
            despill_applied: false,
            despill_unavailable: false,
            source_rgba: None,
        }
    }

    #[test]
    fn cached_previews_report_rgba_bytes() {
        assert_eq!(preview(1).memory_cost(), 8 + 256);
    }

    #[test]
    fn resident_frames_track_inserts_evictions_and_clear() {
        let mut cache = ResultRamCache::new(4096, 2);
        assert!(cache.insert("a".into(), 10, preview(1)));
        assert!(cache.insert("b".into(), 11, preview(2)));
        assert_eq!(cache.resident_frames(), BTreeSet::from([10, 11]));
        // Third item evicts the least recently used frame 10.
        assert!(cache.insert("c".into(), 12, preview(3)));
        assert_eq!(cache.resident_frames(), BTreeSet::from([11, 12]));
        assert!(cache.get("a").is_none());
        // Two keys for one frame collapse to a single resident frame.
        cache.clear();
        assert!(cache.insert("d".into(), 20, preview(1)));
        assert!(cache.insert("e".into(), 20, preview(2)));
        assert_eq!(cache.resident_frames(), BTreeSet::from([20]));
        cache.clear();
        assert!(cache.resident_frames().is_empty());
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn retune_shrinks_and_updates_residents() {
        let mut cache = ResultRamCache::new(4096, 4);
        for frame in 0..4 {
            assert!(cache.insert(format!("k{frame}"), frame, preview(frame as u8)));
        }
        assert_eq!(cache.len(), 4);
        cache.retune(600, 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.resident_frames().len(), 1);
        assert!(cache.bytes() <= 600);
    }

    #[test]
    fn composition_prefix_filters_and_purges_residents() {
        let mut cache = ResultRamCache::new(1 << 20, 8);
        cache.insert("sigA:10".into(), 10, preview(1));
        cache.insert("sigB:10".into(), 10, preview(2));
        cache.insert("sigB:11".into(), 11, preview(3));
        assert_eq!(
            cache.resident_frames_with_prefix("sigA"),
            BTreeSet::from([10])
        );
        assert_eq!(
            cache.resident_frames_with_prefix("sigB"),
            BTreeSet::from([10, 11])
        );
        assert_eq!(cache.purge_except_prefix("sigB"), 1);
        assert!(cache.resident_frames_with_prefix("sigA").is_empty());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn remove_key_updates_residents() {
        let mut cache = ResultRamCache::new(1 << 20, 8);
        cache.insert("sig:10".into(), 10, preview(1));
        assert!(cache.remove("sig:10"));
        assert!(cache.resident_frames_with_prefix("sig").is_empty());
        assert!(!cache.remove("sig:10"));
    }

    #[test]
    fn inserting_a_frame_replaces_its_older_mask_variant_only() {
        let mut cache = ResultRamCache::new(1 << 20, 8);
        cache.insert("sig:10:aaaa".into(), 10, preview(1));
        cache.insert("sig:11:aaaa".into(), 11, preview(2));
        cache.insert("other:10:aaaa".into(), 10, preview(3));
        cache.insert("sig:10:bbbb".into(), 10, preview(4));
        assert!(
            !cache.contains("sig:10:aaaa"),
            "older mask variant of frame 10 replaced"
        );
        assert!(cache.contains("sig:10:bbbb"));
        assert!(cache.contains("sig:11:aaaa"), "other frames untouched");
        assert!(
            cache.contains("other:10:aaaa"),
            "other compositions untouched"
        );
    }

    #[test]
    fn remove_frame_drops_every_composition_of_that_frame() {
        let mut cache = ResultRamCache::new(1 << 20, 8);
        cache.insert("sigA:10".into(), 10, preview(1));
        cache.insert("sigB:10".into(), 10, preview(2));
        cache.insert("sigA:11".into(), 11, preview(3));
        assert_eq!(cache.remove_frame(10), 2);
        assert_eq!(cache.resident_frames(), BTreeSet::from([11]));
        assert_eq!(cache.remove_frame(10), 0);
    }
}
