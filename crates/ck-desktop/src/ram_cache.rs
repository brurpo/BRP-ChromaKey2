//! Generic bounded RAM caches for retained desktop frames and raster masks.
//!
//! This module ports the bounded-cache part of the v1 `BRP_CHROMA` reference
//! (`brp_chroma/core/media.py`: the decoded-frame `FrameCache` plus
//! `MediaSession._trim_caches`): one least-recently-used store bounded both by
//! its own byte budget and by an item-count budget, keeping the newest entries
//! when a budget is exceeded. The desktop keeps the source-preview RGBA frames
//! in it today; the rendered result frames and SAM mask rasters are wired onto
//! the same type in follow-up changes.
//!
//! # Budget and eviction policy
//!
//! * [`RamCache::new`] bounds retained bytes and retained items. A zero budget
//!   disables retention completely: the cache stores nothing until it is
//!   retuned, which makes a zero budget a safe feature switch.
//! * [`RamCache::with_item_ceiling`] adds a per-item byte ceiling. Items larger
//!   than that ceiling, the byte budget or the item budget are rejected on
//!   insert without evicting retained entries, so one oversized frame cannot
//!   flush a cache. The default ceiling is the whole byte budget, which is the
//!   policy the previous private source-preview `FrameCache` used.
//! * [`RamCache::get`] and [`RamCache::get_cloned`] refresh recency, exactly
//!   like v1's `OrderedDict.move_to_end`.
//! * Eviction removes least-recently-used entries first. [`RamCache::insert`]
//!   returns evicted entries (oldest first) so the caller can release adjacent
//!   resources or account for the bytes; [`RamCache::retune`] and
//!   [`RamCache::trim_to_bytes`] shrink the cache before returning evictions.
//! * Inserting an existing key replaces the value and charges only the new
//!   cost, never both costs.
//!
//! Recency is a `HashMap` plus a stamped token queue, so `get`/`insert` are
//! amortized O(1): refreshing appends a token and eviction skips tokens whose
//! stamp no longer matches the live entry. The queue is rebuilt once it exceeds
//! a small multiple of the live entry count, so repeated cache hits cannot grow
//! it without bound.
//!
//! This file is the reusable library surface. Only the source-preview frame LRU
//! consumes it in the current change, so API used only by tests and by the
//! upcoming rendered/mask caches is allowed to be dead in the binary; every
//! documented behaviour is exercised by the unit tests at the bottom.

#![allow(dead_code)]

use image::{GrayImage, RgbaImage};
use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
    sync::Arc,
};

/// Rebuild the recency token queue once it holds this many tokens per live
/// entry (plus [`TOKEN_COMPACT_SLACK`]) so cache hits cannot grow it unbounded.
const TOKEN_COMPACT_FACTOR: usize = 2;
/// Fixed token allowance on top of [`TOKEN_COMPACT_FACTOR`] before a rebuild.
const TOKEN_COMPACT_SLACK: usize = 64;

/// A retained value that reports how many bytes of RAM it accounts for.
///
/// The reported cost is charged against the cache's byte budget. It need not
/// equal `size_of` or the allocation size; it is the number the caller wants to
/// budget (for example `width * height * 4` for an RGBA frame).
pub(crate) trait MemoryCost {
    fn memory_cost(&self) -> usize;
}

impl MemoryCost for [u8] {
    fn memory_cost(&self) -> usize {
        self.len()
    }
}

impl MemoryCost for Vec<u8> {
    fn memory_cost(&self) -> usize {
        self.len()
    }
}

impl<T: MemoryCost + ?Sized> MemoryCost for Arc<T> {
    fn memory_cost(&self) -> usize {
        (**self).memory_cost()
    }
}

impl MemoryCost for RgbaImage {
    fn memory_cost(&self) -> usize {
        self.as_raw().len()
    }
}

impl MemoryCost for GrayImage {
    fn memory_cost(&self) -> usize {
        self.as_raw().len()
    }
}

/// One retained value plus its recency stamp and charged cost.
///
/// Fields stay private so crate-visible access to [`RamCache::entries`] can
/// only observe the retained count, never forge or re-cost an entry.
pub(crate) struct CacheEntry<V> {
    /// Recency stamp of the newest live token for this entry.
    stamp: u64,
    value: V,
    /// Bytes charged against the byte budget while retained.
    cost: usize,
}

/// Everything an [`RamCache::insert`] displaced from the cache.
///
/// `retained` distinguishes a stored value from one rejected by the budget
/// policy. Replacing the previous value for the same key is reported through
/// `replaced` and is not part of `evicted`: only budget pressure evicts.
pub(crate) struct InsertOutcome<K, V> {
    /// `true` when the value was retained. A rejected insert changes nothing:
    /// `replaced` is `None` and `evicted` is empty (no eviction for a rejected
    /// item).
    pub(crate) retained: bool,
    /// Previous value held for the inserted key, if any.
    pub(crate) replaced: Option<V>,
    /// Entries evicted by the byte/item budgets, least-recently-used first.
    pub(crate) evicted: Vec<(K, V)>,
}

/// Least-recently-used store bounded by a byte budget and an item budget.
///
/// Keys are cloned into the recency queue, so `K` must support cheap cloning
/// (frame indices, `u64`, `PathBuf`, small key structs). Values are returned by
/// reference or clone; the cache itself never hands out ownership except when
/// an entry is evicted or removed.
pub(crate) struct RamCache<K, V> {
    /// Retained key -> entry map.
    ///
    /// Crate-visible only because the existing source-preview regression reads
    /// `cache.entries.len()` directly. Treat it as read-only: mutating it
    /// bypasses the byte and recency accounting, so all mutation goes through
    /// the methods on this type (use [`RamCache::len`] in new code).
    pub(crate) entries: HashMap<K, CacheEntry<V>>,
    /// Bytes retained across live entries.
    ///
    /// Crate-visible only for the same existing diagnostic reads
    /// (`cache.bytes`); use [`RamCache::bytes`] in new code.
    pub(crate) bytes: usize,
    /// Recency tokens ordered oldest first as `(stamp, key)`.
    recency: VecDeque<(u64, K)>,
    next_stamp: u64,
    byte_limit: usize,
    item_limit: usize,
    /// Optional explicit per-item byte ceiling; `None` follows the byte budget.
    /// The effective ceiling is always clamped to the byte budget.
    item_ceiling: Option<usize>,
}

impl<K: Eq + Hash + Clone, V> RamCache<K, V> {
    /// An empty cache bounded to `byte_limit` retained bytes and `item_limit`
    /// retained items. A zero byte or item budget disables retention.
    ///
    /// The per-item ceiling defaults to the whole byte budget (following later
    /// retunes): one item may fill the cache, but an item larger than the
    /// budget is rejected instead of evicting everything for it.
    pub(crate) fn new(byte_limit: usize, item_limit: usize) -> Self {
        Self::with_limits(byte_limit, item_limit, None)
    }

    /// An empty cache with an explicit per-item ceiling, clamped to the byte
    /// budget. Items above the ceiling are rejected by [`Self::insert`] even
    /// when the total budget would fit them.
    pub(crate) fn with_item_ceiling(
        byte_limit: usize,
        item_limit: usize,
        item_ceiling: usize,
    ) -> Self {
        Self::with_limits(byte_limit, item_limit, Some(item_ceiling))
    }

    fn with_limits(byte_limit: usize, item_limit: usize, item_ceiling: Option<usize>) -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
            recency: VecDeque::new(),
            next_stamp: 0,
            byte_limit,
            item_limit,
            item_ceiling,
        }
    }

    /// Bytes retained across all entries.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// Number of retained entries.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no entry is retained.
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current retained-byte budget.
    pub(crate) fn byte_limit(&self) -> usize {
        self.byte_limit
    }

    /// Current retained-item budget.
    pub(crate) fn item_limit(&self) -> usize {
        self.item_limit
    }

    /// Every retained key (used to report RAM-cache residents on the timeline).
    pub(crate) fn keys(&self) -> Vec<K>
    where
        K: Clone,
    {
        self.entries.keys().cloned().collect()
    }

    /// `true` when `key` is retained.
    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Look up `key`, refreshing its recency on a hit.
    pub(crate) fn get(&mut self, key: &K) -> Option<&V> {
        let stamp = self.next_stamp;
        let entry = self.entries.get_mut(key)?;
        // The value that is already the newest needs no new token; without this
        // check a single-entry cache would append a token on every lookup.
        if entry.stamp != stamp.wrapping_sub(1) {
            entry.stamp = stamp;
            self.next_stamp = stamp.wrapping_add(1);
            self.recency.push_back((stamp, key.clone()));
        }
        self.compact_recency_if_needed();
        self.entries.get(key).map(|entry| &entry.value)
    }

    /// Look up `key` without refreshing its recency.
    pub(crate) fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|entry| &entry.value)
    }

    /// Look up `key` and clone the retained value, refreshing its recency.
    pub(crate) fn get_cloned(&mut self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.get(key).cloned()
    }

    /// Insert a value that reports its own RAM cost through [`MemoryCost`].
    pub(crate) fn insert(&mut self, key: K, value: V) -> InsertOutcome<K, V>
    where
        V: MemoryCost,
    {
        let cost = value.memory_cost();
        self.insert_with_cost(key, value, cost)
    }

    /// Insert `value` charged as `cost` bytes against the byte budget.
    ///
    /// Least-recently-used entries are evicted until both budgets can hold the
    /// value. The insert is rejected without any state change (and therefore
    /// without evicting anything) when either budget is zero or the item
    /// exceeds the per-item ceiling.
    pub(crate) fn insert_with_cost(
        &mut self,
        key: K,
        value: V,
        cost: usize,
    ) -> InsertOutcome<K, V> {
        if !self.accepts(cost) {
            return InsertOutcome {
                retained: false,
                replaced: None,
                evicted: Vec::new(),
            };
        }
        let replaced = self.entries.remove(&key).map(|entry| {
            self.bytes = self.bytes.saturating_sub(entry.cost);
            entry.value
        });
        let mut evicted = Vec::new();
        while self.bytes.saturating_add(cost) > self.byte_limit
            || self.entries.len().saturating_add(1) > self.item_limit
        {
            match self.evict_oldest() {
                Some(entry) => evicted.push(entry),
                None => break,
            }
        }
        let stamp = self.next_stamp;
        self.next_stamp = stamp.wrapping_add(1);
        self.bytes = self.bytes.saturating_add(cost);
        self.entries
            .insert(key.clone(), CacheEntry { stamp, value, cost });
        self.recency.push_back((stamp, key));
        InsertOutcome {
            retained: true,
            replaced,
            evicted,
        }
    }

    /// Remove `key`, returning its value. Recency tokens for the key become
    /// stale and are skipped by the next eviction.
    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        let entry = self.entries.remove(key)?;
        self.bytes = self.bytes.saturating_sub(entry.cost);
        Some(entry.value)
    }

    /// Drop every retained entry.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.recency.clear();
        self.bytes = 0;
    }

    /// Replace both budgets, then evict least-recently-used entries until the
    /// cache is inside them again. Returns the evicted entries, oldest first.
    ///
    /// This is how a global memory budget shrinks a cache below its current
    /// size. A zero budget empties the cache and disables retention until it is
    /// retuned to positive values.
    pub(crate) fn retune(&mut self, byte_limit: usize, item_limit: usize) -> Vec<(K, V)> {
        self.byte_limit = byte_limit;
        self.item_limit = item_limit;
        if self.byte_limit == 0 || self.item_limit == 0 {
            return self.evict_all();
        }
        self.evict_while(|cache| {
            cache.bytes > cache.byte_limit || cache.entries.len() > cache.item_limit
        })
    }

    /// Drop least-recently-used entries until at most `target_bytes` remain.
    ///
    /// This is v1's `FrameCache.trim_to` and is how a shared memory budget
    /// trims several caches at once without changing their long-term budgets.
    /// `target_bytes == 0` empties the cache.
    pub(crate) fn trim_to_bytes(&mut self, target_bytes: usize) -> Vec<(K, V)> {
        if target_bytes == 0 {
            return self.evict_all();
        }
        self.evict_while(|cache| cache.bytes > target_bytes)
    }

    fn accepts(&self, cost: usize) -> bool {
        self.byte_limit > 0 && self.item_limit > 0 && cost <= self.item_ceiling()
    }

    fn item_ceiling(&self) -> usize {
        self.item_ceiling
            .unwrap_or(self.byte_limit)
            .min(self.byte_limit)
    }

    /// Evict least-recently-used entries while `over_budget` reports true.
    fn evict_while(&mut self, mut over_budget: impl FnMut(&Self) -> bool) -> Vec<(K, V)> {
        let mut evicted = Vec::new();
        while over_budget(self) {
            match self.evict_oldest() {
                Some(entry) => evicted.push(entry),
                None => break,
            }
        }
        evicted
    }

    fn evict_all(&mut self) -> Vec<(K, V)> {
        let mut evicted = Vec::new();
        while let Some(entry) = self.evict_oldest() {
            evicted.push(entry);
        }
        evicted
    }

    /// Pop the least-recently-used live entry, skipping stale recency tokens.
    fn evict_oldest(&mut self) -> Option<(K, V)> {
        while let Some((stamp, key)) = self.recency.pop_front() {
            let live = self
                .entries
                .get(&key)
                .is_some_and(|entry| entry.stamp == stamp);
            if !live {
                continue;
            }
            let entry = self.entries.remove(&key)?;
            self.bytes = self.bytes.saturating_sub(entry.cost);
            return Some((key, entry.value));
        }
        None
    }

    /// Rebuild the token queue once lazy refresh tokens have outgrown it.
    fn compact_recency_if_needed(&mut self) {
        let max_tokens = self
            .entries
            .len()
            .saturating_mul(TOKEN_COMPACT_FACTOR)
            .saturating_add(TOKEN_COMPACT_SLACK);
        if self.recency.len() <= max_tokens {
            return;
        }
        let mut live = self
            .entries
            .iter()
            .map(|(key, entry)| (entry.stamp, key.clone()))
            .collect::<Vec<_>>();
        live.sort_unstable_by_key(|(stamp, _)| *stamp);
        self.recency = live.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(bytes: &[u8]) -> Arc<[u8]> {
        Arc::from(bytes.to_vec())
    }

    fn cache_of(byte_limit: usize, item_limit: usize) -> RamCache<u32, Arc<[u8]>> {
        RamCache::new(byte_limit, item_limit)
    }

    fn keys(entries: &[(u32, Arc<[u8]>)]) -> Vec<u32> {
        entries.iter().map(|(key, _)| *key).collect()
    }

    #[test]
    fn get_refreshes_lru_recency_before_budget_eviction() {
        let mut cache = cache_of(64, 3);
        for key in 0..3 {
            assert!(cache.insert(key, frame(&[key as u8])).retained);
        }
        assert!(cache.get(&0).is_some());
        let evicted = cache.insert(3, frame(&[3])).evicted;
        assert_eq!(keys(&evicted), vec![1]);
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&0) && cache.contains_key(&2) && cache.contains_key(&3));
        // Refreshing 0 and evicting 1 leaves the recency order 2, 0, 3.
        let evicted = cache.insert(4, frame(&[4])).evicted;
        assert_eq!(keys(&evicted), vec![2]);
    }

    #[test]
    fn byte_budget_evicts_oldest_and_returns_evicted_entries() {
        let mut cache = cache_of(10, 10);
        assert!(cache.insert(0, frame(&[0; 4])).retained);
        assert!(cache.insert(1, frame(&[1; 4])).retained);
        assert_eq!(cache.bytes(), 8);
        let outcome = cache.insert(2, frame(&[2; 4]));
        assert!(outcome.retained);
        assert_eq!(keys(&outcome.evicted), vec![0]);
        assert_eq!(cache.bytes(), 8);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&0).is_none());
        // An item that exactly fills the budget is accepted, evicting the rest.
        let outcome = cache.insert(3, frame(&[3; 10]));
        assert!(outcome.retained);
        assert_eq!(keys(&outcome.evicted), vec![1, 2]);
        assert_eq!(cache.bytes(), 10);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&3).is_some());
    }

    #[test]
    fn item_budget_evicts_oldest_and_replacement_is_not_double_counted() {
        let mut cache = cache_of(1024, 2);
        assert!(cache.insert(0, frame(&[0])).retained);
        assert!(cache.insert(1, frame(&[1])).retained);
        let outcome = cache.insert(2, frame(&[2]));
        assert_eq!(keys(&outcome.evicted), vec![0]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.bytes(), 2);
        // Replacing a key keeps one item slot and charges only the new cost.
        let outcome = cache.insert(1, frame(&[9, 9]));
        assert!(outcome.retained);
        assert!(outcome.evicted.is_empty());
        assert_eq!(outcome.replaced.as_deref(), Some(&[1u8][..]));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.bytes(), 3);
        assert_eq!(
            cache.peek(&1).map(|value| value.as_ref()),
            Some(&[9u8, 9][..])
        );
        assert_eq!(cache.remove(&1).as_deref(), Some(&[9u8, 9][..]));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), 1);
    }

    #[test]
    fn items_larger_than_the_budget_policy_are_rejected_without_evicting() {
        // The default ceiling lets one item fill the byte budget.
        let mut cache = cache_of(10, 4);
        assert!(cache.insert(0, frame(&[0; 4])).retained);
        let outcome = cache.insert(1, frame(&[1; 11]));
        assert!(!outcome.retained);
        assert!(outcome.replaced.is_none());
        assert!(outcome.evicted.is_empty());
        assert_eq!(cache.bytes(), 4);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&0).is_some());
        // An explicit per-item ceiling below the total budget.
        let mut cache = RamCache::with_item_ceiling(64, 4, 8);
        assert!(cache.insert(0, frame(&[0; 8])).retained);
        assert!(!cache.insert(1, frame(&[1; 9])).retained);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&0).is_some());
        // The ceiling is clamped to the byte budget.
        let mut cache = RamCache::with_item_ceiling(8, 4, 64);
        assert!(cache.insert(0, frame(&[0; 8])).retained);
        assert!(!cache.insert(1, frame(&[1; 9])).retained);
        // The default ceiling follows a retuned byte budget; an explicit one
        // stays clamped to the budget instead of growing with it.
        let mut cache = RamCache::new(8, 4);
        assert!(cache.insert(0, frame(&[0; 8])).retained);
        cache.retune(16, 4);
        assert!(cache.insert(1, frame(&[0; 16])).retained);
        let mut cache = RamCache::with_item_ceiling(8, 4, 8);
        cache.retune(16, 4);
        assert!(!cache.insert(0, frame(&[0; 9])).retained);
        assert!(cache.insert(0, frame(&[0; 8])).retained);
    }

    #[test]
    fn shrinking_budgets_below_the_current_size_evicts_until_inside_them() {
        let mut cache = cache_of(64, 10);
        for key in 0..4 {
            assert!(cache.insert(key, frame(&[0; 8])).retained);
        }
        assert!(cache.get(&0).is_some());
        assert_eq!(cache.bytes(), 32);
        // Refreshing 0 leaves the recency order 1, 2, 3, 0.
        let evicted = cache.retune(16, 10);
        assert_eq!(keys(&evicted), vec![1, 2]);
        assert_eq!(cache.byte_limit(), 16);
        assert_eq!(cache.bytes(), 16);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&0) && cache.contains_key(&3));
        assert!(!cache.contains_key(&1) && !cache.contains_key(&2));
        let evicted = cache.trim_to_bytes(8);
        assert_eq!(keys(&evicted), vec![3]);
        assert_eq!(cache.bytes(), 8);
        assert_eq!(cache.len(), 1);
        // Shrinking the item budget to the current count keeps everything.
        let evicted = cache.retune(1024, 1);
        assert!(evicted.is_empty());
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&0).is_some());
        let evicted = cache.trim_to_bytes(0);
        assert_eq!(evicted.len(), 1);
        assert!(cache.is_empty());
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn zero_budget_disables_retention_until_retuned() {
        // A zero byte budget rejects even zero-cost items: the cache is off.
        let mut cache = cache_of(0, 4);
        let outcome = cache.insert(0, frame(&[]));
        assert!(!outcome.retained);
        assert!(cache.is_empty());
        assert_eq!(cache.bytes(), 0);
        // A zero item budget behaves the same way.
        let mut cache = cache_of(64, 0);
        assert!(!cache.insert(0, frame(&[0])).retained);
        assert!(cache.is_empty());
        // Positive budgets retain zero-cost items while the item budget allows.
        let mut cache = cache_of(64, 2);
        assert!(cache.insert(0, frame(&[])).retained);
        assert!(cache.insert(1, frame(&[])).retained);
        let outcome = cache.insert(2, frame(&[]));
        assert_eq!(keys(&outcome.evicted), vec![0]);
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.len(), 2);
        // Retuning to zero evicts everything and disables retention again.
        let evicted = cache.retune(0, 2);
        assert_eq!(evicted.len(), 2);
        assert!(cache.is_empty());
        assert!(!cache.insert(3, frame(&[0])).retained);
    }

    #[test]
    fn repeated_lookups_keep_the_recency_queue_bounded_in_order() {
        let mut cache = cache_of(1024, 8);
        for key in 0..8 {
            assert!(cache.insert(key, frame(&[0; 4])).retained);
        }
        for _ in 0..512 {
            for key in [0, 1, 2, 3, 4, 5, 6, 7] {
                assert!(cache.get(&key).is_some());
            }
        }
        let bounded = cache.entries.len() * TOKEN_COMPACT_FACTOR + TOKEN_COMPACT_SLACK;
        assert!(
            cache.recency.len() <= bounded,
            "recency token queue grew to {} tokens for {} entries",
            cache.recency.len(),
            cache.entries.len()
        );
        // The last cycle refreshed 0..=7 in order, so 0 is least recent.
        let evicted = cache.insert(8, frame(&[0; 4])).evicted;
        assert_eq!(keys(&evicted), vec![0]);
    }

    #[test]
    fn explicit_cost_and_memory_cost_impls_charge_the_documented_footprint() {
        // Values without a MemoryCost impl can be charged an explicit cost.
        let mut cache: RamCache<u32, u64> = RamCache::new(10, 4);
        assert!(cache.insert_with_cost(0, 7, 6).retained);
        assert!(cache.insert_with_cost(1, 8, 6).retained);
        let outcome = cache.insert_with_cost(2, 9, 6);
        assert!(outcome.retained);
        assert_eq!(outcome.evicted.len(), 1);
        assert_eq!(cache.bytes(), 6);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.peek(&2), Some(&9));
        assert_eq!(frame(&[0; 5]).memory_cost(), 5);
        assert_eq!(vec![0u8; 7].memory_cost(), 7);
        assert_eq!(RgbaImage::new(3, 2).memory_cost(), 24);
        assert_eq!(GrayImage::new(3, 2).memory_cost(), 6);
    }
}
