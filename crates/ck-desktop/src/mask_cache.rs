//! Bounded RAM cache for rasterised mask applications.
//!
//! Rasterising the garbage/despill stacks per preview request is wasted work
//! when the playhead and mask revision have not changed. The cache keeps the
//! most recent applications under the mask-raster budget (64 MiB / 256 items
//! by default) using the shared LRU `RamCache`.

use crate::mask_composite::MaskApplication;
use crate::ram_cache::RamCache;
use std::sync::Arc;

/// Identity of one rasterised application.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaskCacheKey {
    /// Session/window identity.
    pub session: u64,
    /// [`crate::mask_composite::mask_fingerprint`] of the masks at `frame`.
    pub fingerprint: u64,
    /// Absolute source frame.
    pub frame: i64,
}

pub struct MaskRasterCache {
    cache: RamCache<MaskCacheKey, Arc<MaskApplication>>,
}

impl MaskRasterCache {
    pub fn new(byte_limit: usize, item_limit: usize) -> Self {
        Self {
            cache: RamCache::new(byte_limit, item_limit),
        }
    }

    pub fn get(&mut self, key: &MaskCacheKey) -> Option<Arc<MaskApplication>> {
        self.cache.get_cloned(key)
    }

    pub fn insert(&mut self, key: MaskCacheKey, cost: usize, value: Arc<MaskApplication>) -> bool {
        self.cache.insert_with_cost(key, value, cost).retained
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn bytes(&self) -> usize {
        self.cache.bytes()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Apply a new RAM budget (evicts LRU entries when it shrinks).
    pub fn retune(&mut self, byte_limit: usize, item_limit: usize) {
        let _ = self.cache.retune(byte_limit, item_limit);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Approximate retained bytes of a mask application (rasters plus overhead).
pub fn application_cost(application: &MaskApplication) -> usize {
    let mut bytes = std::mem::size_of::<MaskApplication>();
    for (_, garbage) in &application.garbage {
        bytes += garbage.as_raw().len() + 128;
    }
    if let Some(weight) = &application.despill_weight {
        bytes += weight.as_raw().len() + 128;
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ck_desktop_core::MaskMode;
    use image::GrayImage;

    fn application(size: u32) -> MaskApplication {
        MaskApplication {
            garbage: vec![(MaskMode::Subtract, Arc::new(GrayImage::new(size, size)))],
            despill_weight: None,
        }
    }

    #[test]
    fn cache_hits_only_for_the_same_session_revision_and_frame() {
        let mut cache = MaskRasterCache::new(1 << 20, 8);
        let key = MaskCacheKey {
            session: 1,
            fingerprint: 2,
            frame: 10,
        };
        cache.insert(key, 1024, Arc::new(application(8)));
        assert!(cache.get(&key).is_some());
        assert!(
            cache
                .get(&MaskCacheKey {
                    fingerprint: 3,
                    ..key
                })
                .is_none()
        );
        assert!(cache.get(&MaskCacheKey { frame: 11, ..key }).is_none());
    }

    #[test]
    fn cache_evicts_least_recently_used_entries() {
        let mut cache = MaskRasterCache::new(4096, 2);
        let keys = (0..3)
            .map(|frame| MaskCacheKey {
                session: 1,
                fingerprint: 0,
                frame,
            })
            .collect::<Vec<_>>();
        for key in &keys {
            assert!(cache.insert(*key, 1024, Arc::new(application(4))));
        }
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&keys[0]).is_none());
        assert!(cache.get(&keys[1]).is_some());
        assert!(cache.get(&keys[2]).is_some());
    }

    #[test]
    fn oversize_applications_are_rejected_without_evicting() {
        let mut cache = MaskRasterCache::new(2048, 4);
        assert!(!cache.insert(
            MaskCacheKey {
                session: 1,
                fingerprint: 0,
                frame: 0
            },
            4096,
            Arc::new(application(4)),
        ));
        assert!(cache.is_empty());
    }

    #[test]
    fn application_cost_tracks_raster_bytes() {
        let small = application(4);
        let large = application(64);
        assert!(application_cost(&large) > application_cost(&small));
        assert!(application_cost(&small) >= 4 * 4);
    }
}
