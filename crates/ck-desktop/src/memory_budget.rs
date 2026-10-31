//! v1-style global RAM cache budget.
//!
//! v1 (`brp_chroma/core/memory.py`) allows the frame, result and mask caches to
//! use a configurable percentage of physical memory (70 % by default, clamped to
//! 20–95 %). The native app mirrors that: the percentage is stored in a small
//! per-user config file, converted to byte budgets for the three cache families
//! and enforced by the bounded LRU caches themselves.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_PERCENT: u8 = 70;
pub const MIN_PERCENT: u8 = 20;
pub const MAX_PERCENT: u8 = 95;
/// Config file used by the desktop interface.
const CONFIG_FILE: &str = "cache-settings.json";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CacheSettings {
    /// Percentage of physical RAM the caches may use.
    pub percent: u8,
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            percent: DEFAULT_PERCENT,
        }
    }
}

impl CacheSettings {
    pub fn clamped(percent: u8) -> Self {
        Self {
            percent: percent.clamp(MIN_PERCENT, MAX_PERCENT),
        }
    }
}

/// Byte/item budgets for the three cache families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheLimits {
    pub source_bytes: usize,
    pub source_items: usize,
    pub result_bytes: usize,
    pub result_items: usize,
    pub mask_bytes: usize,
    pub mask_items: usize,
}

impl CacheLimits {
    #[cfg(test)]
    pub fn total_bytes(self) -> usize {
        self.source_bytes
            .saturating_add(self.result_bytes)
            .saturating_add(self.mask_bytes)
    }
}

/// Physical memory in bytes (Windows `GlobalMemoryStatusEx`; a conservative
/// 16 GiB fallback elsewhere).
pub fn total_ram_bytes() -> u64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        }
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status` is a valid, correctly-sized out-parameter.
        if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 && status.total_phys > 0 {
            status.total_phys
        } else {
            16 * 1024 * 1024 * 1024
        }
    }
    #[cfg(not(windows))]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let kb = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(16 * 1024 * 1024);
        kb.saturating_mul(1024)
    }
}

/// Percentage of physical RAM the caches may use.
pub fn budget_bytes(total_ram: u64, percent: u8) -> u64 {
    let percent = u64::from(percent.clamp(MIN_PERCENT, MAX_PERCENT));
    total_ram.saturating_mul(percent) / 100
}

/// Split the global budget between the v1-style cache families. The result
/// cache is the dominant consumer (auto-fill fills the timeline with it);
/// media frames keep a healthy window and masks only need a small share.
pub fn cache_limits(total_ram: u64, percent: u8) -> CacheLimits {
    let budget = budget_bytes(total_ram, percent) as usize;
    let source_bytes = budget * 30 / 100;
    let mask_bytes = budget * 15 / 100;
    let result_bytes = budget
        .saturating_sub(source_bytes)
        .saturating_sub(mask_bytes);
    CacheLimits {
        source_bytes,
        source_items: 4096,
        result_bytes,
        result_items: 4096,
        mask_bytes,
        mask_items: 1024,
    }
}

fn config_dir() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))
        .unwrap_or_else(std::env::temp_dir)
        .join("BRP-Chroma")
}

pub fn config_path() -> PathBuf {
    config_dir().join(CONFIG_FILE)
}

pub fn load() -> CacheSettings {
    std::fs::read_to_string(config_path())
        .ok()
        .and_then(|text| serde_json::from_str::<CacheSettings>(&text).ok())
        .map(|settings| CacheSettings::clamped(settings.percent))
        .unwrap_or_default()
}

pub fn save(settings: &CacheSettings) -> std::io::Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(&CacheSettings::clamped(settings.percent))
        .map_err(std::io::Error::other)?;
    std::fs::write(path, text)
}

/// Human-readable GiB value used by the settings UI.
pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn default_and_clamping_match_v1() {
        assert_eq!(CacheSettings::default().percent, 70);
        assert_eq!(CacheSettings::clamped(1).percent, 20);
        assert_eq!(CacheSettings::clamped(120).percent, 95);
    }

    #[test]
    fn budget_is_a_percentage_and_limits_fit_inside_it() {
        let total = 16 * GIB;
        assert_eq!(budget_bytes(total, 70), total * 70 / 100);
        let limits = cache_limits(total, 70);
        assert!(limits.total_bytes() <= budget_bytes(total, 70) as usize);
        assert!(limits.source_bytes > 0 && limits.result_bytes > 0 && limits.mask_bytes > 0);
        let small = cache_limits(4 * GIB, 20);
        assert!(small.total_bytes() <= budget_bytes(4 * GIB, 20) as usize);
    }

    #[test]
    fn physical_memory_is_plausible() {
        let total = total_ram_bytes();
        assert!(total >= 512 * 1024 * 1024, "detected RAM {total}");
    }

    #[test]
    fn persisted_settings_clamp_on_load() {
        let path = config_path();
        let previous = std::fs::read_to_string(&path).ok();
        assert!(save(&CacheSettings { percent: 200 }).is_ok());
        assert_eq!(load().percent, 95);
        if let Some(previous) = previous {
            let _ = std::fs::write(&path, previous);
        } else {
            let _ = std::fs::remove_file(&path);
        }
    }
}
