//! Subject-mask (SAM) layers: each layer has its own points and per-frame SAM results; the
//! combined mask applies the enabled layers in order (Keep adds, Remove subtracts).
use anyhow::{Context, Result, ensure};
use ck_desktop_core::{SamLayer, SamLayerMode};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// What the combined mask of one frame is made of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Composite {
    /// No enabled layer has a result on this frame.
    None,
    /// Exactly one Keep layer contributes: its own file is the combined mask.
    Single(PathBuf),
    /// Several layers (or a Remove layer) contribute, in application order.
    Combine(Vec<(PathBuf, SamLayerMode)>),
}

/// The contributing layer masks of one frame, in application order.
pub fn plan(layers: &[SamLayer], results: &BTreeMap<u64, PathBuf>) -> Composite {
    let inputs: Vec<(PathBuf, SamLayerMode)> = layers
        .iter()
        .filter(|layer| layer.enabled)
        .filter_map(|layer| {
            results
                .get(&layer.id)
                .map(|path| (path.clone(), layer.mode))
        })
        .collect();
    match inputs.as_slice() {
        [] => Composite::None,
        [(path, SamLayerMode::Keep)] => Composite::Single(path.clone()),
        _ => Composite::Combine(inputs),
    }
}

/// Combine 8-bit masks (> 127 is subject): Keep = union, Remove = subtract, in order.
pub fn combine(inputs: &[(PathBuf, SamLayerMode)]) -> Result<image::GrayImage> {
    let mut result: Option<image::GrayImage> = None;
    for (path, mode) in inputs {
        let mask = image::open(path)
            .with_context(|| format!("reading SAM layer mask {}", path.display()))?
            .into_luma8();
        let out = result.get_or_insert_with(|| image::GrayImage::new(mask.width(), mask.height()));
        ensure!(
            out.dimensions() == mask.dimensions(),
            "SAM layer masks have different sizes"
        );
        for (o, m) in out.pixels_mut().zip(mask.pixels()) {
            let on = m.0[0] > 127;
            match mode {
                SamLayerMode::Keep if on => o.0[0] = 255,
                SamLayerMode::Remove if on => o.0[0] = 0,
                _ => {}
            }
        }
    }
    result.context("no SAM layer masks to combine")
}

/// Write the combined mask of `inputs` to a fresh file in `directory` (unique name, so a file the
/// preview may still be reading is never overwritten).
pub fn write_combined(
    inputs: &[(PathBuf, SamLayerMode)],
    directory: &Path,
    frame: u64,
) -> Result<PathBuf> {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!(
        "{frame:06}-{}-{}.png",
        std::process::id(),
        NONCE.fetch_add(1, Ordering::Relaxed)
    ));
    combine(inputs)?.save(&path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(id: u64, mode: SamLayerMode, enabled: bool) -> SamLayer {
        SamLayer {
            id,
            name: format!("L{id}"),
            mode,
            enabled,
            prompt: String::new(),
            scene: None,
        }
    }

    #[test]
    fn plan_skips_disabled_and_missing_layers_and_keeps_order() {
        let layers = [
            layer(0, SamLayerMode::Keep, true),
            layer(3, SamLayerMode::Remove, true),
            layer(1, SamLayerMode::Keep, false),
        ];
        let mut results = BTreeMap::new();
        assert_eq!(plan(&layers, &results), Composite::None);
        results.insert(0, PathBuf::from("a.png"));
        results.insert(1, PathBuf::from("b.png"));
        assert_eq!(plan(&layers, &results), Composite::Single("a.png".into()));
        results.insert(3, PathBuf::from("c.png"));
        assert_eq!(
            plan(&layers, &results),
            Composite::Combine(vec![
                ("a.png".into(), SamLayerMode::Keep),
                ("c.png".into(), SamLayerMode::Remove)
            ])
        );
        // A lone Remove layer still combines (it removes from nothing: an empty mask).
        let only_remove = [layer(3, SamLayerMode::Remove, true)];
        assert!(matches!(
            plan(&only_remove, &results),
            Composite::Combine(_)
        ));
    }

    #[test]
    fn combine_applies_keep_and_remove_in_order() {
        let dir = std::env::temp_dir().join(format!("brp-sam-layers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, values: [u8; 4]| {
            let path = dir.join(name);
            image::GrayImage::from_raw(4, 1, values.to_vec())
                .unwrap()
                .save(&path)
                .unwrap();
            path
        };
        let a = write("a.png", [255, 255, 0, 0]);
        let b = write("b.png", [0, 255, 255, 0]);
        let c = write("c.png", [0, 0, 255, 255]);
        let keep_remove_keep = combine(&[
            (a.clone(), SamLayerMode::Keep),
            (b.clone(), SamLayerMode::Remove),
            (c.clone(), SamLayerMode::Keep),
        ])
        .unwrap();
        assert_eq!(keep_remove_keep.into_raw(), [255, 0, 255, 255]);
        // Order matters: the same layers with Keep last-but-one gives a different result.
        let keep_keep_remove = combine(&[
            (a.clone(), SamLayerMode::Keep),
            (c.clone(), SamLayerMode::Keep),
            (b.clone(), SamLayerMode::Remove),
        ])
        .unwrap();
        assert_eq!(keep_keep_remove.into_raw(), [255, 0, 0, 255]);
        let written =
            write_combined(&[(a, SamLayerMode::Keep), (b, SamLayerMode::Keep)], &dir, 7).unwrap();
        assert_eq!(
            image::open(&written).unwrap().into_luma8().into_raw(),
            [255, 255, 255, 0]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
