//! BRP's TransNet postprocessing, including its inclusive scene endpoints.
use crate::{ContractError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scene {
    pub index: usize,
    pub start: usize,
    pub end: usize,
}

/// Return original-frame indices for a package window. Its middle 50 frames
/// become the next non-overlapping output block; context duplicates edge frames.
pub fn transnet_window_indices(frame_count: usize, output_start: usize) -> Result<Vec<usize>> {
    if frame_count == 0
        || output_start >= frame_count
        || !output_start.is_multiple_of(50)
        || output_start.checked_add(99).is_none()
    {
        return Err(ContractError("invalid TransNet window origin".into()));
    }
    Ok((0..100)
        .map(|index| {
            (output_start + index)
                .saturating_sub(25)
                .min(frame_count - 1)
        })
        .collect())
}

/// Active runs use >=, merge over at most one inactive frame, and cut after
/// the run. Runs touching either clip edge do not produce a boundary.
pub fn transition_cuts(probabilities: &[f32], threshold: f64) -> Result<Vec<usize>> {
    if !threshold.is_finite() || probabilities.iter().any(|p| !p.is_finite()) {
        return Err(ContractError(
            "nonfinite transition probability/threshold".into(),
        ));
    }
    // NumPy's float32-array/Python-float comparison casts the scalar to float32.
    let threshold = threshold as f32;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability >= threshold {
            start.get_or_insert(index);
        } else if let Some(first) = start.take() {
            runs.push((first, index - 1));
        }
    }
    if let Some(first) = start {
        runs.push((first, probabilities.len() - 1));
    }
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (first, last) in runs {
        if let Some(previous) = merged.last_mut() {
            if first - previous.1 <= 2 {
                previous.1 = last;
                continue;
            }
        }
        merged.push((first, last));
    }
    Ok(merged
        .into_iter()
        .filter(|&(first, last)| first > 0 && last + 1 < probabilities.len())
        .map(|(_, last)| last + 1)
        .collect())
}

/// Count both scene endpoints. The old app omitted +1 for the final scene,
/// incorrectly rejecting a tail whose length exactly equaled the minimum.
pub fn filter_cuts(cuts: &[usize], min_length: usize, last_frame: usize) -> Vec<usize> {
    let mut sorted = cuts.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let min_length = min_length.max(1);
    let mut result = Vec::new();
    for cut in sorted {
        if cut == 0 || cut > last_frame || cut < min_length || last_frame - cut + 1 < min_length {
            continue;
        }
        if result
            .last()
            .is_some_and(|previous| cut - previous < min_length)
        {
            continue;
        }
        result.push(cut);
    }
    result
}

pub fn merge_ranges(cuts: &[usize], last_frame: usize) -> Vec<Scene> {
    let mut cuts = cuts.to_vec();
    cuts.retain(|&cut| cut > 0 && cut <= last_frame);
    cuts.sort_unstable();
    cuts.dedup();
    let mut scenes = Vec::new();
    let mut start = 0;
    for cut in cuts {
        scenes.push(Scene {
            index: scenes.len(),
            start,
            end: cut - 1,
        });
        start = cut;
    }
    scenes.push(Scene {
        index: scenes.len(),
        start,
        end: last_frame,
    });
    scenes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_is_after_merged_transition_run_and_edges_are_ignored() {
        let probabilities = [0.8, 0.0, 0.0, 0.35, 0.0, 0.8, 0.0, 0.0, 0.8];
        assert_eq!(transition_cuts(&probabilities, 0.35).unwrap(), vec![6]);
        assert_eq!(
            transition_cuts(&[1.0; 20], 0.35).unwrap(),
            Vec::<usize>::new()
        );
        assert!(transition_cuts(&[f32::NAN], 0.35).is_err());
    }

    #[test]
    fn minimum_length_counts_both_inclusive_endpoints() {
        assert_eq!(filter_cuts(&[1, 6, 6, 10, 13, 14, 19], 6, 20), vec![6, 13]);
        assert_eq!(
            merge_ranges(&[6, 13], 20),
            vec![
                Scene {
                    index: 0,
                    start: 0,
                    end: 5
                },
                Scene {
                    index: 1,
                    start: 6,
                    end: 12
                },
                Scene {
                    index: 2,
                    start: 13,
                    end: 20
                }
            ]
        );
        assert_eq!(filter_cuts(&[99], 6, 104), vec![99]);
        assert_eq!(filter_cuts(&[99], 6, 103), Vec::<usize>::new());
        assert_eq!(filter_cuts(&[6], 6, 78), vec![6]);
        assert_eq!(filter_cuts(&[100], 1, 100), vec![100]);
        assert_eq!(
            merge_ranges(&[100], 100).last().unwrap(),
            &Scene {
                index: 1,
                start: 100,
                end: 100
            }
        );
    }

    #[test]
    fn windows_cover_partial_exact_and_multiblock_clips() {
        for count in [1, 49, 50, 51, 69, 100, 101] {
            let mut retained = Vec::new();
            for start in (0..count).step_by(50) {
                let indices = transnet_window_indices(count, start).unwrap();
                retained.extend_from_slice(&indices[25..25 + (count - start).min(50)]);
            }
            assert_eq!(retained, (0..count).collect::<Vec<_>>());
        }
        let first = transnet_window_indices(69, 0).unwrap();
        assert!(first[..25].iter().all(|&index| index == 0));
        let last = transnet_window_indices(69, 50).unwrap();
        assert!(last[44..].iter().all(|&index| index == 68));
    }
}
