//! Native orchestration primitives matching the pinned CorridorKey runtime.
//!
//! This crate has no interpreter or inference-runtime dependency. Model sessions
//! belong in the native host; its scalar/tensor contracts remain testable here.

pub mod color;
pub mod hints;
pub mod mask_layers;
pub mod rgb8_encoding;
pub mod sam_memory;
pub mod scenes;
pub mod spatial;
pub mod window_policy;
pub mod windows;

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError(pub String);

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ContractError {}

pub type Result<T> = std::result::Result<T, ContractError>;
