//! Strict float32/C-order NPY interchange for development fixtures, with no Python parser.
use anyhow::{Context, Result, bail, ensure};
use ndarray::{ArrayD, IxDyn};
use std::{fs, path::Path};

fn field<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let start = header
        .find(&format!("'{key}'"))
        .or_else(|| header.find(&format!("\"{key}\"")))
        .with_context(|| format!("NPY header missing {key}"))?;
    Ok(header[start..]
        .split_once(':')
        .context("invalid NPY header field")?
        .1
        .trim_start())
}

pub fn read_npy(path: impl AsRef<Path>) -> Result<ArrayD<f32>> {
    let bytes = fs::read(path)?;
    ensure!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "invalid NPY magic/header"
    );
    let (offset, length): (usize, usize) = match (bytes[6], bytes[7]) {
        (1, 0) => (10, usize::from(u16::from_le_bytes([bytes[8], bytes[9]]))),
        (2 | 3, 0) if bytes.len() >= 12 => {
            (12, u32::from_le_bytes(bytes[8..12].try_into()?) as usize)
        }
        version => bail!("unsupported NPY version {version:?}"),
    };
    let end = offset.checked_add(length).context("NPY header overflow")?;
    ensure!(end <= bytes.len(), "truncated NPY header");
    let header = std::str::from_utf8(&bytes[offset..end])?;
    let dtype = field(header, "descr")?;
    let quote = dtype.chars().next().context("missing NPY dtype")?;
    ensure!(quote == '\'' || quote == '"', "invalid NPY dtype");
    let dtype = dtype[1..]
        .split(quote)
        .next()
        .context("invalid NPY dtype")?;
    ensure!(
        matches!(dtype, "<f4" | ">f4" | "=f4"),
        "NPY dtype must be float32, got {dtype}"
    );
    ensure!(
        field(header, "fortran_order")?.starts_with("False"),
        "Fortran-order NPY tensors are unsupported"
    );
    let shape = field(header, "shape")?;
    ensure!(shape.starts_with('('), "NPY shape must be a tuple");
    let shape: Vec<usize> = shape[1..]
        .split_once(')')
        .context("invalid NPY shape")?
        .0
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let elements = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .context("NPY shape overflow")?;
    let payload = elements.checked_mul(4).context("NPY byte count overflow")?;
    ensure!(bytes.len() - end == payload, "NPY payload size mismatch");
    let big_endian = dtype == ">f4" || (dtype == "=f4" && cfg!(target_endian = "big"));
    let values: Vec<f32> = bytes[end..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| {
            let data = *chunk;
            if big_endian {
                f32::from_be_bytes(data)
            } else {
                f32::from_le_bytes(data)
            }
        })
        .collect();
    Ok(ArrayD::from_shape_vec(IxDyn(&shape), values)?)
}

pub fn write_npy(path: impl AsRef<Path>, array: &ArrayD<f32>) -> Result<()> {
    let dimensions = array
        .shape()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let shape = if array.ndim() == 1 {
        format!("{dimensions},")
    } else {
        dimensions
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape}), }}");
    let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let length: u16 = header.len().try_into().context("NPY v1 header too long")?;
    let mut bytes = Vec::with_capacity(10 + header.len() + array.len() * 4);
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in array {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip_and_truncation() {
        let path = std::env::temp_dir().join(format!("ck-native-npy-{}.npy", std::process::id()));
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.0, -1.0, 0.125, 1000.0]).unwrap();
        write_npy(&path, &input).unwrap();
        assert_eq!(read_npy(&path).unwrap(), input);
        let mut bytes = fs::read(&path).unwrap();
        bytes.pop();
        fs::write(&path, bytes).unwrap();
        assert!(read_npy(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
