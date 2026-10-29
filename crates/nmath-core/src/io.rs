//! Local file I/O with atomic publication. This module implements NPY; other
//! formats in the runtime contract (CSV, JSON, Parquet, Arrow IPC, NPZ) are not
//! implemented yet.

use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

use crate::{
    buffer::Buffer,
    dtype::DType,
    error::{Error, ErrorCode, Result},
    memory::{MemoryBudget, Reservation},
    sync::CancellationToken,
};

const NPY_MAGIC: &[u8] = b"\x93NUMPY";
/// Upper bound for an NPY header; NumPy itself rejects headers above this size
/// by default.
const MAX_HEADER_LEN: usize = 10_000;

fn io_error(err: std::io::Error) -> Error {
    // Messages never include file paths, per the observability contract.
    Error::new(
        ErrorCode::IoFailed,
        format!("A file operation failed ({:?}).", err.kind()),
    )
}

fn format_error(detail: &str) -> Error {
    Error::new(
        ErrorCode::IoFormat,
        format!("The NPY data is invalid: {detail}."),
    )
}

fn descr(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::Bool => "|b1",
        DType::Int32 => "<i4",
        DType::Int64 => "<i8",
        DType::UInt32 => "<u4",
        DType::UInt64 => "<u8",
        DType::Float32 => "<f4",
        DType::Float64 => "<f8",
        other => {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                format!(
                    "NPY output does not support dtype '{}' in this build.",
                    other.id()
                ),
            ));
        }
    })
}

/// Encodes a C-order array as NPY version 1.0.
pub fn encode_npy(shape: &[usize], buffer: &Buffer) -> Result<Vec<u8>> {
    let shape_text = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr(buffer.dtype())?,
        shape_text
    );
    let unpadded = NPY_MAGIC.len() + 2 + 2 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(unpadded + 64 + buffer.byte_len() as usize);
    out.extend_from_slice(NPY_MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    match buffer {
        Buffer::Bool(v) => out.extend(v.iter().map(|&b| b as u8)),
        Buffer::I32(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::I64(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::U32(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::U64(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::F32(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::F64(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
    }
    Ok(out)
}

fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let needle = format!("'{key}':");
    let start = header
        .find(&needle)
        .ok_or_else(|| format_error(&format!("missing '{key}'")))?
        + needle.len();
    Ok(header[start..].trim_start())
}

/// Decodes NPY version 1.x/2.x/3.x data with little-endian or byte-sized
/// dtypes.
pub fn decode_npy(bytes: &[u8]) -> Result<(Vec<usize>, Buffer)> {
    decode_npy_with(bytes, |_| Ok(())).map(|(shape, buffer, ())| (shape, buffer))
}

/// Like [`decode_npy`], but charges the decoded buffer to `budget` after the
/// header is validated and before the buffer is allocated.
pub fn decode_npy_budgeted(
    bytes: &[u8],
    budget: &MemoryBudget,
) -> Result<(Vec<usize>, Buffer, Reservation)> {
    decode_npy_with(bytes, |len| budget.reserve(len))
}

fn decode_npy_with<R>(
    bytes: &[u8],
    reserve: impl FnOnce(u64) -> Result<R>,
) -> Result<(Vec<usize>, Buffer, R)> {
    if bytes.len() < 10 || &bytes[..6] != NPY_MAGIC {
        return Err(format_error("missing magic string"));
    }
    let (header_len, header_start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(format_error("truncated header length"));
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => return Err(format_error(&format!("unsupported version {v}"))),
    };
    if header_len > MAX_HEADER_LEN || header_start + header_len > bytes.len() {
        return Err(format_error("header length is out of range"));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|_| format_error("header is not valid text"))?;
    let descr_text = header_value(header, "descr")?;
    let descr_text = descr_text
        .strip_prefix('\'')
        .and_then(|s| s.split('\'').next())
        .ok_or_else(|| format_error("malformed descr"))?;
    let dtype = match descr_text {
        "|b1" => DType::Bool,
        "<i4" => DType::Int32,
        "<i8" => DType::Int64,
        "<u4" => DType::UInt32,
        "<u8" => DType::UInt64,
        "<f4" => DType::Float32,
        "<f8" => DType::Float64,
        other => {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                format!("NPY input with descr '{other}' is not supported in this build."),
            ));
        }
    };
    if !header_value(header, "fortran_order")?.starts_with("False") {
        return Err(format_error(
            "Fortran-order data is not supported in this build",
        ));
    }
    let shape_text = header_value(header, "shape")?;
    let shape_text = shape_text
        .strip_prefix('(')
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| format_error("malformed shape"))?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format_error("malformed shape"))
        })
        .collect::<Result<Vec<_>>>()?;
    let count = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format_error("shape overflows"))?;
    let data = &bytes[header_start + header_len..];
    let expected = count
        .checked_mul(dtype.item_size())
        .ok_or_else(|| format_error("shape overflows"))?;
    if data.len() != expected {
        return Err(format_error(&format!(
            "expected {expected} data bytes, found {}",
            data.len()
        )));
    }
    let reservation = reserve(expected as u64)?;
    macro_rules! read_le {
        ($t:ty, $n:expr) => {
            data.chunks_exact($n)
                .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
                .into()
        };
    }
    let buffer = match dtype {
        DType::Bool => {
            if data.iter().any(|&b| b > 1) {
                return Err(format_error("bool data contains values other than 0 and 1"));
            }
            Buffer::Bool(data.iter().map(|&b| b == 1).collect())
        }
        DType::Int32 => read_le!(i32, 4),
        DType::Int64 => read_le!(i64, 8),
        DType::UInt32 => read_le!(u32, 4),
        DType::UInt64 => read_le!(u64, 8),
        DType::Float32 => read_le!(f32, 4),
        DType::Float64 => read_le!(f64, 8),
        _ => unreachable!(),
    };
    Ok((shape, buffer, reservation))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteMode {
    /// Fail with `NMATH_IO_EXISTS` if the destination exists.
    CreateNew,
    /// Replace the destination atomically.
    Overwrite,
}

fn temp_path(dest: &Path) -> Result<PathBuf> {
    let parent = dest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = dest
        .file_name()
        .ok_or_else(|| Error::new(ErrorCode::IoFailed, "The output path has no file name."))?;
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok(parent.join(format!(
        ".{}.nmath-{}-{nonce}.tmp",
        name.to_string_lossy(),
        std::process::id()
    )))
}

/// Writes `bytes` to a temporary file next to `dest` and publishes it
/// atomically. Cancellation or failure removes the temporary file and preserves
/// any original.
pub fn write_atomic(
    dest: &Path,
    bytes: &[u8],
    mode: WriteMode,
    token: &CancellationToken,
) -> Result<()> {
    if mode == WriteMode::CreateNew && dest.exists() {
        return Err(Error::new(
            ErrorCode::IoExists,
            "The output file already exists.",
        ));
    }
    let tmp = temp_path(dest)?;
    let result = (|| {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(io_error)?;
        for chunk in bytes.chunks(1 << 20) {
            token.check()?;
            file.write_all(chunk).map_err(io_error)?;
        }
        file.sync_all().map_err(io_error)?;
        token.check()?;
        match mode {
            // A hard link fails if the destination exists, so publication never clobbers a
            // file created concurrently.
            WriteMode::CreateNew => fs::hard_link(&tmp, dest).map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    Error::new(ErrorCode::IoExists, "The output file already exists.")
                } else {
                    io_error(e)
                }
            }),
            WriteMode::Overwrite => fs::rename(&tmp, dest).map_err(io_error),
        }
    })();
    let _ = fs::remove_file(&tmp);
    result
}

/// Reads a whole file and fails with `NMATH_IO_MODIFIED` when its size or
/// modification time changes while reading. Detection is best effort.
pub fn read_checked(path: &Path, token: &CancellationToken) -> Result<Vec<u8>> {
    read_checked_with(path, token, |_| Ok(())).map(|(bytes, ())| bytes)
}

/// Like [`read_checked`], but charges the file contents to `budget` using the
/// size recorded at open, before any read buffer is allocated.
pub fn read_checked_budgeted(
    path: &Path,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<(Vec<u8>, Reservation)> {
    read_checked_with(path, token, |len| budget.reserve(len))
}

fn read_checked_with<R>(
    path: &Path,
    token: &CancellationToken,
    reserve: impl FnOnce(u64) -> Result<R>,
) -> Result<(Vec<u8>, R)> {
    let stamp = |m: &fs::Metadata| (m.len(), m.modified().ok());
    let file = File::open(path).map_err(io_error)?;
    let before = stamp(&file.metadata().map_err(io_error)?);
    let reservation = reserve(before.0)?;
    let mut bytes = Vec::with_capacity(before.0 as usize);
    // Read at most one byte past the recorded size so a growing file cannot
    // allocate beyond the reservation; the extra byte reports the change.
    let mut file = file.take(before.0.saturating_add(1));
    let mut chunk = vec![0u8; before.0.saturating_add(1).min(1 << 20) as usize];
    loop {
        token.check()?;
        let n = file.read(&mut chunk).map_err(io_error)?;
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..n]);
    }
    let after = stamp(&fs::metadata(path).map_err(io_error)?);
    if before != after || bytes.len() as u64 != before.0 {
        return Err(Error::new(
            ErrorCode::IoModified,
            "The input file changed while it was read.",
        ));
    }
    Ok((bytes, reservation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_round_trip() {
        let buffer = Buffer::I64(vec![1, -2, i64::MAX, 4, 5, 6]);
        let bytes = encode_npy(&[2, 3], &buffer).unwrap();
        assert_eq!((bytes.len() - buffer.byte_len() as usize) % 64, 0);
        assert_eq!(decode_npy(&bytes).unwrap(), (vec![2, 3], buffer));
        let scalar = encode_npy(&[], &Buffer::F32(vec![1.5])).unwrap();
        assert_eq!(
            decode_npy(&scalar).unwrap(),
            (vec![], Buffer::F32(vec![1.5]))
        );
    }

    #[test]
    fn budgeted_decode_charges_the_data_before_allocation() {
        let bytes = encode_npy(&[3], &Buffer::F64(vec![1.0, 2.0, 3.0])).unwrap();
        let tight = MemoryBudget::new(crate::memory::Device::Cpu, 23);
        assert_eq!(
            decode_npy_budgeted(&bytes, &tight).unwrap_err().code,
            ErrorCode::MemoryBudgetExceeded
        );
        let fits = MemoryBudget::new(crate::memory::Device::Cpu, 24);
        let (_, buffer, reservation) = decode_npy_budgeted(&bytes, &fits).unwrap();
        assert_eq!(buffer, Buffer::F64(vec![1.0, 2.0, 3.0]));
        assert_eq!(reservation.bytes(), 24);
    }

    #[test]
    fn rejects_malformed_npy() {
        assert_eq!(
            decode_npy(b"not npy").unwrap_err().code,
            ErrorCode::IoFormat
        );
        let mut bytes = encode_npy(&[3], &Buffer::F64(vec![1.0, 2.0, 3.0])).unwrap();
        bytes.pop();
        assert_eq!(decode_npy(&bytes).unwrap_err().code, ErrorCode::IoFormat);
        let mut huge = encode_npy(&[1], &Buffer::F64(vec![1.0])).unwrap();
        huge[8] = 0xff;
        huge[9] = 0xff;
        assert_eq!(decode_npy(&huge).unwrap_err().code, ErrorCode::IoFormat);
    }

    #[test]
    fn atomic_write_respects_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.npy");
        let token = CancellationToken::new();
        write_atomic(&dest, b"first", WriteMode::CreateNew, &token).unwrap();
        assert_eq!(
            write_atomic(&dest, b"second", WriteMode::CreateNew, &token)
                .unwrap_err()
                .code,
            ErrorCode::IoExists
        );
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(write_atomic(&dest, b"third", WriteMode::Overwrite, &cancelled).is_err());
        assert_eq!(fs::read(&dest).unwrap(), b"first");
        write_atomic(&dest, b"fourth", WriteMode::Overwrite, &token).unwrap();
        assert_eq!(read_checked(&dest, &token).unwrap(), b"fourth");
        let tight = MemoryBudget::new(crate::memory::Device::Cpu, 5);
        assert_eq!(
            read_checked_budgeted(&dest, &tight, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        let fits = MemoryBudget::new(crate::memory::Device::Cpu, 6);
        let (bytes, reservation) = read_checked_budgeted(&dest, &fits, &token).unwrap();
        assert_eq!((bytes.as_slice(), reservation.bytes()), (&b"fourth"[..], 6));
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            1,
            "no temporary files remain"
        );
    }
}
