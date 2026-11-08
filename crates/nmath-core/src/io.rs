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

fn descr(dtype: DType) -> Result<String> {
    Ok(match dtype {
        DType::Bool => "|b1",
        DType::Int8 => "|i1",
        DType::Int16 => "<i2",
        DType::Int32 => "<i4",
        DType::Int64 => "<i8",
        DType::UInt8 => "|u1",
        DType::UInt16 => "<u2",
        DType::UInt32 => "<u4",
        DType::UInt64 => "<u8",
        DType::Float16 => "<f2",
        DType::Float32 => "<f4",
        DType::Float64 => "<f8",
        DType::Complex64 => "<c8",
        DType::Complex128 => "<c16",
        DType::Str(n) => return Ok(format!("<U{n}")),
        DType::Datetime64(u) => return Ok(format!("<M8[{}]", u.id())),
        DType::Timedelta64(u) => return Ok(format!("<m8[{}]", u.id())),
    }
    .into())
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
        Buffer::Unicode { values, .. } => values
            .iter()
            .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        Buffer::Temporal { ticks, .. } => ticks
            .iter()
            .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        Buffer::Bool(v) => out.extend(v.iter().map(|&b| b as u8)),
        Buffer::I8(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::I16(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::I32(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::I64(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::U8(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::U16(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::U32(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::U64(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::F16(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::F32(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Buffer::C64(v) => v.iter().for_each(|z| {
            out.extend_from_slice(&z[0].to_le_bytes());
            out.extend_from_slice(&z[1].to_le_bytes());
        }),
        Buffer::C128(v) => v.iter().for_each(|z| {
            out.extend_from_slice(&z[0].to_le_bytes());
            out.extend_from_slice(&z[1].to_le_bytes());
        }),
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

/// Decodes NPY version 1.x/2.x/3.x data. Complex dtypes also accept
/// big-endian components and Fortran-order storage.
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
        "|i1" => DType::Int8,
        "<i2" => DType::Int16,
        "<i4" => DType::Int32,
        "<i8" => DType::Int64,
        "|u1" => DType::UInt8,
        "<u2" => DType::UInt16,
        "<u4" => DType::UInt32,
        "<u8" => DType::UInt64,
        "<f2" | ">f2" => DType::Float16,
        "<f4" => DType::Float32,
        "<f8" => DType::Float64,
        "<c8" | ">c8" => DType::Complex64,
        "<c16" | ">c16" => DType::Complex128,
        other if other.starts_with("<U") || other.starts_with(">U") => {
            DType::from_id(&format!("str[{}]", &other[2..]))?
        }
        other if other.starts_with("<M8[") || other.starts_with(">M8[") => {
            DType::from_id(&format!("datetime64{}", &other[3..]))?
        }
        other if other.starts_with("<m8[") || other.starts_with(">m8[") => {
            DType::from_id(&format!("timedelta64{}", &other[3..]))?
        }
        _ => return Err(format_error("unsupported dtype descriptor")),
    };
    let fortran = header_value(header, "fortran_order")?.starts_with("True");
    if (!fortran && !header_value(header, "fortran_order")?.starts_with("False"))
        || (fortran
            && !matches!(
                dtype,
                DType::Float16
                    | DType::Complex64
                    | DType::Complex128
                    | DType::Str(_)
                    | DType::Datetime64(_)
                    | DType::Timedelta64(_)
            ))
    {
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
    let complex_chunk = |i: usize| {
        let offset = if fortran {
            let (mut c_index, mut stride, mut f_index) = (i, count, 0);
            for &dim in shape.iter().rev() {
                stride /= dim;
                f_index += (c_index % dim) * stride;
                c_index /= dim;
            }
            f_index
        } else {
            i
        };
        &data[offset * dtype.item_size()..(offset + 1) * dtype.item_size()]
    };
    let buffer = match dtype {
        DType::Str(width) => {
            let mut values = Vec::with_capacity(count * width);
            for i in 0..count {
                for c in complex_chunk(i).chunks_exact(4) {
                    let c = if descr_text.starts_with('>') {
                        u32::from_be_bytes(c.try_into().unwrap())
                    } else {
                        u32::from_le_bytes(c.try_into().unwrap())
                    };
                    if char::from_u32(c).is_none() {
                        return Err(format_error("invalid Unicode scalar"));
                    }
                    values.push(c);
                }
            }
            Buffer::Unicode { width, values }
        }
        DType::Datetime64(_) | DType::Timedelta64(_) => Buffer::Temporal {
            dtype,
            ticks: (0..count)
                .map(|i| {
                    let c = complex_chunk(i).try_into().unwrap();
                    if descr_text.starts_with('>') {
                        i64::from_be_bytes(c)
                    } else {
                        i64::from_le_bytes(c)
                    }
                })
                .collect(),
        },
        DType::Bool => {
            if data.iter().any(|&b| b > 1) {
                return Err(format_error("bool data contains values other than 0 and 1"));
            }
            Buffer::Bool(data.iter().map(|&b| b == 1).collect())
        }
        DType::Int8 => read_le!(i8, 1),
        DType::Int16 => read_le!(i16, 2),
        DType::Int32 => read_le!(i32, 4),
        DType::Int64 => read_le!(i64, 8),
        DType::UInt8 => read_le!(u8, 1),
        DType::UInt16 => read_le!(u16, 2),
        DType::UInt32 => read_le!(u32, 4),
        DType::UInt64 => read_le!(u64, 8),
        DType::Float16 => Buffer::F16(
            (0..count)
                .map(|i| {
                    let bytes = complex_chunk(i).try_into().unwrap();
                    if descr_text.starts_with('>') {
                        crate::buffer::Float16::from_be_bytes(bytes)
                    } else {
                        crate::buffer::Float16::from_le_bytes(bytes)
                    }
                })
                .collect(),
        ),
        DType::Float32 => read_le!(f32, 4),
        DType::Float64 => read_le!(f64, 8),
        DType::Complex64 => Buffer::C64(
            (0..count)
                .map(|i| {
                    let c = complex_chunk(i);
                    let read = if descr_text.starts_with('>') {
                        f32::from_be_bytes
                    } else {
                        f32::from_le_bytes
                    };
                    [
                        read(c[..4].try_into().unwrap()),
                        read(c[4..].try_into().unwrap()),
                    ]
                })
                .collect(),
        ),
        DType::Complex128 => Buffer::C128(
            (0..count)
                .map(|i| {
                    let c = complex_chunk(i);
                    let read = if descr_text.starts_with('>') {
                        f64::from_be_bytes
                    } else {
                        f64::from_le_bytes
                    };
                    [
                        read(c[..8].try_into().unwrap()),
                        read(c[8..].try_into().unwrap()),
                    ]
                })
                .collect(),
        ),
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
    let mut file = File::open(path).map_err(io_error)?;
    let before = metadata_stamp(&file.metadata().map_err(io_error)?);
    read_opened(path, &mut file, before, token, reserve)
}

fn metadata_stamp(metadata: &fs::Metadata) -> (u64, Option<SystemTime>) {
    (metadata.len(), metadata.modified().ok())
}

fn read_opened<R>(
    path: &Path,
    file: &mut File,
    before: (u64, Option<SystemTime>),
    token: &CancellationToken,
    reserve: impl FnOnce(u64) -> Result<R>,
) -> Result<(Vec<u8>, R)> {
    let reservation = reserve(before.0)?;
    let size = usize::try_from(before.0)
        .map_err(|_| Error::new(ErrorCode::IoFailed, "The input file is too large to read."))?;
    let mut bytes = vec![0u8; size];
    read_fixed(file, &mut bytes, token)?;
    let after = metadata_stamp(&fs::metadata(path).map_err(io_error)?);
    if before != after {
        return Err(io_modified());
    }
    Ok((bytes, reservation))
}

fn io_modified() -> Error {
    Error::new(
        ErrorCode::IoModified,
        "The input file changed while it was read.",
    )
}

fn read_fixed(reader: &mut impl Read, bytes: &mut [u8], token: &CancellationToken) -> Result<()> {
    for chunk in bytes.chunks_mut(1 << 20) {
        let mut remaining = chunk;
        while !remaining.is_empty() {
            token.check()?;
            let n = reader.read(remaining).map_err(io_error)?;
            if n == 0 {
                return Err(io_modified());
            }
            remaining = &mut remaining[n..];
        }
    }
    token.check()?;
    // The probe stays on the stack, so appended input cannot grow the charged
    // buffer.
    let mut extra = [0u8; 1];
    if reader.read(&mut extra).map_err(io_error)? != 0 {
        return Err(io_modified());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Cursor, Seek, SeekFrom},
        time::UNIX_EPOCH,
    };

    use super::*;

    #[test]
    fn fixed_reader_rejects_growth_and_truncation_without_resizing() {
        let token = CancellationToken::new();
        for input in [&b"abc"[..], &b"a"[..]] {
            let mut bytes = vec![0u8; 2];
            let capacity = bytes.capacity();
            let err = read_fixed(&mut Cursor::new(input), &mut bytes, &token).unwrap_err();
            assert_eq!(err.code, ErrorCode::IoModified);
            assert_eq!(bytes.capacity(), capacity);
        }
        let mut bytes = vec![0u8; 2];
        read_fixed(&mut Cursor::new(b"ab"), &mut bytes, &token).unwrap();
        assert_eq!(bytes, b"ab");
    }

    #[test]
    fn fixed_reader_cancellation_releases_the_input_charge() {
        struct CancellingReader<'a> {
            token: &'a CancellationToken,
            reads: usize,
        }
        impl Read for CancellingReader<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                self.reads += 1;
                if self.reads == 1 {
                    self.token.cancel();
                }
                out[0] = 1;
                Ok(1)
            }
        }
        let budget = MemoryBudget::new(crate::memory::Device::Cpu, 2 << 20);
        let token = CancellationToken::new();
        let result = (|| {
            let _charge = budget.reserve(2 << 20)?;
            let mut bytes = vec![0u8; 2 << 20];
            read_fixed(
                &mut CancellingReader {
                    token: &token,
                    reads: 0,
                },
                &mut bytes,
                &token,
            )
        })();
        assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
        assert_eq!(budget.usage().used, 0);
    }

    #[test]
    fn checked_reader_detects_file_changes_after_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.npy");
        let token = CancellationToken::new();
        for change in 0..3 {
            fs::write(&path, b"ab").unwrap();
            let mut file = File::options().read(true).write(true).open(&path).unwrap();
            let before = metadata_stamp(&file.metadata().unwrap());
            match change {
                0 => {
                    file.seek(SeekFrom::End(0)).unwrap();
                    file.write_all(b"c").unwrap();
                }
                1 => {
                    file.set_len(1).unwrap();
                }
                _ => {
                    file.set_times(fs::FileTimes::new().set_modified(UNIX_EPOCH))
                        .unwrap();
                }
            }
            file.rewind().unwrap();
            let budget = MemoryBudget::new(crate::memory::Device::Cpu, 2);
            let err = read_opened(&path, &mut file, before, &token, |len| budget.reserve(len))
                .unwrap_err();
            assert_eq!(err.code, ErrorCode::IoModified);
            assert_eq!(budget.usage().used, 0);
        }
    }

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
