//! Local file I/O with atomic publication. This module implements NPY; other
//! formats in the runtime contract (CSV, JSON, Parquet, Arrow IPC, NPZ) are not
//! implemented yet.

use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::SystemTime,
};

use crate::{
    buffer::Buffer,
    dtype::{DType, TimeBase, TimeUnit},
    error::{Error, ErrorCode, Result},
    memory::{MemoryBudget, Reservation},
    sync::CancellationToken,
};

const NPY_MAGIC: &[u8] = b"\x93NUMPY";
/// Upper bound for an NPY header; NumPy itself rejects headers above this size
/// by default.
const MAX_HEADER_LEN: usize = 10_000;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

fn decimal_len(value: usize) -> usize {
    if value == 0 {
        1
    } else {
        value.ilog10() as usize + 1
    }
}

fn time_unit_len(unit: TimeUnit) -> usize {
    let base_len = match unit.base {
        TimeBase::Year
        | TimeBase::Month
        | TimeBase::Week
        | TimeBase::Day
        | TimeBase::Hour
        | TimeBase::Minute
        | TimeBase::Second => 1,
        TimeBase::Milli
        | TimeBase::Micro
        | TimeBase::Nano
        | TimeBase::Pico
        | TimeBase::Femto
        | TimeBase::Atto => 2,
    };
    base_len
        + if unit.step == 1 {
            0
        } else {
            decimal_len(unit.step as usize)
        }
}

fn descr_len(dtype: DType) -> usize {
    match dtype {
        DType::Complex128 => 4,
        DType::Str(width) => 2 + decimal_len(width),
        DType::Datetime64(unit) | DType::Timedelta64(unit) => 5 + time_unit_len(unit),
        _ => 3,
    }
}

/// Returns the padded version-1 header length before allocating the header.
pub(crate) fn checked_npy_header_len(shape: &[usize], dtype: DType) -> Result<u16> {
    let shape_len = match shape {
        [] => 2,
        [dimension] => 3 + decimal_len(*dimension),
        dimensions => {
            let digits = dimensions.iter().try_fold(0usize, |length, dimension| {
                length.checked_add(decimal_len(*dimension))
            });
            digits
                .and_then(|length| {
                    (dimensions.len() - 1)
                        .checked_mul(2)
                        .and_then(|separators| length.checked_add(separators))
                })
                .and_then(|length| length.checked_add(2))
                .ok_or_else(|| format_error("header length is out of range"))?
        }
    };
    let body_len = "{'descr': '"
        .len()
        .checked_add(descr_len(dtype))
        .and_then(|length| length.checked_add("', 'fortran_order': False, 'shape': ".len()))
        .and_then(|length| length.checked_add(shape_len))
        .and_then(|length| length.checked_add(", }".len()))
        .ok_or_else(|| format_error("header length is out of range"))?;
    let framing_len = NPY_MAGIC.len() + 2 + 2;
    let unpadded = framing_len
        .checked_add(body_len)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| format_error("header length is out of range"))?;
    let padding = (64 - unpadded % 64) % 64;
    let header_len = body_len
        .checked_add(padding)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| format_error("header length is out of range"))?;
    if header_len > MAX_HEADER_LEN {
        return Err(format_error("header length is out of range"));
    }
    u16::try_from(header_len).map_err(|_| format_error("header length is out of range"))
}

/// Bounds the output capacity and live header/shape formatting workspace.
/// The unbudgeted encoder remains available to NPZ, which owns codec workspace.
pub(crate) fn encode_npy_budgeted(
    shape: &[usize],
    buffer: &Buffer,
    budget: &MemoryBudget,
) -> Result<(Vec<u8>, Reservation)> {
    let header_len = usize::from(checked_npy_header_len(shape, buffer.dtype())?);
    let payload_len = usize::try_from(buffer.byte_len())
        .map_err(|_| format_error("serialization size is out of range"))?;
    let capacity = (NPY_MAGIC.len() + 2 + 2)
        .checked_add(header_len)
        .and_then(|size| size.checked_add(64))
        .and_then(|size| size.checked_add(payload_len))
        .ok_or_else(|| format_error("serialization size is out of range"))?;
    // Shape formatting can hold one String per axis, the collection of those
    // Strings, a joined shape, and the final header alongside the output.
    let workspace = header_len
        .checked_mul(4)
        .and_then(|size| {
            shape
                .len()
                .checked_mul(2 * std::mem::size_of::<String>())
                .and_then(|strings| size.checked_add(strings))
        })
        .and_then(|size| size.checked_add(64))
        .ok_or_else(|| format_error("serialization size is out of range"))?;
    let reservation_bytes = capacity
        .checked_add(workspace)
        .ok_or_else(|| format_error("serialization size is out of range"))?;
    let reservation = budget.reserve(
        u64::try_from(reservation_bytes)
            .map_err(|_| format_error("serialization size is out of range"))?,
    )?;
    Ok((encode_npy(shape, buffer)?, reservation))
}

/// Encodes a C-order array as NPY version 1.0.
pub fn encode_npy(shape: &[usize], buffer: &Buffer) -> Result<Vec<u8>> {
    let header_len = checked_npy_header_len(shape, buffer.dtype())?;
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
    if header.len() != usize::from(header_len) {
        return Err(format_error("header length is out of range"));
    }
    let mut out = Vec::with_capacity(unpadded + 64 + buffer.byte_len() as usize);
    out.extend_from_slice(NPY_MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&header_len.to_le_bytes());
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

struct HeaderParser<'a> {
    text: &'a str,
    offset: usize,
}

impl<'a> HeaderParser<'a> {
    fn skip_space(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.offset += 1;
        }
    }

    fn take(&mut self, byte: u8) -> bool {
        self.skip_space();
        if self.text.as_bytes().get(self.offset) == Some(&byte) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, byte: u8) -> Result<()> {
        if self.take(byte) {
            Ok(())
        } else {
            Err(format_error("malformed header dictionary"))
        }
    }

    fn string(&mut self) -> Result<&'a str> {
        self.skip_space();
        let quote = match self.text.as_bytes().get(self.offset) {
            Some(b'\'' | b'"') => self.text.as_bytes()[self.offset],
            _ => return Err(format_error("expected a quoted header string")),
        };
        self.offset += 1;
        let start = self.offset;
        while let Some(&byte) = self.text.as_bytes().get(self.offset) {
            if byte == quote {
                let value = &self.text[start..self.offset];
                self.offset += 1;
                return Ok(value);
            }
            if byte == b'\\' || byte == b'\n' || byte == b'\r' {
                return Err(format_error("malformed header string"));
            }
            self.offset += 1;
        }
        Err(format_error("unterminated header string"))
    }

    fn boolean(&mut self) -> Result<bool> {
        self.skip_space();
        let rest = &self.text[self.offset..];
        let value = if rest.starts_with("True") {
            self.offset += 4;
            true
        } else if rest.starts_with("False") {
            self.offset += 5;
            false
        } else {
            return Err(format_error("malformed fortran_order"));
        };
        Ok(value)
    }

    fn shape(&mut self) -> Result<Vec<usize>> {
        self.expect(b'(')?;
        let mut shape = Vec::new();
        if self.take(b')') {
            return Ok(shape);
        }
        loop {
            self.skip_space();
            let start = self.offset;
            while self
                .text
                .as_bytes()
                .get(self.offset)
                .is_some_and(u8::is_ascii_digit)
            {
                self.offset += 1;
            }
            if start == self.offset {
                return Err(format_error("malformed shape"));
            }
            shape.push(
                self.text[start..self.offset]
                    .parse::<usize>()
                    .map_err(|_| format_error("malformed shape"))?,
            );
            if self.take(b',') {
                if self.take(b')') {
                    return Ok(shape);
                }
            } else if self.take(b')') && shape.len() > 1 {
                return Ok(shape);
            } else {
                return Err(format_error("malformed shape"));
            }
        }
    }
}

fn parse_header(header: &str) -> Result<(&str, bool, Vec<usize>)> {
    let mut parser = HeaderParser {
        text: header,
        offset: 0,
    };
    parser.expect(b'{')?;
    let (mut descr, mut fortran, mut shape) = (None, None, None);
    loop {
        if parser.take(b'}') {
            break;
        }
        let key = parser.string()?;
        parser.expect(b':')?;
        match key {
            "descr" if descr.is_none() => descr = Some(parser.string()?),
            "fortran_order" if fortran.is_none() => fortran = Some(parser.boolean()?),
            "shape" if shape.is_none() => shape = Some(parser.shape()?),
            _ => return Err(format_error("unknown or duplicate header field")),
        }
        if parser.take(b',') {
            continue;
        }
        parser.expect(b'}')?;
        break;
    }
    parser.skip_space();
    if parser.offset != header.len() {
        return Err(format_error("malformed header dictionary"));
    }
    Ok((
        descr.ok_or_else(|| format_error("missing 'descr'"))?,
        fortran.ok_or_else(|| format_error("missing 'fortran_order'"))?,
        shape.ok_or_else(|| format_error("missing 'shape'"))?,
    ))
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
    let (descr_text, fortran, shape) = parse_header(header)?;
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
            DType::from_id(&format!("str[{}]", &other[2..]))
                .map_err(|_| format_error("unsupported dtype descriptor"))?
        }
        other if other.starts_with("<M8[") || other.starts_with(">M8[") => {
            DType::from_id(&format!("datetime64{}", &other[3..]))
                .map_err(|_| format_error("unsupported dtype descriptor"))?
        }
        other if other.starts_with("<m8[") || other.starts_with(">m8[") => {
            DType::from_id(&format!("timedelta64{}", &other[3..]))
                .map_err(|_| format_error("unsupported dtype descriptor"))?
        }
        _ => return Err(format_error("unsupported dtype descriptor")),
    };
    if fortran
        && !matches!(
            dtype,
            DType::Float16
                | DType::Complex64
                | DType::Complex128
                | DType::Str(_)
                | DType::Datetime64(_)
                | DType::Timedelta64(_)
        )
    {
        return Err(format_error(
            "Fortran-order data is not supported in this build",
        ));
    }
    let count = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format_error("shape overflows"))?
    };
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

fn next_temp_sequence(sequence: &AtomicU64) -> Result<u64> {
    let mut value = sequence.load(Ordering::Relaxed);
    loop {
        let next = value.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::IoFailed,
                "The temporary pathname sequence is exhausted.",
            )
        })?;
        match sequence.compare_exchange_weak(value, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Ok(value),
            Err(observed) => value = observed,
        }
    }
}

fn temp_path_at(dest: &Path, timestamp: u128, sequence: u64) -> Result<PathBuf> {
    let parent = dest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    dest.file_name()
        .ok_or_else(|| Error::new(ErrorCode::IoFailed, "The output path has no file name."))?;
    Ok(parent.join(format!(
        ".nmath-{}-{timestamp}-{sequence}.tmp",
        std::process::id()
    )))
}

fn temp_path(dest: &Path) -> Result<PathBuf> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let sequence = next_temp_sequence(&TEMP_SEQUENCE)?;
    temp_path_at(dest, timestamp, sequence)
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
    write_atomic_with_path(dest, &tmp, mode, token, |file| {
        for chunk in bytes.chunks(1 << 20) {
            token.check()?;
            file.write_all(chunk).map_err(io_error)?;
        }
        file.sync_all().map_err(io_error)
    })
}

#[cfg(test)]
fn write_atomic_at(
    dest: &Path,
    bytes: &[u8],
    mode: WriteMode,
    token: &CancellationToken,
    timestamp: u128,
) -> Result<()> {
    if mode == WriteMode::CreateNew && dest.exists() {
        return Err(Error::new(
            ErrorCode::IoExists,
            "The output file already exists.",
        ));
    }
    let sequence = next_temp_sequence(&TEMP_SEQUENCE)?;
    let tmp = temp_path_at(dest, timestamp, sequence)?;
    write_atomic_with_path(dest, &tmp, mode, token, |file| {
        for chunk in bytes.chunks(1 << 20) {
            token.check()?;
            file.write_all(chunk).map_err(io_error)?;
        }
        file.sync_all().map_err(io_error)
    })
}

fn write_atomic_with_path(
    dest: &Path,
    tmp: &Path,
    mode: WriteMode,
    token: &CancellationToken,
    write_and_sync: impl FnOnce(&mut File) -> Result<()>,
) -> Result<()> {
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(tmp)
        .map_err(io_error)?;
    let result = (|| {
        write_and_sync(&mut file)?;
        token.check()?;
        match mode {
            // A hard link fails if the destination exists, so publication never
            // clobbers a file created concurrently.
            WriteMode::CreateNew => fs::hard_link(tmp, dest).map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    Error::new(ErrorCode::IoExists, "The output file already exists.")
                } else {
                    io_error(e)
                }
            }),
            WriteMode::Overwrite => fs::rename(tmp, dest).map_err(io_error),
        }
    })();
    drop(file);
    // Exclusive creation succeeded, so this operation owns the temporary path.
    let _ = fs::remove_file(tmp);
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
        sync::mpsc,
        time::{Duration, UNIX_EPOCH},
    };

    use super::*;

    fn fixture(header: &str, version: u8, payload: &[u8]) -> Vec<u8> {
        let prefix_len = if version == 1 { 10 } else { 12 };
        let mut padded = header.as_bytes().to_vec();
        padded.extend(vec![b' '; (64 - (prefix_len + padded.len() + 1) % 64) % 64]);
        padded.push(b'\n');
        let mut bytes = NPY_MAGIC.to_vec();
        bytes.extend_from_slice(&[version, 0]);
        if version == 1 {
            bytes.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        } else {
            bytes.extend_from_slice(&(padded.len() as u32).to_le_bytes());
        }
        bytes.extend_from_slice(&padded);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn accepts_equivalent_header_dictionaries_across_versions() {
        let headers = [
            "{\"descr\": \"<f8\", \"fortran_order\": False, \"shape\": (1,), }",
            "{'descr' : '<f8', 'fortran_order' : False, 'shape' : (1,), }",
            "{'descr': \"<f8\", 'fortran_order': False, 'shape': (1,), }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }",
            "{ 'shape' : ( 1 , ) , \"fortran_order\" : False , 'descr' : \"<f8\" }",
        ];
        for version in [1, 2, 3] {
            for header in headers {
                let bytes = fixture(header, version, &1.25f64.to_le_bytes());
                assert_eq!(
                    decode_npy(&bytes).unwrap(),
                    (vec![1], Buffer::F64(vec![1.25]))
                );
            }
        }
    }

    #[test]
    fn rejects_malformed_header_dictionaries_before_reserving_output() {
        let malformed = [
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), } junk",
            "{'descr': '<f8', 'fortran_order': False 'shape': (1,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), 'shape': (1,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), 'extra': 1}",
            "{'fortran_order': False, 'shape': (1,)}",
            "{'descr': 8, 'fortran_order': False, 'shape': (1,)}",
            "{'descr': '<f8', 'fortran_order': 'False', 'shape': (1,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': [1]}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (-1,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (18446744073709551615, 2)}",
            "{'descr': '<x8', 'fortran_order': False, 'shape': (1,)}",
            "{'descr': '<Ubogus', 'fortran_order': False, 'shape': (1,)}",
            "{'descr': '<M8[foo]', 'fortran_order': False, 'shape': (1,)}",
            "{'descr': '<m8[foo]', 'fortran_order': False, 'shape': (1,)}",
            "{'descr': '<f8, 'fortran_order': False, 'shape': (1,)}",
        ];
        for header in malformed {
            let budget = MemoryBudget::new(crate::memory::Device::Cpu, 8);
            let bytes = fixture(header, 1, &1.25f64.to_le_bytes());
            assert_eq!(
                decode_npy_budgeted(&bytes, &budget).unwrap_err().code,
                ErrorCode::IoFormat,
                "{header}"
            );
            assert_eq!(budget.usage().used, 0);
        }
        for payload in [&[][..], &[0; 16][..]] {
            assert_eq!(
                decode_npy(&fixture(
                    "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}",
                    1,
                    payload
                ))
                .unwrap_err()
                .code,
                ErrorCode::IoFormat
            );
        }
    }

    fn max_component_len(parent: &Path) -> usize {
        let probe = |len: usize| {
            let path = parent.join("x".repeat(len));
            if fs::write(&path, []).is_ok() {
                fs::remove_file(path).unwrap();
                true
            } else {
                false
            }
        };
        assert!(probe(1), "fixture directory must be writable");
        let (mut valid, mut invalid) = (1, 4096);
        assert!(!probe(invalid), "increase the component-limit probe bound");
        while valid + 1 < invalid {
            let candidate = valid + (invalid - valid) / 2;
            if probe(candidate) {
                valid = candidate;
            } else {
                invalid = candidate;
            }
        }
        assert!(probe(valid));
        assert!(!probe(valid + 1));
        valid
    }

    fn max_name(limit: usize, extension: &str) -> String {
        let name = format!("{}.{}", "a".repeat(limit - extension.len() - 1), extension);
        assert_eq!(name.len(), limit);
        name
    }

    fn run_long_filename_fixture(root: &Path) {
        let direct = root.join("direct");
        let writers = root.join("writers");
        fs::create_dir(&direct).unwrap();
        fs::create_dir(&writers).unwrap();
        let limit = max_component_len(&direct);
        assert_eq!(max_component_len(&writers), limit);
        if let Some(expected) = std::env::var_os("NMATH_PUBLICATION_NAME_MAX") {
            assert_eq!(limit, expected.to_str().unwrap().parse::<usize>().unwrap());
        }

        let dest = direct.join(max_name(limit, "npy"));
        // Hold naming inputs fixed: independent calls can cross decimal-width
        // boundaries. Maximum values also exercise the component-length bound.
        let tmp = temp_path_at(&dest, u128::MAX, u64::MAX).unwrap();
        let short_tmp = temp_path_at(&direct.join("short.npy"), u128::MAX, u64::MAX).unwrap();
        assert_eq!(tmp.parent(), dest.parent());
        assert_eq!(
            tmp.file_name().unwrap().len(),
            short_tmp.file_name().unwrap().len()
        );
        assert!(tmp.file_name().unwrap().len() < limit);
        assert!(
            tmp.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".nmath-")
        );
        let token = CancellationToken::new();
        write_atomic(&dest, b"first", WriteMode::CreateNew, &token).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"first");
        assert_eq!(
            write_atomic(&dest, b"clobber", WriteMode::CreateNew, &token)
                .unwrap_err()
                .code,
            ErrorCode::IoExists
        );
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert_eq!(
            write_atomic(&dest, b"clobber", WriteMode::Overwrite, &cancelled)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(fs::read(&dest).unwrap(), b"first");
        let failed_tmp = temp_path(&dest).unwrap();
        let error =
            write_atomic_with_path(&dest, &failed_tmp, WriteMode::Overwrite, &token, |file| {
                file.write_all(b"partial").map_err(io_error)?;
                Err(Error::new(ErrorCode::IoFailed, "Fixture write failure."))
            })
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::IoFailed);
        assert_eq!(fs::read(&dest).unwrap(), b"first");
        assert!(!failed_tmp.exists());
        write_atomic(&dest, b"second", WriteMode::Overwrite, &token).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"second");
        let short = direct.join("short.npy");
        write_atomic(&short, b"short first", WriteMode::CreateNew, &token).unwrap();
        write_atomic(&short, b"short second", WriteMode::Overwrite, &token).unwrap();
        assert_eq!(fs::read(&short).unwrap(), b"short second");
        assert_eq!(fs::read_dir(&direct).unwrap().count(), 2);

        let npy = writers.join(max_name(limit, "npy"));
        let ctx = crate::Context::new(crate::ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: 8,
            cpu_memory_budget: 128 << 20,
            gpu_memory_budget: 0,
        })
        .unwrap();
        let array = ctx.array(vec![1], Buffer::I32(vec![42])).unwrap();
        for mode in [WriteMode::CreateNew, WriteMode::Overwrite] {
            let (tx, rx) = mpsc::channel();
            ctx.save_npy(
                &array,
                npy.clone(),
                mode,
                CancellationToken::new(),
                Box::new(move |result| tx.send(result).unwrap()),
            )
            .unwrap();
            assert!(matches!(
                rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap(),
                crate::context::JobOutput::Unit
            ));
            assert_eq!(
                decode_npy(&fs::read(&npy).unwrap()).unwrap(),
                (vec![1], Buffer::I32(vec![42]))
            );
        }
        ctx.close();

        let budget = MemoryBudget::new(crate::memory::Device::Cpu, 128 << 20);
        let npz = writers.join(max_name(limit, "npz"));
        for mode in [WriteMode::CreateNew, WriteMode::Overwrite] {
            crate::io_npz::save_npz(
                &npz,
                &[],
                crate::io_npz::NpzCompression::Stored,
                mode,
                &budget,
                &token,
            )
            .unwrap();
            assert!(
                crate::io_npz::load_npz(&npz, &budget, &token)
                    .unwrap()
                    .arrays
                    .is_empty()
            );
        }

        let table = crate::table::Table {
            columns: vec![crate::table::MaterializedColumn {
                values: crate::buffer::DenseArray::new(
                    vec![1],
                    Buffer::I64(vec![7]),
                    Reservation::none(),
                )
                .unwrap(),
                valid: crate::buffer::DenseArray::new(
                    vec![1],
                    Buffer::Bool(vec![true]),
                    Reservation::none(),
                )
                .unwrap(),
            }],
            rows: 1,
            pivot_columns: None,
        };
        let names = ["value".to_owned()];
        for (format, extension, expected) in [
            (crate::io_text::Format::Csv, "csv", &b"value\n7\n"[..]),
            (
                crate::io_text::Format::Json,
                "json",
                &b"[{\"value\":7}]"[..],
            ),
        ] {
            let path = writers.join(max_name(limit, extension));
            for mode in [WriteMode::CreateNew, WriteMode::Overwrite] {
                crate::io_text::write(&path, format, &names, &table, mode, &budget, &token)
                    .unwrap();
                assert_eq!(fs::read(&path).unwrap(), expected);
            }
        }
        assert_eq!(fs::read_dir(&writers).unwrap().count(), 4);
        println!(
            "component limit {limit}; temporary component {} bytes; direct, NPY, NPZ, CSV, JSON \
             create-new and overwrite passed",
            tmp.file_name().unwrap().len()
        );
    }

    #[test]
    fn maximum_valid_component_publishes_through_all_common_writers() {
        let root = tempfile::tempdir().unwrap();
        run_long_filename_fixture(root.path());
    }

    #[test]
    #[ignore = "run with NMATH_PUBLICATION_FIXTURE_DIR for native validation"]
    fn native_long_filename_fixture() {
        let root = PathBuf::from(std::env::var_os("NMATH_PUBLICATION_FIXTURE_DIR").unwrap());
        assert!(root.is_dir());
        assert!(fs::read_dir(&root).unwrap().next().is_none());
        run_long_filename_fixture(&root);
    }

    #[test]
    fn descriptor_preflight_matches_encoded_descriptor_lengths() {
        for dtype in DType::ALL.into_iter().chain([
            DType::Str(1234),
            DType::Datetime64(TimeUnit {
                base: TimeBase::Milli,
                step: 12,
            }),
        ]) {
            assert_eq!(descr_len(dtype), descr(dtype).unwrap().len());
        }
    }

    #[test]
    fn budgeted_npy_encode_charges_live_serialization_and_releases_it() {
        let buffer = Buffer::F64(vec![]);
        let zero = MemoryBudget::new(crate::memory::Device::Cpu, 0);
        assert_eq!(
            encode_npy_budgeted(&[0], &buffer, &zero).unwrap_err().code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(zero.usage().used, 0);
        assert_eq!(zero.usage().peak, 0);

        let enough = MemoryBudget::new(crate::memory::Device::Cpu, 4096);
        let (bytes, reservation) = encode_npy_budgeted(&[0], &buffer, &enough).unwrap();
        assert_eq!(bytes.len(), 128);
        assert!(reservation.bytes() >= bytes.capacity() as u64);
        assert_eq!(enough.usage().used, reservation.bytes());
        assert_eq!(enough.usage().peak, reservation.bytes());
        drop(reservation);
        assert_eq!(enough.usage().used, 0);
        assert_eq!(decode_npy(&bytes).unwrap(), (vec![0], buffer));
    }

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
    fn npy_header_length_matches_output_and_rejects_unsupported_ranks() {
        let buffer = Buffer::U8(vec![1]);
        for (rank, expected_len) in [(0, 118), (1, 118), (3_300, 9_974), (3_306, 9_974)] {
            let shape = vec![1; rank];
            let bytes = encode_npy(&shape, &buffer).unwrap();
            let declared = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            assert_eq!(declared, expected_len);
            assert_eq!(
                declared,
                checked_npy_header_len(&shape, buffer.dtype()).unwrap() as usize
            );
            assert_eq!(bytes.len(), 10 + declared + 1);
            assert_eq!(decode_npy(&bytes).unwrap(), (shape, buffer.clone()));
        }
        for rank in [3_307, 4_000, 22_000] {
            let shape = vec![1; rank];
            assert_eq!(
                encode_npy(&shape, &buffer).unwrap_err().code,
                ErrorCode::IoFormat
            );
        }
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

    #[test]
    fn atomic_write_collision_preserves_unowned_temporary_file() {
        for mode in [WriteMode::CreateNew, WriteMode::Overwrite] {
            let dir = tempfile::tempdir().unwrap();
            let dest = dir.path().join("destination.npy");
            let tmp = dir.path().join("occupied.tmp");
            if mode == WriteMode::Overwrite {
                fs::write(&dest, b"old destination").unwrap();
            }
            fs::write(&tmp, b"other writer temporary").unwrap();

            let error =
                write_atomic_with_path(&dest, &tmp, mode, &CancellationToken::new(), |_| {
                    panic!("exclusive creation must fail before writing")
                })
                .unwrap_err();
            assert_eq!(error.code, ErrorCode::IoFailed);
            assert!(!error.full_message().contains(dir.path().to_str().unwrap()));
            assert!(!error.full_message().contains("other writer temporary"));
            assert_eq!(fs::read(&tmp).unwrap(), b"other writer temporary");
            if mode == WriteMode::Overwrite {
                assert_eq!(fs::read(&dest).unwrap(), b"old destination");
            } else {
                assert!(!dest.exists());
            }
        }
    }

    #[test]
    fn concurrent_equal_clock_writes_use_distinct_owned_temporary_files() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.npy");
        let second = dir.path().join("second.npy");
        let timestamp = 123_u128;

        std::thread::scope(|scope| {
            let first_job = scope.spawn(|| {
                write_atomic_at(
                    &first,
                    b"first",
                    WriteMode::CreateNew,
                    &CancellationToken::new(),
                    timestamp,
                )
            });
            let second_job = scope.spawn(|| {
                write_atomic_at(
                    &second,
                    b"second",
                    WriteMode::CreateNew,
                    &CancellationToken::new(),
                    timestamp,
                )
            });
            first_job.join().unwrap().unwrap();
            second_job.join().unwrap().unwrap();
        });

        assert_eq!(fs::read(&first).unwrap(), b"first");
        assert_eq!(fs::read(&second).unwrap(), b"second");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[test]
    fn temporary_sequence_exhaustion_is_reported_without_wraparound() {
        let sequence = AtomicU64::new(u64::MAX);
        let error = next_temp_sequence(&sequence).unwrap_err();
        assert_eq!(error.code, ErrorCode::IoFailed);
        assert_eq!(sequence.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn atomic_write_removes_owned_temporary_file_on_prepublication_failure() {
        for mode in [WriteMode::CreateNew, WriteMode::Overwrite] {
            let dir = tempfile::tempdir().unwrap();
            let dest = dir.path().join("destination.npy");
            let tmp = dir.path().join("owned.tmp");
            if mode == WriteMode::Overwrite {
                fs::write(&dest, b"old destination").unwrap();
            }
            let token = CancellationToken::new();
            let error = write_atomic_with_path(&dest, &tmp, mode, &token, |file| {
                file.write_all(b"partial output").map_err(io_error)?;
                token.cancel();
                Ok(())
            })
            .unwrap_err();
            assert_eq!(error.code, ErrorCode::Cancelled);
            assert!(!tmp.exists());
            if mode == WriteMode::Overwrite {
                assert_eq!(fs::read(&dest).unwrap(), b"old destination");
            } else {
                assert!(!dest.exists());
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("destination.npy");
        let tmp = dir.path().join("owned.tmp");
        fs::write(&dest, b"old destination").unwrap();
        for operation in ["write", "sync"] {
            let error = write_atomic_with_path(
                &dest,
                &tmp,
                WriteMode::Overwrite,
                &CancellationToken::new(),
                |file| {
                    if operation == "sync" {
                        file.write_all(b"partial output").map_err(io_error)?;
                    }
                    Err(io_error(std::io::Error::other(operation)))
                },
            )
            .unwrap_err();
            assert_eq!(error.code, ErrorCode::IoFailed);
            assert!(!tmp.exists());
            assert_eq!(fs::read(&dest).unwrap(), b"old destination");
        }

        let missing_parent = dir.path().join("missing");
        let dest = missing_parent.join("destination.npy");
        let error = write_atomic_with_path(
            &dest,
            &tmp,
            WriteMode::Overwrite,
            &CancellationToken::new(),
            |file| file.write_all(b"complete output").map_err(io_error),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::IoFailed);
        assert!(!tmp.exists());
        assert!(!dest.exists());

        let dest = dir.path().join("raced.npy");
        let error = write_atomic_with_path(
            &dest,
            &tmp,
            WriteMode::CreateNew,
            &CancellationToken::new(),
            |file| {
                file.write_all(b"new output").map_err(io_error)?;
                fs::write(&dest, b"racing writer").map_err(io_error)
            },
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::IoExists);
        assert!(!tmp.exists());
        assert_eq!(fs::read(&dest).unwrap(), b"racing writer");
    }

    #[test]
    #[ignore = "run with NMATH_PUBLICATION_FIXTURE_DIR for native validation"]
    fn native_publication_fixture() {
        let root = PathBuf::from(std::env::var_os("NMATH_PUBLICATION_FIXTURE_DIR").unwrap());
        assert!(root.is_dir());
        assert!(fs::read_dir(&root).unwrap().next().is_none());

        for mode in [WriteMode::CreateNew, WriteMode::Overwrite] {
            let name = if mode == WriteMode::CreateNew {
                "create-new"
            } else {
                "overwrite"
            };
            let dir = root.join(name);
            fs::create_dir(&dir).unwrap();
            let dest = dir.join("destination.npy");
            let tmp = dir.join("occupied.tmp");
            if mode == WriteMode::Overwrite {
                fs::write(&dest, b"old destination").unwrap();
            }
            fs::write(&tmp, b"other writer temporary").unwrap();
            let error =
                write_atomic_with_path(&dest, &tmp, mode, &CancellationToken::new(), |_| {
                    panic!("occupied temp must reject exclusive creation")
                })
                .unwrap_err();
            assert_eq!(error.code, ErrorCode::IoFailed);
            assert_eq!(fs::read(&tmp).unwrap(), b"other writer temporary");
            if mode == WriteMode::Overwrite {
                assert_eq!(fs::read(&dest).unwrap(), b"old destination");
            } else {
                assert!(!dest.exists());
            }
            println!("{name}: occupied temporary file and destination preserved");
        }

        let dir = root.join("owned-cancellation");
        fs::create_dir(&dir).unwrap();
        let dest = dir.join("destination.npy");
        let tmp = dir.join("owned.tmp");
        fs::write(&dest, b"old destination").unwrap();
        let token = CancellationToken::new();
        let error = write_atomic_with_path(&dest, &tmp, WriteMode::Overwrite, &token, |file| {
            file.write_all(b"partial output").map_err(io_error)?;
            token.cancel();
            Ok(())
        })
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::Cancelled);
        assert!(!tmp.exists());
        assert_eq!(fs::read(&dest).unwrap(), b"old destination");
        println!("owned cancellation: temporary file removed; destination preserved");

        let dest = root.join("normal.npy");
        let token = CancellationToken::new();
        write_atomic(&dest, b"first", WriteMode::CreateNew, &token).unwrap();
        write_atomic(&dest, b"second", WriteMode::Overwrite, &token).unwrap();
        token.cancel();
        assert_eq!(fs::read(&dest).unwrap(), b"second");
        println!(
            "normal publication: create-new and overwrite succeeded; later cancellation did not \
             undo output"
        );
    }
}
