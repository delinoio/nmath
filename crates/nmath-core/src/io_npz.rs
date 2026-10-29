//! Budgeted NPZ archives. Members are NPY data, never extracted paths.
//!
//! Names are single UTF-8 components; `.npy` is appended on save and removed
//! once on load. Duplicate names, encrypted members and non-NPY members fail.
//! ZIP64 local headers (used by NumPy) are supported. Multi-disk archives and
//! ZIP64 central directories are rejected rather than allocating from their
//! unbounded counts. NPY dtype/order support follows the shared NPY codec.

use std::{
    fs::File,
    io::{Cursor, Read, Write},
    path::Path,
};

use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

use crate::{
    buffer::DenseArray,
    error::{Error, ErrorCode, Result},
    io::{WriteMode, decode_npy_budgeted, encode_npy, write_atomic},
    memory::{MemoryBudget, Reservation},
    sync::CancellationToken,
};

// Conservative workspace bounds for the pinned pure-Rust DEFLATE backend and
// ZIP metadata. They include codec tables, NPY header temporaries and names.
// Reassess these bounds when changing zip/flate2/miniz_oxide versions.
const CODEC_WORKSPACE: u64 = 4 << 20;
const MEMBER_METADATA: u64 = 4096;
const CHUNK: usize = 64 << 10;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NpzCompression {
    #[default]
    Stored,
    Deflated,
}

#[derive(Debug)]
pub struct NamedArray {
    pub name: String,
    pub array: DenseArray,
}

/// Owns both decoded arrays and the charge for their names/shape metadata.
#[derive(Debug)]
pub struct NpzArchive {
    pub arrays: Vec<NamedArray>,
    _metadata: Reservation,
}

fn invalid() -> Error {
    Error::new(
        ErrorCode::IoFormat,
        "The NPZ archive is malformed or unsupported. Use a stored or deflated archive of \
         non-object NPY arrays.",
    )
}
fn io_failed(_: std::io::Error) -> Error {
    Error::new(
        ErrorCode::IoFailed,
        "The NPZ file operation failed. Check local file permissions.",
    )
}
fn checked_add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(invalid)
}
fn checked_mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(invalid)
}
fn name_valid(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name.len() <= 65531
        && !name
            .chars()
            .any(|c| c.is_control() || matches!(c, '/' | '\\' | ':'))
}

fn u16_at(b: &[u8], i: usize) -> Result<usize> {
    Ok(u16::from_le_bytes(b.get(i..i + 2).ok_or_else(invalid)?.try_into().unwrap()) as usize)
}
fn u32_at(b: &[u8], i: usize) -> Result<usize> {
    Ok(u32::from_le_bytes(b.get(i..i + 4).ok_or_else(invalid)?.try_into().unwrap()) as usize)
}

/// Validate directory bounds and count without allocating. This prevents the
/// ZIP parser from allocating from an untrusted count before budget admission.
fn directory(bytes: &[u8]) -> Result<(usize, usize, usize)> {
    let end = bytes.len().checked_sub(22).ok_or_else(invalid)?;
    let eocd = (end.saturating_sub(65535)..=end)
        .rev()
        .find(|&p| {
            bytes[p..].starts_with(b"PK\x05\x06")
                && u16_at(bytes, p + 20).is_ok_and(|n| p + 22 + n == bytes.len())
        })
        .ok_or_else(invalid)?;
    let count = u16_at(bytes, eocd + 10)?;
    let size = u32_at(bytes, eocd + 12)?;
    let start = u32_at(bytes, eocd + 16)?;
    if u16_at(bytes, eocd + 4)? != 0
        || u16_at(bytes, eocd + 6)? != 0
        || count != u16_at(bytes, eocd + 8)?
        || count == 65535
        || size == u32::MAX as usize
        || start == u32::MAX as usize
        || start.checked_add(size) != Some(eocd)
    {
        return Err(invalid());
    }
    let mut p = start;
    for _ in 0..count {
        if !bytes.get(p..).is_some_and(|s| s.starts_with(b"PK\x01\x02")) {
            return Err(invalid());
        }
        let n = u16_at(bytes, p + 28)?;
        let extra = u16_at(bytes, p + 30)?;
        let comment = u16_at(bytes, p + 32)?;
        if u16_at(bytes, p + 34)? != 0
            || u32_at(bytes, p + 20)? == u32::MAX as usize
            || u32_at(bytes, p + 24)? == u32::MAX as usize
            || u32_at(bytes, p + 42)? == u32::MAX as usize
        {
            return Err(invalid());
        }
        p = p
            .checked_add(46 + n + extra + comment)
            .ok_or_else(invalid)?;
        if p > eocd {
            return Err(invalid());
        }
    }
    if p != eocd {
        return Err(invalid());
    }
    Ok((count, start, size))
}

/// Decode borrowed archive bytes. The caller must keep the encoded-byte
/// reservation alive; this function charges metadata, decompression and
/// results.
pub fn decode_npz(
    bytes: &[u8],
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<NpzArchive> {
    token.check()?;
    let (count, central, directory_size) = directory(bytes)?;
    let metadata = budget.reserve(checked_add(
        checked_mul(count as u64, MEMBER_METADATA)?,
        checked_mul(directory_size as u64, 8)?,
    )?)?;
    let _workspace = budget.reserve(CODEC_WORKSPACE)?;
    let mut zip = ZipArchive::new(Cursor::new(bytes)).map_err(|_| invalid())?;
    if zip.len() != count {
        return Err(invalid());
    }
    let mut arrays: Vec<NamedArray> = Vec::with_capacity(count);
    zip.validate_header_consistency().map_err(|_| invalid())?;
    for i in 0..count {
        token.check()?;
        let mut member = zip.by_index(i).map_err(|_| invalid())?;
        let member_name = member.name().map_err(|_| invalid())?;
        let name = member_name.strip_suffix(".npy").ok_or_else(invalid)?;
        if !name_valid(name)
            || member.encrypted()
            || member.is_dir()
            || member
                .unix_mode()
                .is_some_and(|m| m & 0o170000 != 0 && m & 0o170000 != 0o100000)
            || !matches!(
                member.compression(),
                CompressionMethod::Stored | CompressionMethod::Deflated
            )
        {
            return Err(invalid());
        }
        let start = member.data_start().ok_or_else(invalid)?;
        let end = checked_add(start, member.compressed_size())?;
        // Consistency validation above rejects duplicate/overlapping records.
        // Member data must also stay outside the central directory.
        if end > central as u64 {
            return Err(invalid());
        }
        let local = usize::try_from(member.header_start()).map_err(|_| invalid())?;
        let local_name_len = u16_at(bytes, local + 26)?;
        if bytes.get(local + 30..local + 30 + local_name_len) != Some(member_name.as_bytes()) {
            return Err(invalid());
        }
        let name = name.to_owned();
        let size = usize::try_from(member.size()).map_err(|_| invalid())?;
        let _encoded = budget.reserve(size as u64)?;
        let mut npy = vec![0; size];
        for chunk in npy.chunks_mut(CHUNK) {
            token.check()?;
            member.read_exact(chunk).map_err(|_| invalid())?;
        }
        // An EOF read verifies CRC and detects a dishonest uncompressed size.
        let mut extra = [0];
        if member.read(&mut extra).map_err(|_| invalid())? != 0 {
            return Err(invalid());
        }
        token.check()?;
        let (shape, buffer, reservation) = decode_npy_budgeted(&npy, budget).map_err(|e| {
            // The shared codec can quote an attacker-controlled descriptor.
            // NPZ diagnostics must not reveal member contents.
            if e.code == ErrorCode::MemoryBudgetExceeded {
                e
            } else {
                invalid()
            }
        })?;
        arrays.push(NamedArray {
            name,
            array: DenseArray::new(shape, buffer, reservation)?,
        });
    }
    token.check()?;
    Ok(NpzArchive {
        arrays,
        _metadata: metadata,
    })
}

/// Read a stable-size file into a precharged buffer. The final pathname check
/// also detects replacement while decompression was in progress (best effort).
pub fn load_npz(
    path: &Path,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<NpzArchive> {
    load_npz_with_decoder(path, budget, token, decode_npz)
}

// Keep the complete read/decode interval under the same modification check.
// The private decoder parameter also lets tests change the file at that
// boundary.
fn load_npz_with_decoder(
    path: &Path,
    budget: &MemoryBudget,
    token: &CancellationToken,
    decode: impl FnOnce(&[u8], &MemoryBudget, &CancellationToken) -> Result<NpzArchive>,
) -> Result<NpzArchive> {
    token.check()?;
    let mut file = File::open(path).map_err(io_failed)?;
    let stamp = |m: std::fs::Metadata| (m.len(), m.modified().ok());
    let before = stamp(file.metadata().map_err(io_failed)?);
    let size = usize::try_from(before.0).map_err(|_| invalid())?;
    let _encoded = budget.reserve(before.0)?;
    let mut bytes = vec![0; size];
    for chunk in bytes.chunks_mut(CHUNK) {
        token.check()?;
        if let Err(e) = file.read_exact(chunk) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                return Err(modified());
            }
            return Err(io_failed(e));
        }
    }
    let mut extra = [0];
    if file.read(&mut extra).map_err(io_failed)? != 0 {
        return Err(modified());
    }
    let result = decode(&bytes, budget, token);
    if stamp(file.metadata().map_err(io_failed)?) != before
        || stamp(std::fs::metadata(path).map_err(io_failed)?) != before
    {
        return Err(modified());
    }
    result
}
fn modified() -> Error {
    Error::new(
        ErrorCode::IoModified,
        "The input file changed during NPZ execution. Keep it unchanged and retry.",
    )
}

/// Encode into a fixed-capacity, precharged archive buffer. A conservative
/// DEFLATE bound avoids growth allocations, even for incompressible inputs.
/// The caller owns/reserves the input arrays (normally execution snapshots).
pub fn encode_npz(
    arrays: &[NamedArray],
    compression: NpzCompression,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<(Vec<u8>, Reservation)> {
    token.check()?;
    if arrays.len() >= 65535 {
        return Err(invalid());
    }
    let mut capacity = 22u64;
    let mut max_npy = 0u64;
    let mut name_bytes = 0u64;
    for (i, entry) in arrays.iter().enumerate() {
        token.check()?;
        if !name_valid(&entry.name)
            || arrays[..i].iter().any(|a| a.name == entry.name)
            || entry.array.shape.len() > 64
        {
            return Err(invalid());
        }
        let count = entry
            .array
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(invalid)?;
        if count != entry.array.buffer.len() {
            return Err(invalid());
        }
        let npy = checked_add(entry.array.buffer.byte_len(), 4096)?;
        max_npy = max_npy.max(npy);
        name_bytes = checked_add(name_bytes, entry.name.len() as u64)?;
        capacity = checked_add(
            capacity,
            checked_add(
                checked_mul(npy, 2)?,
                checked_add(4096, checked_mul(entry.name.len() as u64, 2)?)?,
            )?,
        )?;
    }
    if capacity >= u32::MAX as u64 {
        return Err(invalid());
    }
    let _metadata = budget.reserve(checked_add(
        checked_mul(arrays.len() as u64, MEMBER_METADATA)?,
        checked_mul(name_bytes, 8)?,
    )?)?;
    let _workspace = budget.reserve(checked_add(CODEC_WORKSPACE, max_npy)?)?;
    let encoded = budget.reserve(capacity)?;
    let mut bytes = vec![0u8; capacity as usize];
    let mut zip = ZipWriter::new(Cursor::new(bytes.as_mut_slice()));
    let method = match compression {
        NpzCompression::Stored => CompressionMethod::Stored,
        NpzCompression::Deflated => CompressionMethod::Deflated,
    };
    let options = SimpleFileOptions::default()
        .compression_method(method)
        .compression_level(if compression == NpzCompression::Deflated {
            Some(6)
        } else {
            None
        });
    for entry in arrays {
        token.check()?;
        let npy = encode_npy(&entry.array.shape, &entry.array.buffer)?;
        zip.start_file(format!("{}.npy", entry.name), options)
            .map_err(|_| invalid())?;
        for chunk in npy.chunks(CHUNK) {
            token.check()?;
            zip.write_all(chunk).map_err(io_failed)?;
        }
    }
    token.check()?;
    let used = zip.finish().map_err(|_| invalid())?.position() as usize;
    bytes.truncate(used);
    token.check()?;
    Ok((bytes, encoded))
}

pub fn save_npz(
    path: &Path,
    arrays: &[NamedArray],
    compression: NpzCompression,
    mode: WriteMode,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<()> {
    let (bytes, _encoded) = encode_npz(arrays, compression, budget, token)?;
    write_atomic(path, &bytes, mode, token)
}

#[derive(Debug)]
pub struct NpzArrays {
    pub arrays: Vec<(String, crate::array::NdArray)>,
    _metadata: Reservation,
}
impl NpzArchive {
    pub(crate) fn into_context(self, id: u64) -> NpzArrays {
        NpzArrays {
            arrays: self
                .arrays
                .into_iter()
                .map(|entry| {
                    (
                        entry.name,
                        crate::array::NdArray::from_dense(id, entry.array),
                    )
                })
                .collect(),
            _metadata: self._metadata,
        }
    }
}

#[cfg(test)]
#[path = "io_npz_tests.rs"]
mod tests;
