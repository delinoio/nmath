// SPDX-License-Identifier: Apache-2.0
//! Scheduled numeric/boolean Arrow IPC file and Parquet I/O.
use std::{
    fs::{self, File},
    io::{Cursor, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    basic::{Compression, Encoding},
    file::properties::WriterProperties,
};

use crate::{
    Buffer, CancellationToken, Context, DType, DenseArray, Error, ErrorCode, Expr, Result, Scalar,
    context::{Done, JobOutput, OperationKind, Work},
    io::WriteMode,
    memory::{Device, MemoryBudget, Reservation},
    table::{MaterializedColumn, SNAPSHOT_GATE, Table},
};
#[path = "io_columnar_preflight.rs"]
mod preflight;

#[derive(Clone, Copy, Debug)]
pub enum Format {
    ArrowIpc,
    Parquet,
}
/// Column names, physical dtypes and declared nullability are retained. Other
/// Arrow types must fail explicitly until the runtime can represent them.
pub struct FileTable {
    pub schema: SchemaRef,
    pub table: Table,
    _metadata: Reservation,
}
pub type ReadDone = Box<dyn FnOnce(Result<FileTable>) + Send>;
#[derive(Clone)]
pub struct InputColumn {
    pub name: String,
    pub values: Expr,
    pub valid: Option<Expr>,
}
fn format_error() -> Error {
    Error::new(
        ErrorCode::IoFormat,
        "Invalid or unsupported columnar file. Check the format, schema and encoding.",
    )
}
fn unsupported() -> Error {
    Error::new(
        ErrorCode::DTypeUnsupported,
        "This file schema cannot be represented losslessly. Use numeric or boolean columns \
         without custom metadata.",
    )
}
fn io_error(_: std::io::Error) -> Error {
    Error::new(
        ErrorCode::IoFailed,
        "A local file operation failed. Check permissions and the destination directory.",
    )
}
fn size_error() -> Error {
    Error::new(
        ErrorCode::MemoryBudgetExceeded,
        "Columnar workspace exceeds the memory budget. Reduce the input size.",
    )
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(size_error)
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(size_error)
}
fn dtype(t: &DataType) -> Result<DType> {
    Ok(match t {
        DataType::Boolean => DType::Bool,
        DataType::Int8 => DType::Int8,
        DataType::Int16 => DType::Int16,
        DataType::Int32 => DType::Int32,
        DataType::Int64 => DType::Int64,
        DataType::UInt8 => DType::UInt8,
        DataType::UInt16 => DType::UInt16,
        DataType::UInt32 => DType::UInt32,
        DataType::UInt64 => DType::UInt64,
        DataType::Float32 => DType::Float32,
        DataType::Float64 => DType::Float64,
        _ => return Err(unsupported()),
    })
}
fn arrow_dtype(t: DType) -> Result<DataType> {
    Ok(match t {
        DType::Bool => DataType::Boolean,
        DType::Int8 => DataType::Int8,
        DType::Int16 => DataType::Int16,
        DType::Int32 => DataType::Int32,
        DType::Int64 => DataType::Int64,
        DType::UInt8 => DataType::UInt8,
        DType::UInt16 => DataType::UInt16,
        DType::UInt32 => DataType::UInt32,
        DType::UInt64 => DataType::UInt64,
        DType::Float32 => DataType::Float32,
        DType::Float64 => DataType::Float64,
        _ => return Err(unsupported()),
    })
}
fn check_schema(schema: &Schema) -> Result<()> {
    if !schema.metadata().is_empty() {
        return Err(unsupported());
    }
    for (i, f) in schema.fields().iter().enumerate() {
        dtype(f.data_type())?;
        if !f.metadata().is_empty()
            || schema.fields()[..i]
                .iter()
                .any(|old| old.name() == f.name())
        {
            return Err(unsupported());
        }
    }
    Ok(())
}

fn check_ipc_schema(schema: arrow_ipc::Schema<'_>) -> Result<()> {
    if schema.custom_metadata().is_some_and(|m| !m.is_empty()) {
        return Err(unsupported());
    }
    if let Some(fields) = schema.fields() {
        // FlatBuffers may reuse a string offset many times. Bound expanded
        // schema storage independently of the serialized file length.
        if fields.len() > 1024 {
            return Err(unsupported());
        }
        for field in fields {
            if field.name().is_some_and(|n| n.len() > 4096)
                || field.children().is_some_and(|c| !c.is_empty())
                || field.dictionary().is_some()
                || field.custom_metadata().is_some_and(|m| !m.is_empty())
                || !matches!(
                    field.type_type(),
                    arrow_ipc::Type::Int | arrow_ipc::Type::FloatingPoint | arrow_ipc::Type::Bool
                )
            {
                return Err(unsupported());
            }
        }
    }

    Ok(())
}

/// Validate IPC blocks and count rows before the Arrow reader allocates
/// buffers.
fn ipc_rows(bytes: &[u8]) -> Result<usize> {
    if bytes.len() < 10 || !bytes.starts_with(b"ARROW1") || !bytes.ends_with(b"ARROW1") {
        return Err(format_error());
    }
    let end = bytes.len() - 10;
    let len = u32::from_le_bytes(bytes[end..end + 4].try_into().unwrap()) as usize;
    let start = end.checked_sub(len).ok_or_else(format_error)?;
    let footer = arrow_ipc::root_as_footer(&bytes[start..end]).map_err(|_| format_error())?;
    if footer.dictionaries().is_some_and(|d| !d.is_empty()) {
        return Err(unsupported());
    }
    if footer.custom_metadata().is_some_and(|m| !m.is_empty()) {
        return Err(unsupported());
    }
    check_ipc_schema(footer.schema().ok_or_else(format_error)?)?;
    let mut rows = 0usize;
    if let Some(blocks) = footer.recordBatches() {
        for block in blocks {
            let offset = usize::try_from(block.offset()).map_err(|_| format_error())?;
            let meta = usize::try_from(block.metaDataLength()).map_err(|_| format_error())?;
            let body = usize::try_from(block.bodyLength()).map_err(|_| format_error())?;
            let finish = offset
                .checked_add(meta)
                .and_then(|n| n.checked_add(body))
                .filter(|&n| n <= start)
                .ok_or_else(format_error)?;
            let data = bytes.get(offset..finish).ok_or_else(format_error)?;
            let prefix = if data.starts_with(&[255; 4]) { 8 } else { 4 };
            let msg = arrow_ipc::root_as_message(data.get(prefix..meta).ok_or_else(format_error)?)
                .map_err(|_| format_error())?;
            if usize::try_from(msg.bodyLength()).ok() != Some(body) {
                return Err(format_error());
            }
            let batch = msg.header_as_record_batch().ok_or_else(format_error)?;
            // Compression requires an independently bounded decompressor.
            // Reject it until that preflight exists; do not trust
            // embedded output sizes.
            if batch.compression().is_some() {
                return Err(format_error());
            }
            let count = usize::try_from(batch.length()).map_err(|_| format_error())?;
            rows = rows.checked_add(count).ok_or_else(size_error)?;
            if let Some(nodes) = batch.nodes() {
                for node in nodes {
                    if node.length() != batch.length()
                        || node.null_count() < 0
                        || node.null_count() > node.length()
                    {
                        return Err(format_error());
                    }
                }
            }
            if let Some(buffers) = batch.buffers() {
                for b in buffers {
                    let p = u64::try_from(b.offset()).map_err(|_| format_error())?;
                    let n = u64::try_from(b.length()).map_err(|_| format_error())?;
                    if add(p, n)? > body as u64 {
                        return Err(format_error());
                    }
                }
            }
        }
    }
    Ok(rows)
}
fn parquet_footer(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 12 || !bytes.starts_with(b"PAR1") || !bytes.ends_with(b"PAR1") {
        return Err(format_error());
    }
    let end = bytes.len() - 8;
    let len = u32::from_le_bytes(bytes[end..end + 4].try_into().unwrap()) as usize;
    let start = end
        .checked_sub(len)
        .filter(|&n| n >= 4)
        .ok_or_else(format_error)?;
    let mut scan = preflight::Compact::new(&bytes[start..end]);
    scan.structure(0, false)?;
    if scan.pos != len {
        return Err(format_error());
    }
    Ok(())
}

pub fn read(
    path: &Path,
    format: Format,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<FileTable> {
    token.check()?;
    let mut file = File::open(path).map_err(io_error)?;
    let before = file.metadata().map_err(io_error)?;
    let len = usize::try_from(before.len()).map_err(|_| size_error())?;
    let _input = budget.reserve(before.len())?;
    let mut bytes = vec![0; len];
    for chunk in bytes.chunks_mut(64 * 1024) {
        token.check()?;
        file.read_exact(chunk).map_err(io_error)?;
    }
    // Conservative metadata/codec bound for the pinned Arrow/Parquet versions.
    // Preflight constrains nested lengths before any library parse. Numeric
    // pages additionally reserve declared decoded sizes and value slots below.
    let metadata = budget.reserve(add(mul(before.len(), 64)?, 8 * 1024 * 1024)?)?;
    let result = match format {
        Format::ArrowIpc => {
            let rows = ipc_rows(&bytes)?;
            let reader = arrow_ipc::reader::FileReader::try_new(Cursor::new(bytes), None)
                .map_err(|_| format_error())?;
            let schema = reader.schema();
            check_schema(&schema)?;
            let _workspace =
                budget.reserve(mul(mul(rows as u64, schema.fields().len() as u64)?, 32)?)?;
            collect(reader, schema, rows, budget, token, metadata)?
        }
        Format::Parquet => {
            parquet_footer(&bytes)?;
            let data = bytes::Bytes::from(bytes);
            // Parquet can embed an Arrow FlatBuffer. Validate its expanded
            // schema before the library copies referenced strings and fields.
            let meta = parquet::file::metadata::ParquetMetaDataReader::new()
                .parse_and_finish(&data)
                .map_err(|_| format_error())?;
            let mut embedded = None;
            if let Some(entries) = meta.file_metadata().key_value_metadata() {
                for entry in entries {
                    if entry.key != "ARROW:schema" || embedded.is_some() {
                        return Err(unsupported());
                    }
                    use base64::Engine;
                    let encoded = entry.value.as_ref().ok_or_else(format_error)?;
                    let decoded = base64::prelude::BASE64_STANDARD
                        .decode(encoded)
                        .map_err(|_| format_error())?;
                    let message_bytes = if decoded.starts_with(&[255; 4]) {
                        decoded.get(8..).ok_or_else(format_error)?
                    } else {
                        &decoded
                    };
                    let message =
                        arrow_ipc::root_as_message(message_bytes).map_err(|_| format_error())?;
                    let raw = message.header_as_schema().ok_or_else(format_error)?;
                    check_ipc_schema(raw)?;
                    embedded = Some(
                        arrow_ipc::convert::try_fb_to_schema(raw).map_err(|_| format_error())?,
                    );
                }
            }

            let builder = ParquetRecordBatchReaderBuilder::try_new(data.clone())
                .map_err(|_| format_error())?;
            let schema = builder.schema().clone();
            if embedded.as_ref().is_some_and(|s| s != schema.as_ref()) {
                return Err(unsupported());
            }

            check_schema(&schema)?;
            let rows = usize::try_from(builder.metadata().file_metadata().num_rows())
                .map_err(|_| format_error())?;
            let mut workspace = 0u64;
            for group in builder.metadata().row_groups() {
                for col in group.columns() {
                    if !matches!(
                        col.compression(),
                        Compression::UNCOMPRESSED | Compression::SNAPPY
                    ) {
                        return Err(format_error());
                    }
                    let (offset, length) = col.byte_range();
                    let end = add(offset, length)?;
                    let mut p = usize::try_from(offset).map_err(|_| format_error())?;
                    let end = usize::try_from(end).map_err(|_| format_error())?;
                    let chunk = data.get(..end).ok_or_else(format_error)?;
                    while p < end {
                        token.check()?;
                        let mut scan = preflight::Compact::new(&chunk[p..]);
                        scan.structure(0, true)?;
                        workspace = add(
                            workspace,
                            add(mul(scan.unpacked, 4)?, mul(scan.values, 32)?)?,
                        )?;
                        p = p
                            .checked_add(scan.pos)
                            .and_then(|n| n.checked_add(usize::try_from(scan.packed).ok()?))
                            .filter(|&n| n <= end)
                            .ok_or_else(format_error)?;
                    }
                }
            }
            workspace = add(
                workspace,
                mul(mul(rows as u64, schema.fields().len() as u64)?, 32)?,
            )?;
            let _workspace = budget.reserve(workspace)?;
            let reader = builder
                .with_batch_size(4096)
                .build()
                .map_err(|_| format_error())?;
            collect(reader, schema, rows, budget, token, metadata)?
        }
    };
    token.check()?;
    check_unchanged(path, &file, &before)?;
    Ok(result)
}
fn check_unchanged(path: &Path, file: &File, before: &fs::Metadata) -> Result<()> {
    let stamp = |m: &fs::Metadata| (m.len(), m.modified().ok());
    if stamp(before) != stamp(&file.metadata().map_err(io_error)?)
        || stamp(before) != stamp(&fs::metadata(path).map_err(io_error)?)
    {
        return Err(Error::new(
            ErrorCode::IoModified,
            "The input file changed during decoding. Retry with a stable file.",
        ));
    }
    Ok(())
}
fn allocate(dtype: DType, rows: usize, budget: &MemoryBudget) -> Result<DenseArray> {
    let charge = budget.reserve(mul(rows as u64, dtype.item_size() as u64)?)?;
    DenseArray::new(
        vec![rows],
        Buffer::filled(dtype, rows, Scalar::Int(0))?,
        charge,
    )
}
fn collect(
    reader: impl Iterator<Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>>,
    schema: SchemaRef,
    rows: usize,
    budget: &MemoryBudget,
    token: &CancellationToken,
    metadata: Reservation,
) -> Result<FileTable> {
    let mut columns = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        columns.push(MaterializedColumn {
            values: allocate(dtype(field.data_type())?, rows, budget)?,
            valid: allocate(DType::Bool, rows, budget)?,
        });
    }
    let mut offset = 0usize;
    for batch in reader {
        token.check()?;
        let batch = batch.map_err(|_| format_error())?;
        let end = offset
            .checked_add(batch.num_rows())
            .filter(|&n| n <= rows)
            .ok_or_else(format_error)?;
        for (source, dest) in batch.columns().iter().zip(&mut columns) {
            macro_rules! copy {
                ($variant:ident, $array:ty) => {
                    if let Buffer::$variant(v) = &mut dest.values.buffer {
                        let a = source
                            .as_any()
                            .downcast_ref::<$array>()
                            .ok_or_else(format_error)?;
                        for i in 0..a.len() {
                            if i % 1024 == 0 {
                                token.check()?;
                            }
                            v[offset + i] = a.value(i);
                        }
                    }
                };
            }
            copy!(Bool, arrow_array::BooleanArray);
            copy!(I8, arrow_array::Int8Array);
            copy!(I16, arrow_array::Int16Array);
            copy!(I32, arrow_array::Int32Array);
            copy!(I64, arrow_array::Int64Array);
            copy!(U8, arrow_array::UInt8Array);
            copy!(U16, arrow_array::UInt16Array);
            copy!(U32, arrow_array::UInt32Array);
            copy!(U64, arrow_array::UInt64Array);
            copy!(F32, arrow_array::Float32Array);
            copy!(F64, arrow_array::Float64Array);
            if let Buffer::Bool(v) = &mut dest.valid.buffer {
                for i in 0..source.len() {
                    v[offset + i] = source.is_valid(i);
                }
            }
        }
        offset = end;
    }
    if offset != rows {
        return Err(format_error());
    }
    Ok(FileTable {
        schema,
        table: Table { columns, rows },
        _metadata: metadata,
    })
}

fn batch(
    schema: SchemaRef,
    table: &Table,
    start: usize,
    count: usize,
    token: &CancellationToken,
) -> Result<RecordBatch> {
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(table.columns.len());
    for column in &table.columns {
        token.check()?;
        let valid = match &column.valid.buffer {
            Buffer::Bool(v) => v,
            _ => return Err(format_error()),
        };
        macro_rules! array {
            ($v:expr, $t:ty) => {
                Arc::new(<$t>::from(
                    (start..start + count)
                        .map(|i| valid[i].then_some($v[i]))
                        .collect::<Vec<_>>(),
                )) as ArrayRef
            };
        }
        arrays.push(match &column.values.buffer {
            Buffer::Bool(v) => array!(v, arrow_array::BooleanArray),
            Buffer::I8(v) => array!(v, arrow_array::Int8Array),
            Buffer::I16(v) => array!(v, arrow_array::Int16Array),
            Buffer::I32(v) => array!(v, arrow_array::Int32Array),
            Buffer::I64(v) => array!(v, arrow_array::Int64Array),
            Buffer::U8(v) => array!(v, arrow_array::UInt8Array),
            Buffer::U16(v) => array!(v, arrow_array::UInt16Array),
            Buffer::U32(v) => array!(v, arrow_array::UInt32Array),
            Buffer::U64(v) => array!(v, arrow_array::UInt64Array),
            Buffer::F32(v) => array!(v, arrow_array::Float32Array),
            Buffer::F64(v) => array!(v, arrow_array::Float64Array),
            _ => return Err(unsupported()),
        });
    }
    RecordBatch::try_new_with_options(
        schema,
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(count)),
    )
    .map_err(|_| format_error())
}

/// Streaming serialization into a sibling temporary file. No output-file-sized
/// buffer is constructed, and only publication can turn this into success.
pub fn write(
    path: &Path,
    format: Format,
    schema: SchemaRef,
    table: &Table,
    mode: WriteMode,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<()> {
    token.check()?;
    check_schema(&schema)?;
    if schema.fields().len() != table.columns.len() {
        return Err(format_error());
    }
    for (f, c) in schema.fields().iter().zip(&table.columns) {
        if dtype(f.data_type())? != c.values.dtype()
            || c.values.shape != [table.rows]
            || c.valid.shape != [table.rows]
            || c.valid.dtype() != DType::Bool
        {
            return Err(format_error());
        }
        if !f.is_nullable() && (0..table.rows).any(|i| c.valid.buffer.get(i) != Scalar::Bool(true))
        {
            return Err(format_error());
        }
    }
    // Dictionary encoding is disabled and row groups are flushed per 4096-row
    // batch. Reserve input conversion, page/encoder scratch and file metadata
    // before entering the codec. The bound includes Arrow's minimum capacities.
    let cells = mul(table.columns.len() as u64, table.rows.min(4096) as u64)?;
    let groups = (table.rows as u64 / 4096) + 1;
    let names = schema
        .fields()
        .iter()
        .try_fold(0u64, |n, f| add(n, f.name().len() as u64))?;
    let workspace = add(
        add(
            mul(cells, 256)?,
            mul(table.columns.len() as u64 + 1, 1024 * 1024)?,
        )?,
        add(
            mul(
                groups,
                add(names, mul(table.columns.len() as u64 + 1, 4096)?)?,
            )?,
            mul(path.as_os_str().len() as u64 + 256, 8)?,
        )?,
    )?;
    let _workspace = budget.reserve(workspace)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| format_error())?
        .as_nanos();
    let tmp = parent.join(format!(
        ".nmath-columnar-{}-{time}-{nonce}.tmp",
        std::process::id()
    ));
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(io_error)?;
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    let cleanup = Cleanup(tmp);
    match format {
        Format::ArrowIpc => {
            let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut file, &schema)
                .map_err(|_| format_error())?;
            for start in (0..table.rows).step_by(4096) {
                token.check()?;
                writer
                    .write(&batch(
                        schema.clone(),
                        table,
                        start,
                        (table.rows - start).min(4096),
                        token,
                    )?)
                    .map_err(|_| format_error())?;
            }
            writer.finish().map_err(|_| format_error())?;
        }
        Format::Parquet => {
            // Snappy has bounded workspace and broad interoperability. Other
            // codecs need separate workspace preflight before enabling them.
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .set_dictionary_enabled(false)
                .set_encoding(Encoding::PLAIN)
                .set_max_row_group_row_count(Some(4096))
                .build();
            let mut writer = ArrowWriter::try_new(&mut file, schema.clone(), Some(props))
                .map_err(|_| format_error())?;
            for start in (0..table.rows).step_by(4096) {
                token.check()?;
                writer
                    .write(&batch(
                        schema.clone(),
                        table,
                        start,
                        (table.rows - start).min(4096),
                        token,
                    )?)
                    .map_err(|_| format_error())?;
                writer.flush().map_err(|_| format_error())?;
            }
            writer.close().map_err(|_| format_error())?;
        }
    }
    file.flush().map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    drop(file);
    token.check()?;
    match mode {
        WriteMode::Overwrite => fs::rename(&cleanup.0, path).map_err(io_error)?,
        WriteMode::CreateNew => fs::hard_link(&cleanup.0, path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                Error::new(
                    ErrorCode::IoExists,
                    "The destination exists. Set overwrite explicitly to replace it.",
                )
            } else {
                io_error(e)
            }
        })?,
    }
    Ok(())
}

pub fn schedule_read(
    context: &Context,
    path: PathBuf,
    format: Format,
    token: CancellationToken,
    done: ReadDone,
) -> Result<u64> {
    let result = Arc::new(Mutex::new(None));
    let output = result.clone();
    let work: Work = Box::new(move |env| {
        *output.lock().unwrap() = Some(read(&path, format, &env.budget, &env.token)?);
        Ok(JobOutput::Unit)
    });
    let finish: Done = Box::new(move |status| {
        let value = result.lock().unwrap().take();
        done(status.and_then(|_| value.ok_or_else(format_error)));
    });
    context.submit(
        OperationKind::FileRead,
        "read_columnar",
        Device::Cpu,
        token,
        work,
        finish,
    )
}
pub fn schedule_write(
    context: &Context,
    columns: Vec<InputColumn>,
    path: PathBuf,
    format: Format,
    mode: WriteMode,
    token: CancellationToken,
    done: Done,
) -> Result<u64> {
    let mut rows = None;
    for (i, c) in columns.iter().enumerate() {
        arrow_dtype(c.values.dtype())?;
        if columns[..i].iter().any(|old| old.name == c.name) {
            return Err(unsupported());
        }
        if c.values.shape().len() != 1 || rows.is_some_and(|n| c.values.shape()[0] != n) {
            return Err(format_error());
        }
        rows = Some(c.values.shape()[0]);
        for e in std::iter::once(&c.values).chain(c.valid.iter()) {
            if e.context_id().is_some_and(|id| id != context.id()) {
                return Err(Error::new(
                    ErrorCode::ContextMismatch,
                    "File columns must use the execution context. Copy them explicitly.",
                ));
            }
            if !e.generators().is_empty() {
                return Err(unsupported());
            }
        }
        if c.valid
            .as_ref()
            .is_some_and(|v| v.dtype() != DType::Bool || v.shape() != c.values.shape())
        {
            return Err(format_error());
        }
    }
    let work: Work = Box::new(move |env| {
        let names = columns
            .iter()
            .try_fold(0u64, |n, c| add(n, c.name.len() as u64))?;
        let _metadata = env
            .budget
            .reserve(add(names, mul(columns.len() as u64 + 1, 4096)?)?)?;
        let rows = rows.unwrap_or(0);
        {
            let _gate = SNAPSHOT_GATE.lock().unwrap();
            let largest = columns
                .iter()
                .flat_map(|c| std::iter::once(&c.values).chain(c.valid.iter()))
                .flat_map(Expr::inputs)
                .map(|a| a.size())
                .max()
                .unwrap_or(0);
            let _offsets = env
                .budget
                .reserve(mul(largest as u64, std::mem::size_of::<usize>() as u64)?)?;
            for c in &columns {
                env.token.check()?;
                env.freeze(&c.values)?;
                if let Some(v) = &c.valid {
                    env.freeze(v)?;
                }
            }
        }
        let mut fields = Vec::with_capacity(columns.len());
        let mut data = Vec::with_capacity(columns.len());
        for c in columns {
            env.token.check()?;
            fields.push(Field::new(
                c.name,
                arrow_dtype(c.values.dtype())?,
                c.valid.is_some(),
            ));
            let values = env.eval_dense(&c.values)?;
            let valid = if let Some(v) = c.valid {
                env.eval_dense(&v)?
            } else {
                let charge = env.budget.reserve(rows as u64)?;
                DenseArray::new(vec![rows], Buffer::Bool(vec![true; rows]), charge)?
            };
            data.push(MaterializedColumn { values, valid });
        }
        write(
            &path,
            format,
            Arc::new(Schema::new(fields)),
            &Table {
                columns: data,
                rows,
            },
            mode,
            &env.budget,
            &env.token,
        )?;
        Ok(JobOutput::Unit)
    });
    context.submit(
        OperationKind::FileWrite,
        "write_columnar",
        Device::Cpu,
        token,
        work,
        done,
    )
}

#[cfg(test)]
#[path = "io_columnar_tests.rs"]
mod tests;
