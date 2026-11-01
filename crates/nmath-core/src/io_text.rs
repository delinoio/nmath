// SPDX-License-Identifier: Apache-2.0
//! Local numeric CSV and JSON records. Parsing and publication run in scheduled
//! jobs.
use std::{
    fs::File,
    io::Read,
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{
    Buffer, CancellationToken, Context, DType, DenseArray, Error, ErrorCode, Expr, Result, Scalar,
    context::{Done, JobOutput, OperationKind, Work},
    io::{WriteMode, write_atomic},
    memory::{Device, MemoryBudget, Reservation},
    table::{MaterializedColumn, Table},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Csv,
    Json,
}
pub struct NamedTable {
    pub names: Vec<String>,
    pub table: Table,
    pub metadata: Reservation,
}
pub type ReadDone = Box<dyn FnOnce(Result<NamedTable>) + Send>;
pub type Schema = Vec<(String, DType)>;
fn invalid() -> Error {
    Error::new(
        ErrorCode::IoFormat,
        "Invalid or unsupported numeric table file. Check the format and schema.",
    )
}
fn failed(_: std::io::Error) -> Error {
    Error::new(
        ErrorCode::IoFailed,
        "The table file operation failed. Check local file permissions.",
    )
}
fn reserve(b: &MemoryBudget, n: usize, width: usize) -> Result<Reservation> {
    b.reserve(
        n.checked_mul(width)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MemoryBudgetExceeded,
                    "Table workspace size overflows.",
                )
            })?,
    )
}
fn names_valid(names: &[String]) -> Result<()> {
    if names
        .iter()
        .enumerate()
        .any(|(i, n)| names[..i].contains(n))
    {
        return Err(invalid());
    }
    Ok(())
}
// Strict CSV state machine: quoted delimiters/newlines and doubled quotes are
// supported. Reject broken quoting instead of silently consuming the remainder
// of a damaged file.
fn csv_rows(bytes: &[u8], token: &CancellationToken) -> Result<Vec<Vec<String>>> {
    std::str::from_utf8(bytes).map_err(|_| invalid())?;
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = Vec::new();
    let (mut quoted, mut closed, mut started) = (false, false, false);
    let mut i = 0;
    while i < bytes.len() {
        if i % 4096 == 0 {
            token.check()?;
        }
        let c = bytes[i];
        if quoted {
            if c == b'"' {
                if bytes.get(i + 1) == Some(&b'"') {
                    field.push(c);
                    i += 1;
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field.push(c);
            }
        } else if c == b',' || c == b'\n' || c == b'\r' {
            row.push(String::from_utf8(std::mem::take(&mut field)).map_err(|_| invalid())?);
            closed = false;
            started = false;
            if c != b',' {
                rows.push(std::mem::take(&mut row));
                if c == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
            }
        } else if c == b'"' && !started && !closed {
            quoted = true;
            started = true;
        } else {
            if closed || c == b'"' {
                return Err(invalid());
            }
            field.push(c);
            started = true;
        }
        i += 1;
    }
    if quoted {
        return Err(invalid());
    }
    if started || closed || !field.is_empty() || !row.is_empty() {
        row.push(String::from_utf8(field).map_err(|_| invalid())?);
        rows.push(row);
    }
    Ok(rows)
}
// Deserialize objects as ordered pairs so duplicate keys cannot silently
// overwrite data.
struct Record(Vec<(String, serde_json::Value)>);
impl<'de> serde::Deserialize<'de> for Record {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Record;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a numeric record")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Record, M::Error> {
                let mut pairs = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, serde_json::Value>()? {
                    if pairs.iter().any(|(name, _)| name == &k) {
                        return Err(serde::de::Error::custom("duplicate column"));
                    }
                    pairs.push((k, v));
                }
                Ok(Record(pairs))
            }
        }
        d.deserialize_map(Visitor)
    }
}
fn parse(
    bytes: &[u8],
    format: Format,
    schema: &Schema,
    token: &CancellationToken,
) -> Result<(Vec<String>, Vec<Vec<Option<String>>>)> {
    match format {
        Format::Csv => {
            let mut rows = csv_rows(bytes, token)?.into_iter();
            let names = rows.next().ok_or_else(invalid)?;
            names_valid(&names)?;
            let rows = rows
                .map(|r| {
                    if r.len() != names.len() {
                        return Err(invalid());
                    }
                    Ok(r.into_iter()
                        .map(|v| if v.is_empty() { None } else { Some(v) })
                        .collect())
                })
                .collect::<Result<_>>()?;
            Ok((names, rows))
        }
        Format::Json => {
            let records: Vec<Record> = serde_json::from_slice(bytes).map_err(|_| invalid())?;
            let names: Vec<String> = records
                .first()
                .map(|r| r.0.iter().map(|(n, _)| n.clone()).collect())
                .unwrap_or_else(|| schema.iter().map(|(n, _)| n.clone()).collect());
            if names.is_empty() && !records.is_empty() {
                return Err(invalid());
            }
            let mut rows = Vec::with_capacity(records.len());
            for Record(record) in records {
                token.check()?;
                if record.len() != names.len() || record.iter().any(|(n, _)| !names.contains(n)) {
                    return Err(invalid());
                }
                let mut row = Vec::with_capacity(names.len());
                for name in &names {
                    let v = &record
                        .iter()
                        .find(|(n, _)| n == name)
                        .ok_or_else(invalid)?
                        .1;
                    row.push(match v {
                        serde_json::Value::Null => None,
                        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
                            Some(v.to_string())
                        }
                        _ => return Err(invalid()),
                    });
                }
                rows.push(row);
            }
            Ok((names, rows))
        }
    }
}
fn infer<'a>(values: impl Iterator<Item = &'a Option<String>>) -> Result<DType> {
    let mut kind = None;
    for s in values.flatten() {
        let next = if s == "true" || s == "false" || s == "True" || s == "False" {
            DType::Bool
        } else if s.parse::<i64>().is_ok() {
            DType::Int64
        } else if s.parse::<u64>().is_ok() {
            DType::UInt64
        } else if (s.contains(['.', 'e', 'E'])
            || matches!(
                s.as_str(),
                "NaN" | "nan" | "inf" | "-inf" | "Infinity" | "-Infinity"
            ))
            && s.parse::<f64>().is_ok()
        {
            DType::Float64
        } else {
            return Err(invalid());
        };
        kind = Some(match (kind, next) {
            (None, n) => n,
            (Some(a), b) if a == b => a,
            (Some(DType::Int64 | DType::UInt64), DType::Int64 | DType::UInt64) => DType::UInt64,
            (Some(DType::Bool), _) | (_, DType::Bool) => return Err(invalid()),
            _ => DType::Float64,
        });
    }
    Ok(kind.unwrap_or(DType::Float64))
}
fn column(
    rows: &[Vec<Option<String>>],
    i: usize,
    dtype: DType,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<MaterializedColumn> {
    let charge = reserve(budget, rows.len(), dtype.item_size())?;
    let mask_charge = reserve(budget, rows.len(), 1)?;
    let mut valid = Vec::with_capacity(rows.len());
    macro_rules! numbers {
        ($ty:ty, $variant:ident) => {{
            let mut out = Vec::with_capacity(rows.len());
            for (j, row) in rows.iter().enumerate() {
                if j % 4096 == 0 {
                    token.check()?;
                }
                valid.push(row[i].is_some());
                out.push(match &row[i] {
                    None => 0 as $ty,
                    Some(s) => s.parse::<$ty>().map_err(|_| invalid())?,
                });
            }
            Buffer::$variant(out)
        }};
    }
    let buffer = match dtype {
        DType::Int8 => numbers!(i8, I8),
        DType::Int16 => numbers!(i16, I16),
        DType::Int32 => numbers!(i32, I32),
        DType::Int64 => numbers!(i64, I64),
        DType::UInt8 => numbers!(u8, U8),
        DType::UInt16 => numbers!(u16, U16),
        DType::UInt32 => numbers!(u32, U32),
        DType::UInt64 => numbers!(u64, U64),
        DType::Float32 => numbers!(f32, F32),
        DType::Float64 => numbers!(f64, F64),
        DType::Bool => {
            let mut out = Vec::with_capacity(rows.len());
            for (j, row) in rows.iter().enumerate() {
                if j % 4096 == 0 {
                    token.check()?;
                }
                valid.push(row[i].is_some());
                out.push(match row[i].as_deref() {
                    None | Some("false" | "False") => false,
                    Some("true" | "True") => true,
                    _ => return Err(invalid()),
                });
            }
            Buffer::Bool(out)
        }
        _ => {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Text tables require real numeric or boolean columns.",
            ));
        }
    };
    Ok(MaterializedColumn {
        values: DenseArray::new(vec![rows.len()], buffer, charge)?,
        valid: DenseArray::new(vec![rows.len()], Buffer::Bool(valid), mask_charge)?,
    })
}
pub fn read(
    path: &Path,
    format: Format,
    schema: &Schema,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<NamedTable> {
    read_with(path, format, schema, budget, token, || {})
}
fn read_with(
    path: &Path,
    format: Format,
    schema: &Schema,
    budget: &MemoryBudget,
    token: &CancellationToken,
    after_read: impl FnOnce(),
) -> Result<NamedTable> {
    token.check()?;
    let mut file = File::open(path).map_err(failed)?;
    let stamp = |m: std::fs::Metadata| (m.len(), m.modified().ok());
    let before = stamp(file.metadata().map_err(failed)?);
    let len = usize::try_from(before.0).map_err(|_| invalid())?;
    // Covers the input, parser trees, cell strings, row vectors and codec
    // scratch. 512 bytes per source byte bounds even one-character cells
    // and deep JSON containers (serde_json's default recursion limit stays
    // enabled).
    let _codec = budget.reserve(65536)?;
    let _workspace = reserve(budget, len, 512)?;
    let mut bytes = vec![0; len];
    for chunk in bytes.chunks_mut(65536) {
        token.check()?;
        file.read_exact(chunk).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::new(ErrorCode::IoModified, "The input file changed.")
            } else {
                failed(e)
            }
        })?;
    }
    let mut extra = [0u8; 1];
    if file.read(&mut extra).map_err(failed)? != 0 {
        return Err(Error::new(ErrorCode::IoModified, "The input file changed."));
    }
    after_read();
    token.check()?;
    let (names, rows) = parse(&bytes, format, schema, token)?;
    if !schema.is_empty()
        && (schema.len() != names.len() || schema.iter().zip(&names).any(|((a, _), b)| a != b))
    {
        return Err(invalid());
    }
    names_valid(&names)?;
    let metadata = reserve(
        budget,
        names
            .iter()
            .map(String::len)
            .sum::<usize>()
            .checked_add(names.len().saturating_mul(256))
            .ok_or_else(invalid)?,
        1,
    )?;
    let mut columns = Vec::with_capacity(names.len());
    for i in 0..names.len() {
        token.check()?;
        let dtype = if schema.is_empty() {
            infer(rows.iter().map(|r| &r[i]))?
        } else {
            schema[i].1
        };
        columns.push(column(&rows, i, dtype, budget, token)?);
    }
    if before != stamp(file.metadata().map_err(failed)?)
        || before != stamp(std::fs::metadata(path).map_err(failed)?)
    {
        return Err(Error::new(ErrorCode::IoModified, "The input file changed."));
    }
    token.check()?;
    Ok(NamedTable {
        names,
        table: Table {
            columns,
            rows: rows.len(),
            pivot_columns: None,
        },
        metadata,
    })
}
fn csv_quote(s: &str) -> String {
    if s.contains([',', '"', '\r', '\n']) || s.is_empty() {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}
pub fn write(
    path: &Path,
    format: Format,
    names: &[String],
    table: &Table,
    mode: WriteMode,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<()> {
    names_valid(names)?;
    if names.is_empty() && (format == Format::Csv || table.rows != 0) {
        return Err(invalid());
    }
    if table.columns.iter().any(|c| {
        matches!(
            c.values.dtype(),
            DType::Complex64
                | DType::Complex128
                | DType::Str(_)
                | DType::Datetime64(_)
                | DType::Timedelta64(_)
        )
    }) {
        return Err(Error::new(
            ErrorCode::DTypeUnsupported,
            "Text tables require real numeric or boolean columns.",
        ));
    }
    if names.len() != table.columns.len()
        || table.columns.iter().any(|c| {
            c.values.shape != [table.rows]
                || c.valid.shape != [table.rows]
                || c.valid.dtype() != DType::Bool
        })
    {
        return Err(invalid());
    }
    let name_bytes = names
        .iter()
        .try_fold(0usize, |n, s| n.checked_add(s.len().saturating_mul(6) + 8))
        .ok_or_else(invalid)?;
    let row_bytes = name_bytes
        .checked_add(names.len().checked_mul(768).ok_or_else(invalid)?)
        .and_then(|n| n.checked_add(4))
        .ok_or_else(invalid)?;
    let capacity = row_bytes
        .checked_mul(table.rows.checked_add(1).ok_or_else(invalid)?)
        .and_then(|n| n.checked_add(64))
        .ok_or_else(invalid)?;
    let _workspace = reserve(budget, capacity, 2)?;
    let _path = reserve(budget, path.as_os_str().len().saturating_add(4096), 4)?;
    let mut out = String::with_capacity(capacity);
    if format == Format::Csv {
        for (i, n) in names.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&csv_quote(n));
        }
        out.push('\n');
    } else {
        out.push('[');
    }
    for r in 0..table.rows {
        token.check()?;
        if format == Format::Json {
            if r > 0 {
                out.push(',');
            }
            out.push('{');
        }
        for (i, c) in table.columns.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            if format == Format::Json {
                out.push_str(&serde_json::to_string(&names[i]).map_err(|_| invalid())?);
                out.push(':');
            }
            if c.valid.buffer.get(r) == Scalar::Bool(false) {
                if format == Format::Json {
                    out.push_str("null");
                }
                continue;
            }
            let value = match c.values.buffer.get(r) {
                Scalar::Int(v) => v.to_string(),
                Scalar::UInt(v) => v.to_string(),
                Scalar::Bool(v) => v.to_string(),
                Scalar::Float(v) => {
                    if format == Format::Json && !v.is_finite() {
                        "null".to_owned()
                    } else {
                        v.to_string()
                    }
                }
                _ => {
                    return Err(Error::new(
                        ErrorCode::DTypeUnsupported,
                        "Text tables require real numeric or boolean columns.",
                    ));
                }
            };
            out.push_str(&value);
        }
        if format == Format::Json {
            out.push('}');
        } else {
            out.push('\n');
        }
    }
    if format == Format::Json {
        out.push(']');
    }
    token.check()?;
    // Publication is the last fallible step. A successful write is terminal.
    write_atomic(path, out.as_bytes(), mode, token)
}
impl Context {
    pub fn read_text(
        &self,
        path: std::path::PathBuf,
        format: Format,
        schema: Schema,
        token: CancellationToken,
        done: ReadDone,
    ) -> Result<u64> {
        let result = Arc::new(Mutex::new(None));
        let output = result.clone();
        let work: Work = Box::new(move |env| {
            *output.lock().unwrap() = Some(read(&path, format, &schema, &env.budget, &env.token)?);
            Ok(JobOutput::Unit)
        });
        let finish: Done = Box::new(move |status| {
            done(status.and_then(|_| result.lock().unwrap().take().ok_or_else(invalid)))
        });
        self.submit(
            OperationKind::FileRead,
            "read_text",
            Device::Cpu,
            token,
            work,
            finish,
        )
    }

    pub fn write_text(
        &self,
        path: std::path::PathBuf,
        format: Format,
        columns: Vec<(String, Expr, Option<Expr>)>,
        mode: WriteMode,
        token: CancellationToken,
        done: Done,
    ) -> Result<u64> {
        if columns.iter().any(|(_, v, m)| {
            v.context_id().is_some_and(|id| id != self.id())
                || m.as_ref()
                    .and_then(Expr::context_id)
                    .is_some_and(|id| id != self.id())
        }) {
            return Err(Error::new(
                ErrorCode::ContextMismatch,
                "Text table columns must belong to the writing context.",
            ));
        }
        let work: Work = Box::new(move |env| {
            let _metadata = reserve(
                &env.budget,
                columns.iter().map(|(n, _, _)| n.len() + 512).sum(),
                1,
            )?;
            let mut names = Vec::with_capacity(columns.len());
            let mut materialized = Vec::with_capacity(columns.len());
            let rows = columns
                .first()
                .map(|(_, v, _)| v.shape().first().copied().unwrap_or(0))
                .unwrap_or(0);
            {
                let _gate = crate::table::SNAPSHOT_GATE.lock().unwrap();
                for (_, v, m) in &columns {
                    env.token.check()?;
                    if v.shape() != [rows]
                        || !v.generators().is_empty()
                        || m.as_ref().is_some_and(|v| {
                            v.shape() != [rows]
                                || v.dtype() != DType::Bool
                                || !v.generators().is_empty()
                        })
                    {
                        return Err(invalid());
                    }
                    let largest = v
                        .inputs()
                        .iter()
                        .map(|a| a.size())
                        .chain(m.iter().flat_map(Expr::inputs).map(|a| a.size()))
                        .max()
                        .unwrap_or(0);
                    let _offsets = reserve(&env.budget, largest, std::mem::size_of::<usize>())?;
                    env.freeze(v)?;
                    if let Some(m) = m {
                        env.freeze(m)?;
                    }
                }
            }
            for (name, v, m) in columns {
                env.token.check()?;
                let values = env.eval_dense(&v)?;
                let valid = match m {
                    Some(m) => env.eval_dense(&m)?,
                    None => {
                        let charge = reserve(&env.budget, rows, 1)?;
                        DenseArray::new(vec![rows], Buffer::Bool(vec![true; rows]), charge)?
                    }
                };
                names.push(name);
                materialized.push(MaterializedColumn { values, valid });
            }
            write(
                &path,
                format,
                &names,
                &Table {
                    columns: materialized,
                    rows,
                    pivot_columns: None,
                },
                mode,
                &env.budget,
                &env.token,
            )?;
            Ok(JobOutput::Unit)
        });
        self.submit(
            OperationKind::FileWrite,
            "write_text",
            Device::Cpu,
            token,
            work,
            done,
        )
    }
}
#[cfg(test)]
#[path = "io_text_tests.rs"]
mod tests;
