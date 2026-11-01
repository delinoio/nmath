// SPDX-License-Identifier: Apache-2.0
//! Numeric, nullable tables. One plan has one admission and one frozen input
//! set.
use std::{
    cmp::Ordering,
    sync::{Arc, Mutex},
};

use crate::{
    Buffer, CancellationToken, Context, DType, DenseArray, Error, ErrorCode, Expr, Result, Scalar,
    context::{Done, JobOutput, OperationKind, Work},
    expr::JobEnv,
    memory::{Device, Reservation},
};

#[path = "table_stream.rs"]
mod streaming;

// Array mutation commits take this gate too. Keep the gate only while freezing,
// never while evaluating kernels or waiting for scheduler/storage tickets.
pub(crate) static SNAPSHOT_GATE: Mutex<()> = Mutex::new(());

#[derive(Clone)]
pub struct Column {
    pub values: Expr,
    pub valid: Option<Expr>,
}
#[derive(Clone)]
pub enum Plan {
    Source(Vec<Column>),
    Select(Box<Plan>, Vec<usize>),
    Filter(Box<Plan>, usize),
    Sort(Box<Plan>, usize, bool, bool),
    Join(Box<Plan>, Box<Plan>, usize, usize, bool, bool),
    Group(Box<Plan>, usize, usize, Aggregate, bool, Flavor),
    Pivot(Box<Plan>, usize, usize, usize, PivotMode, Flavor),
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Pandas,
    Polars,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PivotMode {
    Unique,
    Sum,
    Mean,
    Min,
    Max,
    Count,
}
#[derive(Clone, Copy)]
pub enum Aggregate {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}
pub struct Table {
    pub columns: Vec<MaterializedColumn>,
    pub rows: usize,
    /// Generated value-column names for a pivot plan. Other plans return None.
    pub pivot_columns: Option<Vec<String>>,
}
pub struct MaterializedColumn {
    pub values: DenseArray,
    pub valid: DenseArray,
}
pub type TableDone = Box<dyn FnOnce(Result<Table>) + Send>;
fn invalid() -> Error {
    Error::new(
        ErrorCode::IndexInvalid,
        "Invalid table operation. Check column positions and argument types.",
    )
}
fn shape() -> Error {
    Error::new(
        ErrorCode::ShapeMismatch,
        "Table columns must have equal one-dimensional shapes. Check the schema.",
    )
}
impl Plan {
    fn validate(&self, context: u64) -> Result<usize> {
        Ok(match self {
            Self::Source(cols) => {
                let mut len = None;
                for c in cols {
                    if matches!(c.values.dtype(), DType::Complex64 | DType::Complex128) {
                        return Err(Error::new(
                            ErrorCode::DTypeUnsupported,
                            "Complex table columns are not supported. Use array operations.",
                        ));
                    }
                    if c.values.context_id().is_some_and(|id| id != context)
                        || c.valid
                            .as_ref()
                            .and_then(Expr::context_id)
                            .is_some_and(|id| id != context)
                    {
                        return Err(Error::new(
                            ErrorCode::ContextMismatch,
                            "Table inputs must use one context. Copy inputs explicitly.",
                        ));
                    }
                    if c.values.shape().len() != 1 || len.is_some_and(|n| n != c.values.shape()[0])
                    {
                        return Err(shape());
                    }
                    len = Some(c.values.shape()[0]);
                    // Random state transactions are not yet part of table
                    // plans.
                    if !c.values.generators().is_empty() {
                        return Err(invalid());
                    }
                    if let Some(v) = &c.valid {
                        if v.dtype() != DType::Bool
                            || v.shape() != c.values.shape()
                            || !v.generators().is_empty()
                        {
                            return Err(shape());
                        }
                    }
                }
                cols.len()
            }
            Self::Select(p, cols) => {
                let n = p.validate(context)?;
                if cols.iter().any(|&i| i >= n) {
                    return Err(invalid());
                }
                cols.len()
            }
            Self::Filter(p, k) | Self::Sort(p, k, ..) => {
                let n = p.validate(context)?;
                if *k >= n {
                    return Err(invalid());
                }
                n
            }
            Self::Join(a, b, x, y, ..) => {
                let n = a.validate(context)?;
                let m = b.validate(context)?;
                if *x >= n || *y >= m {
                    return Err(invalid());
                }
                n + m
            }
            Self::Group(p, k, v, ..) => {
                let n = p.validate(context)?;
                if *k >= n || *v >= n {
                    return Err(invalid());
                }
                2
            }
            Self::Pivot(p, row, column, value, ..) => {
                let n = p.validate(context)?;
                if *row >= n || *column >= n || *value >= n {
                    return Err(invalid());
                }
                n
            }
        })
    }
}
/// Submit the complete plan through the existing FIFO scheduler.
pub fn compute(
    context: &Context,
    plan: Plan,
    token: CancellationToken,
    done: TableDone,
) -> Result<u64> {
    plan.validate(context.id())?;
    let result = Arc::new(Mutex::new(None));
    let output = result.clone();
    let work: Work = Box::new(move |env| {
        let frozen = streaming::Frozen::new(&plan, env)?;
        let table = execute(&plan, env, &frozen)?;
        env.token.check()?;
        *output.lock().unwrap() = Some(table);
        Ok(JobOutput::Unit)
    });
    let finish: Done = Box::new(move |status| {
        done(status.and_then(|_| result.lock().unwrap().take().ok_or_else(invalid)));
    });
    context.submit(
        OperationKind::Compute,
        "table",
        Device::Cpu,
        token,
        work,
        finish,
    )
}
fn reserve(env: &JobEnv, n: usize, width: usize) -> Result<Reservation> {
    let bytes = n
        .checked_mul(width)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MemoryBudgetExceeded,
                "Table allocation is too large. Reduce the input size.",
            )
        })?;
    env.budget.reserve(bytes)
}
fn allocate(env: &JobEnv, dtype: DType, n: usize) -> Result<DenseArray> {
    let charge = reserve(env, n, dtype.item_size())?;
    DenseArray::new(vec![n], Buffer::filled(dtype, n, Scalar::Int(0))?, charge)
}
fn all_valid(env: &JobEnv, n: usize) -> Result<DenseArray> {
    let charge = reserve(env, n, 1)?;
    DenseArray::new(vec![n], Buffer::Bool(vec![true; n]), charge)
}
impl MaterializedColumn {
    fn valid(&self, i: usize) -> bool {
        self.valid.buffer.get(i) == Scalar::Bool(true)
    }
}
fn cmp(a: Scalar, b: Scalar) -> Ordering {
    match (a, b) {
        (Scalar::Int(a), Scalar::Int(b)) => a.cmp(&b),
        (Scalar::UInt(a), Scalar::UInt(b)) => a.cmp(&b),
        (Scalar::Bool(a), Scalar::Bool(b)) => a.cmp(&b),
        (Scalar::Float(a), Scalar::Float(b)) => {
            if a == b {
                Ordering::Equal
            } else if a.is_nan() {
                if b.is_nan() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            } else if b.is_nan() {
                Ordering::Less
            } else {
                a.partial_cmp(&b).unwrap()
            }
        }
        _ => Ordering::Equal,
    }
}
fn equal(a: &MaterializedColumn, i: usize, b: &MaterializedColumn, j: usize, nulls: bool) -> bool {
    match (a.valid(i), b.valid(j)) {
        (true, true) => cmp(a.values.buffer.get(i), b.values.buffer.get(j)) == Ordering::Equal,
        (false, false) => nulls,
        _ => false,
    }
}
// Fixed-capacity open addressing keeps all row-index storage reserved before
// allocation. Randomized hashes avoid predictable collision chains. Keys use
// the same NaN, signed-zero, exact-integer, and null equivalence as cmp/equal.
struct KeyIndex {
    slots: Vec<usize>,
    next: Vec<usize>,
    hash: std::collections::hash_map::RandomState,
    _slots_charge: Reservation,
    _next_charge: Reservation,
}
const NO_ROW: usize = usize::MAX;
impl KeyIndex {
    fn excluded(key: &MaterializedColumn, row: usize, nulls: bool, exclude_nan: bool) -> bool {
        (!nulls && !key.valid(row))
            || (exclude_nan
                && key.valid(row)
                && matches!(key.values.buffer.get(row), Scalar::Float(n) if n.is_nan()))
    }

    fn new(key: &MaterializedColumn, nulls: bool, exclude_nan: bool, env: &JobEnv) -> Result<Self> {
        let rows = key.values.size();
        let capacity = rows
            .checked_mul(2)
            .and_then(|n| n.max(1).checked_next_power_of_two())
            .ok_or_else(invalid)?;
        let slots_charge = reserve(env, capacity, std::mem::size_of::<usize>())?;
        let next_charge = reserve(env, rows, std::mem::size_of::<usize>())?;
        let mut index = Self {
            slots: vec![NO_ROW; capacity],
            next: vec![NO_ROW; rows],
            hash: std::collections::hash_map::RandomState::new(),
            _slots_charge: slots_charge,
            _next_charge: next_charge,
        };
        // Prepending in reverse order preserves original right-match order and
        // the input accumulation order within each group.
        for i in (0..rows).rev() {
            env.token.check()?;
            if Self::excluded(key, i, nulls, exclude_nan) {
                continue;
            }
            let slot = index.slot(key, key, i, env)?;
            index.next[i] = index.slots[slot];
            index.slots[slot] = i;
        }
        Ok(index)
    }

    fn slot(
        &self,
        indexed: &MaterializedColumn,
        key: &MaterializedColumn,
        row: usize,
        env: &JobEnv,
    ) -> Result<usize> {
        use std::hash::BuildHasher;
        if key.values.dtype().kind() == crate::dtype::DTypeKind::Other {
            return Err(crate::dtype::unsupported());
        }
        let bits = if !key.valid(row) {
            0
        } else {
            match key.values.buffer.get(row) {
                Scalar::Int(n) => n as u64,
                Scalar::UInt(n) => n,
                Scalar::Bool(n) => u64::from(n),
                Scalar::Float(n) if n == 0.0 => 0,
                Scalar::Float(n) if n.is_nan() => f64::NAN.to_bits(),
                Scalar::Float(n) => n.to_bits(),
                Scalar::Complex(_) | Scalar::Text(_) | Scalar::Temporal(..) => {
                    return Err(invalid());
                }
            }
        };
        let mut slot = self.hash.hash_one((key.valid(row), bits)) as usize & (self.slots.len() - 1);
        loop {
            env.token.check()?;
            let head = self.slots[slot];
            if head == NO_ROW || equal(indexed, head, key, row, true) {
                return Ok(slot);
            }
            slot = (slot + 1) & (self.slots.len() - 1);
        }
    }

    fn find(
        &self,
        indexed: &MaterializedColumn,
        key: &MaterializedColumn,
        row: usize,
        nulls: bool,
        exclude_nan: bool,
        env: &JobEnv,
    ) -> Result<usize> {
        if Self::excluded(key, row, nulls, exclude_nan) {
            return Ok(NO_ROW);
        }
        Ok(self.slots[self.slot(indexed, key, row, env)?])
    }
}
fn take(table: &Table, rows: &[Option<usize>], env: &JobEnv) -> Result<Table> {
    let mut columns = Vec::new();
    for col in &table.columns {
        let mut values = allocate(env, col.values.buffer.dtype(), rows.len())?;
        let mut valid = allocate(env, DType::Bool, rows.len())?;
        for (i, row) in rows.iter().enumerate() {
            if i % 1024 == 0 {
                env.token.check()?;
            }
            if let Some(j) = row {
                values.buffer.set_from(i, &col.values.buffer, *j);
                valid.buffer.set_from(i, &col.valid.buffer, *j);
            }
        }
        columns.push(MaterializedColumn { values, valid });
    }
    Ok(Table {
        columns,
        rows: rows.len(),
        pivot_columns: None,
    })
}
fn execute(plan: &Plan, env: &mut JobEnv, frozen: &streaming::Frozen) -> Result<Table> {
    if let Some(result) = frozen.execute(plan, env) {
        return result;
    }
    env.token.check()?;
    match plan {
        Plan::Source(cols) => {
            let mut columns = Vec::new();
            let mut rows = 0;
            for c in cols {
                let values = env.eval_dense(&c.values)?;
                rows = values.buffer.len();
                let valid = match &c.valid {
                    Some(v) => env.eval_dense(v)?,
                    None => all_valid(env, rows)?,
                };
                columns.push(MaterializedColumn { values, valid });
            }
            Ok(Table {
                columns,
                rows,
                pivot_columns: None,
            })
        }
        Plan::Select(p, indices) => {
            let t = execute(p, env, frozen)?;
            let mut columns = Vec::new();
            for &i in indices {
                let c = &t.columns[i];
                let a = reserve(env, t.rows, c.values.buffer.dtype().item_size())?;
                let b = reserve(env, t.rows, 1)?;
                columns.push(MaterializedColumn {
                    values: DenseArray::new(vec![t.rows], c.values.buffer.clone(), a)?,
                    valid: DenseArray::new(vec![t.rows], c.valid.buffer.clone(), b)?,
                });
            }
            Ok(Table {
                columns,
                rows: t.rows,
                pivot_columns: None,
            })
        }
        Plan::Filter(p, k) => {
            let t = execute(p, env, frozen)?;
            let c = &t.columns[*k];
            if c.values.buffer.dtype() != DType::Bool {
                return Err(invalid());
            }
            let _charge = reserve(env, t.rows, std::mem::size_of::<Option<usize>>())?;
            let mut rows = Vec::with_capacity(t.rows);
            for i in 0..t.rows {
                if i % 1024 == 0 {
                    env.token.check()?;
                }
                if c.valid(i) && c.values.buffer.get(i) == Scalar::Bool(true) {
                    rows.push(Some(i));
                }
            }
            take(&t, &rows, env)
        }
        Plan::Sort(p, k, descending, nulls_last) => {
            let t = execute(p, env, frozen)?;
            let c = &t.columns[*k];
            if c.values.dtype().kind() == crate::dtype::DTypeKind::Other {
                return Err(crate::dtype::unsupported());
            }
            let _charge = reserve(env, t.rows, 2 * std::mem::size_of::<Option<usize>>())?;
            let mut rows: Vec<_> = (0..t.rows).map(Some).collect();
            let mut scratch = vec![None; t.rows];
            let compare = |a: usize, b: usize| match (c.valid(a), c.valid(b)) {
                (false, true) => {
                    if *nulls_last {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
                (true, false) => {
                    if *nulls_last {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
                (false, false) => Ordering::Equal,
                _ => {
                    let o = cmp(c.values.buffer.get(a), c.values.buffer.get(b));
                    if *descending { o.reverse() } else { o }
                }
            };
            let mut width = 1usize;
            while width < t.rows {
                for start in (0..t.rows).step_by(width.saturating_mul(2)) {
                    let mid = (start + width).min(t.rows);
                    let end = mid.saturating_add(width).min(t.rows);
                    let (mut left, mut right) = (start, mid);
                    for dest in &mut scratch[start..end] {
                        env.token.check()?;
                        if right == end
                            || (left < mid
                                && compare(rows[left].unwrap(), rows[right].unwrap())
                                    != Ordering::Greater)
                        {
                            *dest = rows[left];
                            left += 1;
                        } else {
                            *dest = rows[right];
                            right += 1;
                        }
                    }
                }
                std::mem::swap(&mut rows, &mut scratch);
                width = width.saturating_mul(2);
            }
            take(&t, &rows, env)
        }
        Plan::Join(a, b, x, y, left, nulls) => {
            let a = execute(a, env, frozen)?;
            let b = execute(b, env, frozen)?;
            if a.columns[*x].values.buffer.dtype() != b.columns[*y].values.buffer.dtype() {
                return Err(Error::new(
                    ErrorCode::DTypeUnsupported,
                    "Join keys must have the same numeric dtype. Cast keys explicitly.",
                ));
            }
            let index = KeyIndex::new(&b.columns[*y], *nulls, false, env)?;
            let mut count = 0usize;
            for i in 0..a.rows {
                let mut j = index.find(&b.columns[*y], &a.columns[*x], i, *nulls, false, env)?;
                if *left && j == NO_ROW {
                    count = count.checked_add(1).ok_or_else(invalid)?;
                }
                while j != NO_ROW {
                    env.token.check()?;
                    count = count.checked_add(1).ok_or_else(invalid)?;
                    j = index.next[j];
                }
            }
            let _charge = reserve(env, count, 2 * std::mem::size_of::<Option<usize>>())?;
            let mut ar = Vec::with_capacity(count);
            let mut br = Vec::with_capacity(count);
            for i in 0..a.rows {
                let mut j = index.find(&b.columns[*y], &a.columns[*x], i, *nulls, false, env)?;
                if *left && j == NO_ROW {
                    ar.push(Some(i));
                    br.push(None);
                }
                while j != NO_ROW {
                    env.token.check()?;
                    ar.push(Some(i));
                    br.push(Some(j));
                    j = index.next[j];
                }
            }
            let mut out = take(&a, &ar, env)?;
            out.columns.extend(take(&b, &br, env)?.columns);
            Ok(out)
        }
        Plan::Group(p, k, v, op, drop_nulls, flavor) => group(
            execute(p, env, frozen)?,
            *k,
            *v,
            *op,
            *drop_nulls,
            *flavor,
            env,
        ),
        Plan::Pivot(p, row, column, value, mode, flavor) => pivot(
            execute(p, env, frozen)?,
            *row,
            *column,
            *value,
            *mode,
            *flavor,
            env,
        ),
    }
}
#[derive(Clone)]
struct PivotCell {
    occurrences: usize,
    count: usize,
    signed: i64,
    unsigned: u64,
    float: f64,
    extreme: Option<Scalar>,
    single: Option<usize>,
}

fn key_name(value: Scalar) -> String {
    match value {
        Scalar::Int(n) => n.to_string(),
        Scalar::UInt(n) => n.to_string(),
        Scalar::Bool(n) => n.to_string(),
        Scalar::Float(n) => format!("{n}"),
        Scalar::Complex(_) | Scalar::Text(_) | Scalar::Temporal(..) => String::new(),
    }
}

fn distinct_rows(index: &KeyIndex, key: &MaterializedColumn, env: &JobEnv) -> Result<Vec<usize>> {
    let _charge = reserve(env, key.values.size(), std::mem::size_of::<usize>())?;
    let mut reps = Vec::new();
    for i in 0..key.values.size() {
        env.token.check()?;
        if index.find(key, key, i, false, false, env)? == i {
            reps.push(i);
        }
    }
    Ok(reps)
}

fn pivot(
    t: Table,
    row_key: usize,
    column_key: usize,
    value: usize,
    mode: PivotMode,
    flavor: Flavor,
    env: &JobEnv,
) -> Result<Table> {
    let rows = &t.columns[row_key];
    let columns = &t.columns[column_key];
    let values = &t.columns[value];
    for key in [rows, columns] {
        if matches!(key.values.dtype().kind(), crate::dtype::DTypeKind::Other) {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Pivot keys must be numeric or boolean. Cast keys explicitly.",
            ));
        }
    }
    let input_dtype = values.values.dtype();
    if matches!(
        input_dtype,
        DType::Complex64 | DType::Complex128 | DType::Str(_)
    ) || input_dtype.temporal_unit().is_some()
    {
        return Err(Error::new(
            ErrorCode::DTypeUnsupported,
            "Pivot values must be numeric or boolean. Cast values explicitly.",
        ));
    }
    let row_index = KeyIndex::new(rows, false, false, env)?;
    let column_index = KeyIndex::new(columns, false, false, env)?;
    let mut row_reps = distinct_rows(&row_index, rows, env)?;
    let mut column_reps = distinct_rows(&column_index, columns, env)?;
    let key = |value: &MaterializedColumn, row: usize| value.values.buffer.get(row);
    if flavor == Flavor::Pandas {
        row_reps.sort_by(|&a, &b| cmp(key(rows, a), key(rows, b)));
        column_reps.sort_by(|&a, &b| cmp(key(columns, a), key(columns, b)));
    }
    let _maps_charge = reserve(env, t.rows, 2 * (2 * std::mem::size_of::<usize>() + 32))?;
    let row_ids: std::collections::HashMap<usize, usize> = row_reps
        .iter()
        .enumerate()
        .map(|(id, &row)| (row, id))
        .collect();
    let column_ids: std::collections::HashMap<usize, usize> = column_reps
        .iter()
        .enumerate()
        .map(|(id, &row)| (row, id))
        .collect();
    let _cells_charge = reserve(
        env,
        t.rows,
        std::mem::size_of::<PivotCell>() + std::mem::size_of::<((usize, usize), usize)>() + 64,
    )?;
    let mut cells: Vec<PivotCell> = Vec::new();
    let mut cell_ids: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for i in 0..t.rows {
        env.token.check()?;
        let row_head = row_index.find(rows, rows, i, false, false, env)?;
        let column_head = column_index.find(columns, columns, i, false, false, env)?;
        if row_head == NO_ROW || column_head == NO_ROW {
            continue;
        }
        let cell_key = (row_ids[&row_head], column_ids[&column_head]);
        let cell_id = match cell_ids.get(&cell_key) {
            Some(&id) => id,
            None => {
                let id = cells.len();
                cells.push(PivotCell {
                    occurrences: 0,
                    count: 0,
                    signed: 0,
                    unsigned: 0,
                    float: 0.0,
                    extreme: None,
                    single: None,
                });
                cell_ids.insert(cell_key, id);
                id
            }
        };
        let cell = &mut cells[cell_id];
        cell.occurrences += 1;
        if mode == PivotMode::Unique {
            if cell.occurrences > 1 {
                return Err(Error::new(
                    ErrorCode::IndexInvalid,
                    "Pivot requires unique index and column pairs. Use an aggregate mode.",
                ));
            }
            cell.single = Some(i);
            continue;
        }
        if !values.valid(i) {
            continue;
        }
        let sample = values.values.buffer.get(i);
        if flavor == Flavor::Pandas
            && matches!(sample, Scalar::Float(n) if n.is_nan())
            && mode != PivotMode::Count
        {
            continue;
        }
        cell.count += 1;
        match sample {
            Scalar::Int(n) => cell.signed = cell.signed.wrapping_add(n),
            Scalar::UInt(n) => cell.unsigned = cell.unsigned.wrapping_add(n),
            Scalar::Bool(n) => cell.signed += i64::from(n),
            Scalar::Float(n) => cell.float += n,
            _ => return Err(invalid()),
        }
        if !matches!(sample, Scalar::Float(_)) {
            cell.float += sample.as_f64();
        }
        if mode == PivotMode::Min || mode == PivotMode::Max {
            if cell.extreme.as_ref().is_none_or(|old: &Scalar| {
                let ordering = cmp(sample.clone(), old.clone());
                if mode == PivotMode::Min {
                    ordering == Ordering::Less
                } else {
                    ordering == Ordering::Greater
                }
            }) {
                cell.extreme = Some(sample);
            }
        }
    }
    let dtype = match mode {
        PivotMode::Unique | PivotMode::Min | PivotMode::Max => input_dtype,
        PivotMode::Count => {
            if flavor == Flavor::Pandas {
                DType::Int64
            } else {
                DType::UInt32
            }
        }
        PivotMode::Mean => {
            if flavor == Flavor::Polars && input_dtype == DType::Float32 {
                DType::Float32
            } else {
                DType::Float64
            }
        }
        PivotMode::Sum if flavor == Flavor::Pandas => {
            if input_dtype == DType::Bool {
                DType::Int64
            } else {
                input_dtype
            }
        }
        PivotMode::Sum => match input_dtype {
            DType::Bool => DType::UInt32,
            DType::Int8 | DType::Int16 | DType::UInt8 | DType::UInt16 => DType::Int64,
            d => d,
        },
    };
    let out_rows = row_reps.len();
    let out_columns = column_reps.len();
    let mut key_values = allocate(env, rows.values.buffer.dtype(), out_rows)?;
    let mut key_valid = allocate(env, DType::Bool, out_rows)?;
    for (i, &row) in row_reps.iter().enumerate() {
        env.token.check()?;
        key_values.buffer.set_from(i, &rows.values.buffer, row);
        key_valid.buffer.set_from(i, &rows.valid.buffer, row);
    }
    let mut output = Vec::with_capacity(out_columns + 1);
    output.push(MaterializedColumn {
        values: key_values,
        valid: key_valid,
    });
    let mut names = Vec::with_capacity(out_columns);
    for (column_id, &column) in column_reps.iter().enumerate() {
        names.push(key_name(key(columns, column)));
        let mut data = allocate(env, dtype, out_rows)?;
        let mut valid = allocate(env, DType::Bool, out_rows)?;
        for row_id in 0..out_rows {
            env.token.check()?;
            let Some(&cell_id) = cell_ids.get(&(row_id, column_id)) else {
                continue;
            };
            let cell = &cells[cell_id];
            let (result, is_valid) = match mode {
                PivotMode::Unique => match cell.single {
                    Some(row) if values.valid(row) => (values.values.buffer.get(row), true),
                    _ => (Scalar::Int(0), false),
                },
                PivotMode::Count => (
                    if flavor == Flavor::Pandas {
                        Scalar::Int(cell.count as i64)
                    } else {
                        Scalar::UInt(cell.count as u64)
                    },
                    true,
                ),
                PivotMode::Mean if cell.count > 0 => {
                    (Scalar::Float(cell.float / cell.count as f64), true)
                }
                PivotMode::Sum => {
                    let result = match input_dtype {
                        DType::Bool | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => {
                            Scalar::Int(cell.signed)
                        }
                        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => {
                            Scalar::UInt(cell.unsigned)
                        }
                        _ => Scalar::Float(cell.float),
                    };
                    (result, flavor == Flavor::Pandas || cell.count > 0)
                }
                PivotMode::Min | PivotMode::Max if cell.count > 0 => {
                    (cell.extreme.clone().unwrap_or(Scalar::Int(0)), true)
                }
                _ => (Scalar::Int(0), false),
            };
            if is_valid {
                data.buffer.set_from(
                    row_id,
                    &Buffer::filled(dtype, 1, result).map_err(|_| invalid())?,
                    0,
                );
                valid.buffer.set_from(
                    row_id,
                    &Buffer::filled(DType::Bool, 1, Scalar::Bool(true)).map_err(|_| invalid())?,
                    0,
                );
            }
        }
        output.push(MaterializedColumn {
            values: data,
            valid,
        });
    }
    Ok(Table {
        columns: output,
        rows: out_rows,
        pivot_columns: Some(names),
    })
}

fn group(
    t: Table,
    k: usize,
    v: usize,
    op: Aggregate,
    drop_nulls: bool,
    flavor: Flavor,
    env: &JobEnv,
) -> Result<Table> {
    let key = &t.columns[k];
    let value = &t.columns[v];
    if value.values.dtype().kind() == crate::dtype::DTypeKind::Other {
        return Err(crate::dtype::unsupported());
    }
    let exclude_nan = drop_nulls && flavor == Flavor::Pandas;
    let index = KeyIndex::new(key, !drop_nulls, exclude_nan, env)?;
    let _charge = reserve(env, t.rows, std::mem::size_of::<Option<usize>>())?;
    let mut reps: Vec<Option<usize>> = Vec::with_capacity(t.rows);
    for i in 0..t.rows {
        env.token.check()?;
        if index.find(key, key, i, !drop_nulls, exclude_nan, env)? == i {
            reps.push(Some(i));
        }
    }
    let input_dtype = value.values.buffer.dtype();
    let mut dtype = match op {
        Aggregate::Count => {
            if flavor == Flavor::Pandas {
                DType::Int64
            } else {
                DType::UInt32
            }
        }
        Aggregate::Mean => {
            if flavor == Flavor::Polars && input_dtype == DType::Float32 {
                DType::Float32
            } else {
                DType::Float64
            }
        }
        Aggregate::Sum => {
            if flavor == Flavor::Pandas {
                if input_dtype == DType::Bool {
                    DType::Int64
                } else {
                    input_dtype
                }
            } else {
                match input_dtype {
                    DType::Bool => DType::UInt32,
                    DType::Int8 | DType::Int16 | DType::UInt8 | DType::UInt16 => DType::Int64,
                    d => d,
                }
            }
        }
        _ => input_dtype,
    };
    let _results_charge = reserve(env, reps.len(), std::mem::size_of::<Scalar>())?;
    let mut results = Vec::with_capacity(reps.len());
    let mut valid = all_valid(env, reps.len())?;
    for (g, &rep) in reps.iter().enumerate() {
        let mut count = 0i64;
        let mut signed = 0i64;
        let mut unsigned = 0u64;
        let mut float = 0f64;
        let mut extreme = None;
        let mut row = rep.unwrap();
        while row != NO_ROW {
            env.token.check()?;
            let i = row;
            row = index.next[i];
            if !value.valid(i) {
                continue;
            }
            let s = value.values.buffer.get(i);
            if flavor == Flavor::Pandas && matches!(s,Scalar::Float(n) if n.is_nan()) {
                continue;
            }
            count += 1;
            match s {
                Scalar::Int(n) => signed = signed.wrapping_add(n),
                Scalar::UInt(n) => unsigned = unsigned.wrapping_add(n),
                Scalar::Bool(n) => signed += i64::from(n),
                Scalar::Float(n) => float += n,
                Scalar::Complex(_) | Scalar::Text(_) | Scalar::Temporal(..) => {
                    return Err(invalid());
                }
            };
            if !matches!(s, Scalar::Float(_)) {
                float += s.as_f64();
            }
            if extreme.as_ref().is_none_or(|old: &Scalar| {
                let c = cmp(s.clone(), old.clone());
                if matches!(op, Aggregate::Min) {
                    c == Ordering::Less
                } else {
                    c == Ordering::Greater
                }
            }) {
                extreme = Some(s);
            }
        }
        let result = match op {
            Aggregate::Count => Scalar::Int(count),
            Aggregate::Mean => Scalar::Float(float / count as f64),
            Aggregate::Sum => match input_dtype {
                DType::Bool | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => {
                    Scalar::Int(signed)
                }
                DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => {
                    Scalar::UInt(unsigned)
                }
                _ => Scalar::Float(float),
            },
            _ => extreme.unwrap_or(Scalar::Int(0)),
        };
        // pandas retains a small integer dtype if every group sum fits it.
        // An overflowing group widens the entire result column to 64 bits.
        if flavor == Flavor::Pandas && matches!(op, Aggregate::Sum) {
            dtype = match (dtype, result.clone()) {
                (DType::Int8, Scalar::Int(n)) if i8::try_from(n).is_err() => DType::Int64,
                (DType::Int16, Scalar::Int(n)) if i16::try_from(n).is_err() => DType::Int64,
                (DType::Int32, Scalar::Int(n)) if i32::try_from(n).is_err() => DType::Int64,
                (DType::UInt8, Scalar::UInt(n)) if u8::try_from(n).is_err() => DType::UInt64,
                (DType::UInt16, Scalar::UInt(n)) if u16::try_from(n).is_err() => DType::UInt64,
                (DType::UInt32, Scalar::UInt(n)) if u32::try_from(n).is_err() => DType::UInt64,
                _ => dtype,
            };
        }
        if count == 0 && !matches!(op, Aggregate::Count | Aggregate::Sum) {
            if let Buffer::Bool(m) = &mut valid.buffer {
                m[g] = false;
            }
        }
        results.push(result);
    }
    let mut data = allocate(env, dtype, reps.len())?;
    let _scalar_charge = reserve(env, 1, 8)?;
    for (i, result) in results.into_iter().enumerate() {
        env.token.check()?;
        data.buffer
            .set_from(i, &Buffer::filled(dtype, 1, result)?, 0);
    }
    let mut key_data = allocate(env, key.values.buffer.dtype(), reps.len())?;
    let mut key_valid = allocate(env, DType::Bool, reps.len())?;
    for (i, j) in reps.iter().enumerate() {
        key_data.buffer.set_from(i, &key.values.buffer, j.unwrap());
        key_valid.buffer.set_from(i, &key.valid.buffer, j.unwrap());
    }
    Ok(Table {
        rows: reps.len(),
        columns: vec![
            MaterializedColumn {
                values: key_data,
                valid: key_valid,
            },
            MaterializedColumn {
                values: data,
                valid,
            },
        ],
        pivot_columns: None,
    })
}

#[cfg(test)]
#[path = "table_tests.rs"]
mod tests;
