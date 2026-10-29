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
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Pandas,
    Polars,
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
    fn sources(&self, out: &mut Vec<Expr>) {
        match self {
            Self::Source(cols) => {
                for c in cols {
                    out.push(c.values.clone());
                    if let Some(v) = &c.valid {
                        out.push(v.clone());
                    }
                }
            }
            Self::Join(a, b, ..) => {
                a.sources(out);
                b.sources(out);
            }
            Self::Select(a, ..) | Self::Filter(a, ..) | Self::Sort(a, ..) | Self::Group(a, ..) => {
                a.sources(out)
            }
        }
    }

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
        let mut inputs = Vec::new();
        plan.sources(&mut inputs);
        {
            let _gate = SNAPSHOT_GATE.lock().unwrap();
            env.token.check()?;
            // NdArray::snapshot currently uses an offset vector. Reserve its
            // largest temporary here until snapshot owns that reservation.
            let largest = inputs
                .iter()
                .flat_map(Expr::inputs)
                .map(|a| a.size())
                .max()
                .unwrap_or(0);
            let _offsets = reserve(env, largest, std::mem::size_of::<usize>())?;
            for input in &inputs {
                env.token.check()?;
                env.freeze(input)?;
            }
        }
        let table = execute(&plan, env)?;
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
    })
}
fn execute(plan: &Plan, env: &mut JobEnv) -> Result<Table> {
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
            Ok(Table { columns, rows })
        }
        Plan::Select(p, indices) => {
            let t = execute(p, env)?;
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
            })
        }
        Plan::Filter(p, k) => {
            let t = execute(p, env)?;
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
            let t = execute(p, env)?;
            let c = &t.columns[*k];
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
            let a = execute(a, env)?;
            let b = execute(b, env)?;
            if a.columns[*x].values.buffer.dtype() != b.columns[*y].values.buffer.dtype() {
                return Err(Error::new(
                    ErrorCode::DTypeUnsupported,
                    "Join keys must have the same numeric dtype. Cast keys explicitly.",
                ));
            }
            let mut count = 0usize;
            for i in 0..a.rows {
                let mut found = false;
                for j in 0..b.rows {
                    env.token.check()?;
                    if equal(&a.columns[*x], i, &b.columns[*y], j, *nulls) {
                        count = count.checked_add(1).ok_or_else(invalid)?;
                        found = true;
                    }
                }
                if *left && !found {
                    count = count.checked_add(1).ok_or_else(invalid)?;
                }
            }
            let _charge = reserve(env, count, 2 * std::mem::size_of::<Option<usize>>())?;
            let mut ar = Vec::with_capacity(count);
            let mut br = Vec::with_capacity(count);
            for i in 0..a.rows {
                let mut found = false;
                for j in 0..b.rows {
                    env.token.check()?;
                    if equal(&a.columns[*x], i, &b.columns[*y], j, *nulls) {
                        ar.push(Some(i));
                        br.push(Some(j));
                        found = true;
                    }
                }
                if *left && !found {
                    ar.push(Some(i));
                    br.push(None);
                }
            }
            let mut out = take(&a, &ar, env)?;
            out.columns.extend(take(&b, &br, env)?.columns);
            Ok(out)
        }
        Plan::Group(p, k, v, op, drop_nulls, flavor) => {
            group(execute(p, env)?, *k, *v, *op, *drop_nulls, *flavor, env)
        }
    }
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
    let _charge = reserve(env, t.rows, std::mem::size_of::<Option<usize>>())?;
    let mut reps: Vec<Option<usize>> = Vec::with_capacity(t.rows);
    for i in 0..t.rows {
        env.token.check()?;
        if drop_nulls && !key.valid(i) {
            continue;
        }
        let mut found = false;
        for &j in &reps {
            env.token.check()?;
            if equal(key, i, key, j.unwrap(), true) {
                found = true;
                break;
            }
        }
        if !found {
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
        for i in 0..t.rows {
            env.token.check()?;
            if !equal(key, rep.unwrap(), key, i, true) || !value.valid(i) {
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
                Scalar::Complex(_) => return Err(invalid()),
            };
            if !matches!(s, Scalar::Float(_)) {
                float += s.as_f64();
            }
            if extreme.is_none_or(|old| {
                let c = cmp(s, old);
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
            dtype = match (dtype, result) {
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
    })
}

#[cfg(test)]
#[path = "table_tests.rs"]
mod tests;
