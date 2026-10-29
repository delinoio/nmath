// SPDX-License-Identifier: Apache-2.0
//! Bounded batches for row-local expressions and select/filter pipelines.
//! Sort, join, group, and non-row-local expressions remain materialization
//! boundaries. Snapshots and completed results still need their full budget.
use std::collections::HashMap;

use super::*;
use crate::{expr::ExprKind, kernels};

const BATCH_ROWS: usize = 1024;

pub(super) struct Frozen {
    inputs: HashMap<usize, DenseArray>,
}

fn supported(e: &Expr) -> bool {
    if e.shape().len() > 1 {
        return false;
    }
    match e.kind() {
        ExprKind::Input(_) | ExprKind::Scalar(_) => true,
        ExprKind::Unary(_, e) | ExprKind::Cast(e, _) => supported(e),
        ExprKind::Binary { lhs, rhs, .. } => supported(lhs) && supported(rhs),
        _ => false,
    }
}
fn source_supported(cols: &[Column]) -> bool {
    cols.iter()
        .all(|c| supported(&c.values) && c.valid.as_ref().is_none_or(supported))
}
impl Frozen {
    pub(super) fn new(plan: &Plan, env: &mut JobEnv) -> Result<Self> {
        let mut frozen = Self {
            inputs: HashMap::new(),
        };
        // One gate covers both batched and materialized source expressions,
        // including every validity mask in both branches of a join.
        let _gate = SNAPSHOT_GATE.lock().unwrap();
        frozen.freeze_plan(plan, env)?;
        Ok(frozen)
    }

    fn freeze_plan(&mut self, plan: &Plan, env: &mut JobEnv) -> Result<()> {
        match plan {
            Plan::Source(cols) => {
                let batched = source_supported(cols);
                for c in cols {
                    for e in std::iter::once(&c.values).chain(c.valid.iter()) {
                        env.token.check()?;
                        if batched {
                            self.freeze_expr(e, env)?;
                        } else {
                            let largest = e.inputs().iter().map(|a| a.size()).max().unwrap_or(0);
                            let _offsets = reserve(env, largest, std::mem::size_of::<usize>())?;
                            env.freeze(e)?;
                        }
                    }
                }
            }
            Plan::Join(a, b, ..) => {
                self.freeze_plan(a, env)?;
                self.freeze_plan(b, env)?;
            }
            Plan::Select(p, ..) | Plan::Filter(p, ..) | Plan::Sort(p, ..) | Plan::Group(p, ..) => {
                self.freeze_plan(p, env)?
            }
        }
        Ok(())
    }

    fn freeze_expr(&mut self, e: &Expr, env: &JobEnv) -> Result<()> {
        env.token.check()?;
        match e.kind() {
            ExprKind::Input(a) => {
                let id = a as *const crate::NdArray as usize;
                if let std::collections::hash_map::Entry::Vacant(entry) = self.inputs.entry(id) {
                    // NdArray::snapshot gathers through a temporary offset
                    // vector.
                    let _offsets = reserve(env, a.size(), std::mem::size_of::<usize>())?;
                    entry.insert(a.snapshot(&env.budget)?);
                }
            }
            ExprKind::Unary(_, e) | ExprKind::Cast(e, _) => self.freeze_expr(e, env)?,
            ExprKind::Binary { lhs, rhs, .. } => {
                self.freeze_expr(lhs, env)?;
                self.freeze_expr(rhs, env)?;
            }
            ExprKind::Scalar(_) => (),
            _ => return Err(invalid()),
        }
        Ok(())
    }

    fn eval(&self, e: &Expr, start: usize, len: usize, env: &JobEnv) -> Result<DenseArray> {
        env.token.check()?;
        match e.kind() {
            ExprKind::Input(a) => {
                let data = self
                    .inputs
                    .get(&(a as *const crate::NdArray as usize))
                    .ok_or_else(invalid)?;
                let singleton = data.size() == 1;
                let n = if singleton { 1 } else { len };
                let mut out = allocate(env, e.dtype(), n)?;
                out.shape = if e.shape().is_empty() {
                    vec![]
                } else {
                    vec![n]
                };
                for i in 0..n {
                    out.buffer
                        .set_from(i, &data.buffer, if singleton { 0 } else { start + i });
                }
                Ok(out)
            }
            ExprKind::Scalar(s) => {
                let charge = reserve(env, 1, e.dtype().item_size())?;
                DenseArray::new(vec![], Buffer::filled(e.dtype(), 1, s.clone())?, charge)
            }
            ExprKind::Unary(op, e) => kernels::unary(
                *op,
                &self.eval(e, start, len, env)?,
                &env.budget,
                &env.token,
            ),
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                compute_dtype,
            } => {
                let a = self.eval(lhs, start, len, env)?;
                let b = self.eval(rhs, start, len, env)?;
                kernels::binary(*op, &a, &b, *compute_dtype, &env.budget, &env.token)
            }
            ExprKind::Cast(e, dtype) => {
                let a = self.eval(e, start, len, env)?;
                let charge = reserve(env, a.size(), dtype.item_size())?;
                DenseArray::new(
                    a.shape.clone(),
                    a.buffer.cast_cancellable(*dtype, &env.token)?,
                    charge,
                )
            }
            _ => Err(invalid()),
        }
    }

    pub(super) fn execute(&self, plan: &Plan, env: &JobEnv) -> Option<Result<Table>> {
        let mut projection = Vec::new();
        let mut filters = Vec::new();
        let cols = pipeline(plan, &mut projection, &mut filters)?;
        Some(self.collect(cols, &projection, &filters, env))
    }

    fn batch(&self, cols: &[Column], start: usize, len: usize, env: &JobEnv) -> Result<Table> {
        let mut columns = Vec::with_capacity(cols.len());
        for c in cols {
            let values = self.eval(&c.values, start, len, env)?;
            let valid = match &c.valid {
                Some(e) => self.eval(e, start, len, env)?,
                None => all_valid(env, len)?,
            };
            columns.push(MaterializedColumn { values, valid });
        }
        Ok(Table { columns, rows: len })
    }

    fn collect(
        &self,
        cols: &[Column],
        projection: &[usize],
        filters: &[usize],
        env: &JobEnv,
    ) -> Result<Table> {
        env.token.check()?;
        for &k in filters {
            if cols[k].values.dtype() != DType::Bool {
                return Err(invalid());
            }
        }
        let rows = cols.first().map_or(0, |c| c.values.shape()[0]);
        // Scalar subexpressions must still report errors on empty inputs.
        if rows == 0 {
            self.batch(cols, 0, 0, env)?;
        }
        let keep = |batch: &Table, i| {
            filters.iter().all(|&k| {
                batch.columns[k].valid(i)
                    && batch.columns[k].values.buffer.get(i) == Scalar::Bool(true)
            })
        };
        // Count against the same frozen inputs before reserving the exact
        // result. No input-sized selection vector or growing uncharged
        // output is needed.
        let mut count = rows;
        if !filters.is_empty() {
            count = 0;
            for start in (0..rows).step_by(BATCH_ROWS) {
                let batch = self.batch(cols, start, (rows - start).min(BATCH_ROWS), env)?;
                for i in 0..batch.rows {
                    count += usize::from(keep(&batch, i));
                }
            }
        }
        let mut columns = Vec::with_capacity(projection.len());
        for &k in projection {
            columns.push(MaterializedColumn {
                values: allocate(env, cols[k].values.dtype(), count)?,
                valid: allocate(env, DType::Bool, count)?,
            });
        }
        let mut dest = 0;
        for start in (0..rows).step_by(BATCH_ROWS) {
            let batch = self.batch(cols, start, (rows - start).min(BATCH_ROWS), env)?;
            for i in 0..batch.rows {
                if keep(&batch, i) {
                    for (out, &k) in columns.iter_mut().zip(projection) {
                        out.values
                            .buffer
                            .set_from(dest, &batch.columns[k].values.buffer, i);
                        out.valid
                            .buffer
                            .set_from(dest, &batch.columns[k].valid.buffer, i);
                    }
                    dest += 1;
                }
            }
        }
        env.token.check()?;
        Ok(Table {
            columns,
            rows: count,
        })
    }
}

fn pipeline<'a>(
    plan: &'a Plan,
    projection: &mut Vec<usize>,
    filters: &mut Vec<usize>,
) -> Option<&'a [Column]> {
    match plan {
        Plan::Source(cols) if source_supported(cols) => {
            projection.extend(0..cols.len());
            Some(cols)
        }
        Plan::Select(p, indices) => {
            let cols = pipeline(p, projection, filters)?;
            *projection = indices.iter().map(|&i| projection[i]).collect();
            Some(cols)
        }
        Plan::Filter(p, k) => {
            let cols = pipeline(p, projection, filters)?;
            filters.push(projection[*k]);
            Some(cols)
        }
        _ => None,
    }
}
