// SPDX-License-Identifier: Apache-2.0
use nmath_core::table::{self, Aggregate, Column, Flavor, Plan};

use super::*;

#[napi]
pub struct NativeTable {
    plan: Plan,
}
#[napi(object, object_from_js = false)]
pub struct NativeTableResult {
    pub values: Vec<NativeArray>,
    pub valid: Vec<NativeArray>,
    pub rows: u32,
}
fn invalid_table() -> napi::Error {
    core_err(
        ErrorCode::IndexInvalid,
        "Invalid table arguments. Check the schema and operation options.",
    )
}
#[napi]
impl NativeTable {
    #[napi(factory)]
    pub fn source(
        values: Vec<ClassInstance<NativeExpr>>,
        validity: Vec<ClassInstance<NativeExpr>>,
        nullable: Vec<bool>,
    ) -> Result<Self> {
        if values.len() != nullable.len()
            || validity.len() != nullable.iter().filter(|&&b| b).count()
        {
            return Err(invalid_table());
        }
        let mut validity = validity.into_iter();
        let cols = values
            .into_iter()
            .zip(nullable)
            .map(|(v, n)| Column {
                values: v.inner.clone(),
                valid: if n {
                    Some(validity.next().unwrap().inner.clone())
                } else {
                    None
                },
            })
            .collect();
        Ok(Self {
            plan: Plan::Source(cols),
        })
    }

    #[napi]
    pub fn select(&self, positions: Vec<u32>) -> Self {
        Self {
            plan: Plan::Select(
                Box::new(self.plan.clone()),
                positions.into_iter().map(|n| n as usize).collect(),
            ),
        }
    }

    #[napi]
    pub fn filter(&self, position: u32) -> Self {
        Self {
            plan: Plan::Filter(Box::new(self.plan.clone()), position as usize),
        }
    }

    #[napi]
    pub fn sort(&self, position: u32, descending: bool, nulls_last: bool) -> Self {
        Self {
            plan: Plan::Sort(
                Box::new(self.plan.clone()),
                position as usize,
                descending,
                nulls_last,
            ),
        }
    }

    #[napi]
    pub fn join(
        &self,
        other: &NativeTable,
        left_key: u32,
        right_key: u32,
        left: bool,
        nulls_equal: bool,
    ) -> Self {
        Self {
            plan: Plan::Join(
                Box::new(self.plan.clone()),
                Box::new(other.plan.clone()),
                left_key as usize,
                right_key as usize,
                left,
                nulls_equal,
            ),
        }
    }

    #[napi]
    pub fn group(
        &self,
        key: u32,
        value: u32,
        operation: String,
        drop_nulls: bool,
        pandas: bool,
    ) -> Result<Self> {
        let op = match operation.as_str() {
            "sum" => Aggregate::Sum,
            "mean" => Aggregate::Mean,
            "min" => Aggregate::Min,
            "max" => Aggregate::Max,
            "count" => Aggregate::Count,
            _ => return Err(invalid_table()),
        };
        Ok(Self {
            plan: Plan::Group(
                Box::new(self.plan.clone()),
                key as usize,
                value as usize,
                op,
                drop_nulls,
                if pandas {
                    Flavor::Pandas
                } else {
                    Flavor::Polars
                },
            ),
        })
    }

    #[napi(ts_return_type = "Promise<NativeTableResult>")]
    pub fn compute<'env>(
        &self,
        env: &'env Env,
        context: &NativeContext,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        type Resolve = Box<dyn FnOnce(Env) -> Result<NativeTableResult> + Send>;
        let (deferred, promise) = env.create_deferred::<NativeTableResult, Resolve>()?;
        let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(deferred)));
        let output = slot.clone();
        let id = context.inner.id();
        let done = Box::new(move |result: nmath_core::Result<table::Table>| {
            let Some(deferred) = output.lock().unwrap().take() else {
                return;
            };
            match result {
                Err(e) => deferred.reject(to_napi(e)),
                Ok(t) => deferred.resolve(Box::new(move |_| {
                    let mut values = Vec::new();
                    let mut valid = Vec::new();
                    for c in t.columns {
                        values.push(NativeArray {
                            inner: Some(NdArray::from_dense(id, c.values)),
                        });
                        valid.push(NativeArray {
                            inner: Some(NdArray::from_dense(id, c.valid)),
                        });
                    }
                    Ok(NativeTableResult {
                        values,
                        valid,
                        rows: u32::try_from(t.rows).map_err(|_| invalid_table())?,
                    })
                })),
            }
        });
        if let Err(e) = table::compute(&context.inner, self.plan.clone(), token_of(signal), done)
            && let Some(deferred) = slot.lock().unwrap().take()
        {
            deferred.reject(to_napi(e));
        }
        Ok(promise)
    }
}
