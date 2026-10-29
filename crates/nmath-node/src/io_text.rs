// SPDX-License-Identifier: Apache-2.0
use nmath_core::io_text::{Format, NamedTable};

use super::*;
fn format(value: &str) -> Result<Format> {
    match value {
        "csv" => Ok(Format::Csv),
        "json" => Ok(Format::Json),
        _ => Err(core_err(ErrorCode::IoFormat, "Select csv or json.")),
    }
}
#[napi(object, object_from_js = false)]
pub struct NativeTextTable {
    pub names: Vec<String>,
    pub values: Vec<NativeArray>,
    pub valid: Vec<NativeArray>,
}
// Keep schema bookkeeping charged until N-API finishes constructing the JS
// object.
struct TextResult(NamedTable, u64);
impl ToNapiValue for TextResult {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        mut val: Self,
    ) -> Result<napi::sys::napi_value> {
        let mut values = Vec::with_capacity(val.0.table.columns.len());
        let mut valid = Vec::with_capacity(val.0.table.columns.len());
        for c in std::mem::take(&mut val.0.table.columns) {
            values.push(NativeArray {
                inner: Some(NdArray::from_dense(val.1, c.values)),
            });
            valid.push(NativeArray {
                inner: Some(NdArray::from_dense(val.1, c.valid)),
            });
        }
        let object = NativeTextTable {
            names: std::mem::take(&mut val.0.names),
            values,
            valid,
        };
        // SAFETY: forwards to the generated object conversion in the same
        // environment.
        let result = unsafe { NativeTextTable::to_napi_value(env, object) };
        drop(val);
        result
    }
}
#[napi]
impl NativeContext {
    #[napi(ts_return_type = "Promise<NativeTextTable>")]
    pub fn read_text<'env>(
        &self,
        env: &'env Env,
        path: String,
        kind: String,
        names: Vec<String>,
        dtypes: Vec<String>,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        if names.len() != dtypes.len() {
            return Err(core_err(
                ErrorCode::IoFormat,
                "Schema names and dtypes must match.",
            ));
        }
        let kind = format(&kind)?;
        let schema = names
            .into_iter()
            .zip(dtypes)
            .map(|(n, d)| Ok((n, DType::from_id(&d).map_err(to_napi)?)))
            .collect::<Result<Vec<_>>>()?;
        type Resolve = Box<dyn FnOnce(Env) -> Result<TextResult> + Send>;
        let (deferred, promise) = env.create_deferred::<TextResult, Resolve>()?;
        let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(deferred)));
        let output = slot.clone();
        let id = self.inner.id();
        let done = Box::new(move |result: nmath_core::Result<NamedTable>| {
            let Some(deferred) = output.lock().unwrap().take() else {
                return;
            };
            match result {
                Err(e) => deferred.reject(to_napi(e)),
                Ok(t) => deferred.resolve(Box::new(move |_| Ok(TextResult(t, id)))),
            }
        });
        if let Err(e) = self
            .inner
            .read_text(path.into(), kind, schema, token_of(signal), done)
            && let Some(deferred) = slot.lock().unwrap().take()
        {
            deferred.reject(to_napi(e));
        }
        Ok(promise)
    }

    #[napi(ts_return_type = "Promise<void>")]
    pub fn write_text<'env>(
        &self,
        env: &'env Env,
        path: String,
        kind: String,
        names: Vec<String>,
        values: Vec<ClassInstance<NativeExpr>>,
        validity: Vec<ClassInstance<NativeExpr>>,
        nullable: Vec<bool>,
        overwrite: bool,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        let kind = format(&kind)?;
        if names.len() != values.len()
            || names.len() != nullable.len()
            || validity.len() != nullable.iter().filter(|&&n| n).count()
        {
            return Err(core_err(ErrorCode::IoFormat, "Invalid table schema."));
        }
        let mut masks = validity.into_iter();
        let columns = names
            .into_iter()
            .zip(values)
            .zip(nullable)
            .map(|((n, v), nullable)| {
                Ok((
                    n,
                    v.inner.clone(),
                    if nullable {
                        Some(masks.next().unwrap().inner.clone())
                    } else {
                        None
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let token = token_of(signal);
        let mode = if overwrite {
            WriteMode::Overwrite
        } else {
            WriteMode::CreateNew
        };
        self.promise(env, move |ctx, done| {
            ctx.write_text(path.into(), kind, columns, mode, token, done)
        })
    }
}
