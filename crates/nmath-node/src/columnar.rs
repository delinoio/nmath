// SPDX-License-Identifier: Apache-2.0
use nmath_core::io_columnar::{self, FileTable, Format, InputColumn};

use super::*;
fn format(value: &str) -> Result<Format> {
    match value {
        "arrow" => Ok(Format::ArrowIpc),
        "parquet" => Ok(Format::Parquet),
        _ => Err(core_err(ErrorCode::IoFormat, "Choose arrow or parquet.")),
    }
}
#[napi(object, object_from_js = false)]
pub struct NativeColumnarResult {
    pub names: Vec<String>,
    pub values: Vec<NativeArray>,
    pub valid: Vec<NativeArray>,
    pub nullable: Vec<bool>,
    pub rows: u32,
}
#[napi]
impl NativeContext {
    #[napi(ts_return_type = "Promise<NativeColumnarResult>")]
    pub fn read_columnar<'env>(
        &self,
        env: &'env Env,
        path: String,
        file_format: String,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        type Resolve = Box<dyn FnOnce(Env) -> Result<NativeColumnarResult> + Send>;
        let (deferred, promise) = env.create_deferred::<NativeColumnarResult, Resolve>()?;
        let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(deferred)));
        let output = slot.clone();
        let id = self.inner.id();
        let done = Box::new(move |result: nmath_core::Result<FileTable>| {
            let Some(deferred) = output.lock().unwrap().take() else {
                return;
            };
            match result {
                Err(e) => deferred.reject(to_napi(e)),
                Ok(t) => deferred.resolve(Box::new(move |_| {
                    let rows = u32::try_from(t.table.rows).map_err(|_| {
                        core_err(
                            ErrorCode::IoFormat,
                            "The table has too many rows for this binding.",
                        )
                    })?;
                    let names = t.schema.fields().iter().map(|f| f.name().clone()).collect();
                    let nullable = t.schema.fields().iter().map(|f| f.is_nullable()).collect();
                    let mut values = Vec::new();
                    let mut valid = Vec::new();
                    for c in t.table.columns {
                        values.push(NativeArray {
                            inner: Some(NdArray::from_dense(id, c.values)),
                        });
                        valid.push(NativeArray {
                            inner: Some(NdArray::from_dense(id, c.valid)),
                        });
                    }
                    Ok(NativeColumnarResult {
                        names,
                        values,
                        valid,
                        nullable,
                        rows,
                    })
                })),
            }
        });
        let admission = format(&file_format).and_then(|f| {
            io_columnar::schedule_read(&self.inner, PathBuf::from(path), f, token_of(signal), done)
                .map_err(to_napi)
        });
        if let Err(e) = admission
            && let Some(deferred) = slot.lock().unwrap().take()
        {
            deferred.reject(e);
        }
        Ok(promise)
    }

    #[napi(ts_return_type = "Promise<void>")]
    pub fn write_columnar<'env>(
        &self,
        env: &'env Env,
        path: String,
        file_format: String,
        names: Vec<String>,
        values: Vec<ClassInstance<NativeExpr>>,
        validity: Vec<ClassInstance<NativeExpr>>,
        nullable: Vec<bool>,
        overwrite: bool,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        type Resolve = Box<dyn FnOnce(Env) -> Result<()> + Send>;
        let (deferred, promise) = env.create_deferred::<(), Resolve>()?;
        let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(deferred)));
        let output = slot.clone();
        let done = Box::new(move |result: nmath_core::Result<JobOutput>| {
            if let Some(deferred) = output.lock().unwrap().take() {
                match result {
                    Ok(_) => deferred.resolve(Box::new(|_| Ok(()))),
                    Err(e) => deferred.reject(to_napi(e)),
                }
            }
        });
        let admission = (|| {
            let f = format(&file_format)?;
            if names.len() != values.len()
                || nullable.len() != values.len()
                || validity.len() != nullable.iter().filter(|&&n| n).count()
            {
                return Err(core_err(
                    ErrorCode::IoFormat,
                    "Column names, values and masks must match.",
                ));
            }
            let mut validity = validity.into_iter();
            let columns = names
                .into_iter()
                .zip(values)
                .zip(nullable)
                .map(|((name, v), n)| InputColumn {
                    name,
                    values: v.inner.clone(),
                    valid: if n {
                        Some(validity.next().unwrap().inner.clone())
                    } else {
                        None
                    },
                })
                .collect();
            io_columnar::schedule_write(
                &self.inner,
                columns,
                PathBuf::from(path),
                f,
                if overwrite {
                    WriteMode::Overwrite
                } else {
                    WriteMode::CreateNew
                },
                token_of(signal),
                done,
            )
            .map_err(to_napi)
        })();
        if let Err(e) = admission
            && let Some(deferred) = slot.lock().unwrap().take()
        {
            deferred.reject(e);
        }
        Ok(promise)
    }
}
