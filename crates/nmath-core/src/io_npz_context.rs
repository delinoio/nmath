//! Context extension; register as a child module of `context` so admission
//! continues through the existing private scheduler.
use super::*;
use crate::io_npz::{self, NamedArray, NpzArrays, NpzCompression};

pub type NpzDone = Box<dyn FnOnce(Result<NpzArrays>) + Send>;

impl Context {
    /// Snapshot all named inputs when the scheduled worker actually starts.
    /// An empty list writes a valid empty NPZ archive.
    pub fn save_npz(
        &self,
        arrays: Vec<(String, NdArray)>,
        path: PathBuf,
        compression: NpzCompression,
        mode: WriteMode,
        token: CancellationToken,
        done: Done,
    ) -> Result<u64> {
        for (_, array) in &arrays {
            self.ensure_owned(Some(array.context_id()))?;
        }
        let work: Work = Box::new(move |env| {
            let metadata_bytes = arrays
                .iter()
                .try_fold(0u64, |n, (name, array)| {
                    n.checked_add(4096 + name.len() as u64 + array.ndim() as u64 * 16)
                })
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::MemoryBudgetExceeded,
                        "NPZ snapshot metadata exceeds the memory budget.",
                    )
                })?;
            let _metadata = env.budget.reserve(metadata_bytes)?;
            let mut snapshots = Vec::with_capacity(arrays.len());
            for (name, array) in arrays {
                env.token.check()?;
                snapshots.push(NamedArray {
                    name,
                    array: array.snapshot(&env.budget)?,
                });
            }
            io_npz::save_npz(
                &path,
                &snapshots,
                compression,
                mode,
                &env.budget,
                &env.token,
            )?;
            Ok(JobOutput::Unit)
        });
        self.submit(
            OperationKind::FileWrite,
            "save_npz",
            Device::Cpu,
            token,
            work,
            done,
        )
    }

    /// The side slot carries the typed archive through the existing scheduler.
    /// Scheduler cancellation/error drops it instead of returning partial data.
    pub fn load_npz(&self, path: PathBuf, token: CancellationToken, done: NpzDone) -> Result<u64> {
        let id = self.id();
        let output = Arc::new(Mutex::new(None));
        let result_slot = output.clone();
        let work: Work = Box::new(move |env| {
            let archive = io_npz::load_npz(&path, &env.budget, &env.token)?.into_context(id);
            *result_slot.lock().unwrap() = Some(archive);
            Ok(JobOutput::Unit)
        });
        let completion: Done = Box::new(move |result| {
            let archive = output.lock().unwrap().take();
            match result {
                Ok(_) => done(archive.ok_or_else(|| {
                    Error::new(ErrorCode::IoFailed, "NPZ execution returned no archive.")
                })),
                Err(error) => {
                    drop(archive);
                    done(Err(error));
                }
            }
        });
        self.submit(
            OperationKind::FileRead,
            "load_npz",
            Device::Cpu,
            token,
            work,
            completion,
        )
    }
}
