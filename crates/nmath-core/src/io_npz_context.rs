//! Context extension; register as a child module of `context` so admission
//! continues through the existing private scheduler.
use super::*;
use crate::{
    io_npz::{self, NamedArray, NpzArrays, NpzCompression},
    table::SNAPSHOT_GATE,
};

fn snapshot_npz_inputs(
    arrays: Vec<(String, NdArray)>,
    budget: &MemoryBudget,
    token: &CancellationToken,
    mut between_members: impl FnMut(),
) -> Result<Vec<NamedArray>> {
    let mut snapshots = Vec::with_capacity(arrays.len());
    let _gate = SNAPSHOT_GATE.lock().unwrap();
    token.check()?;
    for (name, array) in arrays {
        token.check()?;
        let offset_bytes = array
            .size()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MemoryBudgetExceeded,
                    "NPZ snapshot offsets exceed the memory budget.",
                )
            })?;
        let offsets = budget.reserve(offset_bytes)?;
        let snapshot = array.snapshot(budget)?;
        drop(offsets);
        snapshots.push(NamedArray {
            name,
            array: snapshot,
        });
        between_members();
    }
    Ok(snapshots)
}

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
            let snapshots = snapshot_npz_inputs(arrays, &env.budget, &env.token, || {})?;
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

#[cfg(test)]
mod snapshot_tests {
    use std::{sync::mpsc, time::Duration};

    use super::*;
    use crate::{buffer::Buffer, memory::Reservation};

    fn array(values: Vec<u8>) -> NdArray {
        let len = values.len();
        NdArray::from_dense(
            1,
            DenseArray::new(vec![len], Buffer::U8(values), Reservation::none()).unwrap(),
        )
    }

    #[test]
    fn aliases_share_one_snapshot_gate_and_release_it() {
        let input = array(vec![7]);
        let budget = MemoryBudget::new(Device::Cpu, 1024);
        let mut seen = 0;
        let (started_tx, started_rx) = mpsc::channel();
        let (written_tx, written_rx) = mpsc::channel();
        let writer = input.clone();
        let mut handle = None;
        let snapshots = snapshot_npz_inputs(
            vec![
                ("first".into(), input.clone()),
                ("second".into(), input.clone()),
            ],
            &budget,
            &CancellationToken::new(),
            || {
                assert!(SNAPSHOT_GATE.try_lock().is_err());
                seen += 1;
                if seen == 1 {
                    let writer = writer.clone();
                    let started_tx = started_tx.clone();
                    let written_tx = written_tx.clone();
                    handle = Some(std::thread::spawn(move || {
                        let token = CancellationToken::new();
                        let turn = writer.write_ticket().wait(&token).unwrap();
                        let value =
                            DenseArray::new(vec![1], Buffer::U8(vec![9]), Reservation::none())
                                .unwrap();
                        started_tx.send(()).unwrap();
                        writer.assign(&turn, &[], &value, &token).unwrap();
                        written_tx.send(()).unwrap();
                    }));
                    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    assert!(written_rx.recv_timeout(Duration::from_millis(50)).is_err());
                }
            },
        )
        .unwrap();
        assert_eq!(seen, 2);
        assert_eq!(snapshots[0].array.buffer, snapshots[1].array.buffer);
        written_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        handle.unwrap().join().unwrap();
        assert_eq!(input.to_buffer(), Buffer::U8(vec![9]));
        drop(SNAPSHOT_GATE.lock().unwrap());
        drop(snapshots);
        assert_eq!(budget.usage().used, 0);
    }

    #[test]
    fn offset_scratch_overlaps_payload_and_prior_members() {
        let budget = MemoryBudget::new(Device::Cpu, 10_000);
        let metadata = budget.reserve(100).unwrap();
        let snapshots = snapshot_npz_inputs(
            vec![
                ("first".into(), array(vec![1; 4])),
                ("second".into(), array(vec![2; 1024])),
            ],
            &budget,
            &CancellationToken::new(),
            || {},
        )
        .unwrap();
        assert_eq!(
            budget.usage().peak,
            100 + 4 + 1024 * (std::mem::size_of::<usize>() as u64 + 1)
        );
        assert_eq!(budget.usage().used, 100 + 4 + 1024);
        drop(snapshots);
        drop(metadata);
        assert_eq!(budget.usage().used, 0);
    }

    #[test]
    fn later_offset_failure_and_cancellation_release_gate_and_reservations() {
        let budget = MemoryBudget::new(Device::Cpu, 200);
        let metadata = budget.reserve(100).unwrap();
        let failed = snapshot_npz_inputs(
            vec![
                ("first".into(), array(vec![1; 4])),
                ("second".into(), array(vec![2; 1024])),
            ],
            &budget,
            &CancellationToken::new(),
            || {},
        );
        assert_eq!(failed.unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
        assert_eq!(budget.usage().used, 100);
        drop(SNAPSHOT_GATE.lock().unwrap());
        drop(metadata);
        assert_eq!(budget.usage().used, 0);

        let cancelled = CancellationToken::new();
        let result = snapshot_npz_inputs(
            vec![
                ("first".into(), array(vec![1])),
                ("second".into(), array(vec![2])),
            ],
            &budget,
            &cancelled,
            || cancelled.cancel(),
        );
        assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
        assert_eq!(budget.usage().used, 0);
        drop(SNAPSHOT_GATE.lock().unwrap());
    }
}
