//! Explicit execution contexts: admission, FIFO scheduling, budgets, close and
//! metrics.

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crate::{
    array::NdArray,
    buffer::{Buffer, DenseArray},
    error::{Error, ErrorCode, Result},
    expr::{Expr, JobEnv, Value},
    io::{self, WriteMode},
    memory::{Device, MemoryBudget, MemoryUsage},
    random::Generator,
    shape::Selector,
    sync::{CancellationToken, FifoGuard, Ticket},
};

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Finished executions kept for `metrics()`; cumulative counters cover all
/// executions.
const RECENT_EXECUTIONS: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextOptions {
    pub max_concurrent_jobs: usize,
    pub max_queue_length: usize,
    pub cpu_memory_budget: u64,
    pub gpu_memory_budget: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationKind {
    Compute,
    Write,
    Copy,
    FileRead,
    FileWrite,
}

impl OperationKind {
    pub fn id(self) -> &'static str {
        match self {
            OperationKind::Compute => "compute",
            OperationKind::Write => "write",
            OperationKind::Copy => "copy",
            OperationKind::FileRead => "file_read",
            OperationKind::FileWrite => "file_write",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl ExecutionStatus {
    pub fn id(self) -> &'static str {
        match self {
            ExecutionStatus::Queued => "queued",
            ExecutionStatus::Running => "running",
            ExecutionStatus::Succeeded => "succeeded",
            ExecutionStatus::Failed => "failed",
            ExecutionStatus::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExecutionRecord {
    pub id: u64,
    pub kind: OperationKind,
    /// Root operator ID for compute jobs, for example `add` or `matmul`.
    pub operation: &'static str,
    pub device: Device,
    pub status: ExecutionStatus,
    pub queue_time: Duration,
    pub execution_time: Duration,
    queued_at: Instant,
    started_at: Option<Instant>,
}

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub executions: Vec<ExecutionRecord>,
    pub total_submitted: u64,
    pub total_succeeded: u64,
    pub total_failed: u64,
    pub total_cancelled: u64,
    pub total_rejected: u64,
    pub cpu_memory: MemoryUsage,
    pub gpu_memory: MemoryUsage,
}

/// The value a finished job returns.
#[derive(Debug)]
pub enum JobOutput {
    Value(Value),
    Array(NdArray),
    Unit,
}

type Start = Box<dyn FnOnce() + Send>;
type Cancel = Box<dyn FnOnce(Error) + Send>;

struct Queued {
    id: u64,
    token: CancellationToken,
    start: Start,
    cancel: Cancel,
}

#[derive(Default)]
struct Sched {
    running: HashMap<u64, CancellationToken>,
    queue: VecDeque<Queued>,
    closed: bool,
}

#[derive(Default)]
struct MetricsState {
    live: HashMap<u64, ExecutionRecord>,
    recent: VecDeque<ExecutionRecord>,
    submitted: u64,
    succeeded: u64,
    failed: u64,
    cancelled: u64,
    rejected: u64,
}

struct Inner {
    id: u64,
    options: ContextOptions,
    cpu: MemoryBudget,
    gpu: MemoryBudget,
    sched: Mutex<Sched>,
    idle: Condvar,
    metrics: Mutex<MetricsState>,
    next_execution: AtomicU64,
}

/// An explicit execution context. Cloning shares it.
#[derive(Clone)]
pub struct Context {
    inner: Arc<Inner>,
}

/// Work for one job. Runs on a native worker thread after the job obtains a
/// slot.
pub type Work = Box<dyn FnOnce(&mut JobEnv) -> Result<JobOutput> + Send>;
/// Completion callback; called exactly once for an admitted job.
pub type Done = Box<dyn FnOnce(Result<JobOutput>) + Send>;

impl Context {
    pub fn new(options: ContextOptions) -> Result<Context> {
        if options.max_concurrent_jobs == 0 {
            return Err(Error::new(
                ErrorCode::ContextInvalidOptions,
                "maxConcurrentJobs must be at least 1.",
            ));
        }
        Ok(Context {
            inner: Arc::new(Inner {
                id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed),
                options,
                cpu: MemoryBudget::new(Device::Cpu, options.cpu_memory_budget),
                gpu: MemoryBudget::new(Device::Gpu, options.gpu_memory_budget),
                sched: Mutex::new(Sched::default()),
                idle: Condvar::new(),
                metrics: Mutex::new(MetricsState::default()),
                next_execution: AtomicU64::new(1),
            }),
        })
    }

    pub fn id(&self) -> u64 {
        self.inner.id
    }

    pub fn options(&self) -> ContextOptions {
        self.inner.options
    }

    pub fn is_closed(&self) -> bool {
        self.inner.sched.lock().unwrap().closed
    }

    fn ensure_open(&self) -> Result<()> {
        if self.is_closed() {
            Err(Error::new(
                ErrorCode::ContextClosed,
                "The context is closed.",
            ))
        } else {
            Ok(())
        }
    }

    /// Creates an array charged to this context's CPU budget.
    pub fn array(&self, shape: Vec<usize>, buffer: Buffer) -> Result<NdArray> {
        self.ensure_open()?;
        let reservation = self.inner.cpu.reserve(buffer.byte_len())?;
        Ok(NdArray::from_dense(
            self.id(),
            DenseArray::new(shape, buffer, reservation)?,
        ))
    }

    pub fn generator(&self, seed: u128) -> Generator {
        Generator::new(seed)
    }

    fn ensure_owned(&self, context_id: Option<u64>) -> Result<()> {
        match context_id {
            Some(id) if id != self.id() => Err(Error::new(
                ErrorCode::ContextMismatch,
                format!(
                    "The value belongs to context {id}, not context {}.",
                    self.id()
                ),
            )),
            _ => Ok(()),
        }
    }

    /// `expr.compute({ device, signal })`.
    pub fn compute(
        &self,
        expr: &Expr,
        device: Device,
        token: CancellationToken,
        done: Done,
    ) -> Result<u64> {
        self.ensure_owned(expr.context_id())?;
        if device == Device::Gpu {
            expr.check_gpu_support()?;
        }
        let generators = expr.generators();
        let operation = expr.operation_id();
        let expr = expr.clone();
        self.submit_with_work(
            OperationKind::Compute,
            operation,
            device,
            token,
            move || {
                let generator_tickets: Vec<Ticket> =
                    generators.iter().map(Generator::ticket).collect();
                Box::new(move |env| {
                    let _turns: Vec<FifoGuard> = generator_tickets
                        .into_iter()
                        .map(|t| t.wait(&env.token))
                        .collect::<Result<_>>()?;
                    env.freeze(&expr)?;
                    let value = env.eval(&expr)?;
                    env.token.check()?;
                    for (generator, state) in env.take_generator_states() {
                        generator.commit(state);
                    }
                    Ok(JobOutput::Value(value))
                })
            },
            done,
        )
    }

    /// `target[selectors] = value`, serialized per storage in admission order and
    /// atomic.
    pub fn assign(
        &self,
        target: &NdArray,
        selectors: Vec<Selector>,
        value: &Expr,
        token: CancellationToken,
        done: Done,
    ) -> Result<u64> {
        self.ensure_owned(Some(target.context_id()))?;
        self.ensure_owned(value.context_id())?;
        // Like `compute()`, random nodes serialize on their generators in
        // admission order and commit the advanced state only on success.
        let generators = value.generators();
        let target = target.clone();
        let value = value.clone();
        self.submit_with_work(
            OperationKind::Write,
            "assign",
            Device::Cpu,
            token,
            move || {
                let generator_tickets: Vec<Ticket> =
                    generators.iter().map(Generator::ticket).collect();
                let ticket = target.write_ticket();
                Box::new(move |env| {
                    let _turns: Vec<FifoGuard> = generator_tickets
                        .into_iter()
                        .map(|t| t.wait(&env.token))
                        .collect::<Result<_>>()?;
                    let turn = ticket.wait(&env.token)?;
                    env.freeze(&value)?;
                    // Assignment copies values, so view-valued expressions
                    // (basic indexing of an input)
                    // materialize from the frozen snapshot.
                    let dense = env.eval_dense(&value)?;
                    target.assign(&turn, &selectors, &dense, &env.token)?;
                    for (generator, state) in env.take_generator_states() {
                        generator.commit(state);
                    }
                    Ok(JobOutput::Unit)
                })
            },
            done,
        )
    }

    /// Copies an array from another context into this one, charged to this
    /// context.
    pub fn copy_from(&self, source: &NdArray, token: CancellationToken, done: Done) -> Result<u64> {
        let source = source.clone();
        let id = self.id();
        let work: Work = Box::new(move |env| {
            let snap = source.snapshot(&env.budget)?;
            Ok(JobOutput::Array(NdArray::from_dense(id, snap)))
        });
        self.submit(OperationKind::Copy, "copy", Device::Cpu, token, work, done)
    }

    pub fn save_npy(
        &self,
        array: &NdArray,
        path: PathBuf,
        mode: WriteMode,
        token: CancellationToken,
        done: Done,
    ) -> Result<u64> {
        self.ensure_owned(Some(array.context_id()))?;
        let array = array.clone();
        let work: Work = Box::new(move |env| {
            let snap = array.snapshot(&env.budget)?;
            let bytes = io::encode_npy(&snap.shape, &snap.buffer)?;
            io::write_atomic(&path, &bytes, mode, &env.token)?;
            Ok(JobOutput::Unit)
        });
        self.submit(
            OperationKind::FileWrite,
            "save_npy",
            Device::Cpu,
            token,
            work,
            done,
        )
    }

    pub fn load_npy(&self, path: PathBuf, token: CancellationToken, done: Done) -> Result<u64> {
        let id = self.id();
        let work: Work = Box::new(move |env| {
            // The encoded bytes and the decoded buffer are both live while
            // decoding, so each carries its own reservation.
            let (bytes, encoded) = io::read_checked_budgeted(&path, &env.budget, &env.token)?;
            let (shape, buffer, reservation) = io::decode_npy_budgeted(&bytes, &env.budget)?;
            drop(bytes);
            drop(encoded);
            Ok(JobOutput::Array(NdArray::from_dense(
                id,
                DenseArray::new(shape, buffer, reservation)?,
            )))
        });
        self.submit(
            OperationKind::FileRead,
            "load_npy",
            Device::Cpu,
            token,
            work,
            done,
        )
    }

    pub(crate) fn submit(
        &self,
        kind: OperationKind,
        operation: &'static str,
        device: Device,
        token: CancellationToken,
        work: Work,
        done: Done,
    ) -> Result<u64> {
        self.submit_with_work(kind, operation, device, token, || work, done)
    }

    // Build ticketed work only after admission checks, while the scheduler
    // lock fixes the order of ticket creation and execution admission.
    fn submit_with_work(
        &self,
        kind: OperationKind,
        operation: &'static str,
        device: Device,
        token: CancellationToken,
        make_work: impl FnOnce() -> Work,
        done: Done,
    ) -> Result<u64> {
        token.check()?;
        let inner = &self.inner;
        let mut sched = inner.sched.lock().unwrap();
        if sched.closed {
            return Err(Error::new(
                ErrorCode::ContextClosed,
                "The context is closed.",
            ));
        }
        let start_now = sched.running.len() < inner.options.max_concurrent_jobs;
        if !start_now && sched.queue.len() >= inner.options.max_queue_length {
            inner.metrics.lock().unwrap().rejected += 1;
            return Err(Error::new(
                ErrorCode::SchedQueueFull,
                format!(
                    "The queue already holds {} jobs.",
                    inner.options.max_queue_length
                ),
            ));
        }
        let work = make_work();
        let id = inner.next_execution.fetch_add(1, Ordering::Relaxed);
        {
            let mut m = inner.metrics.lock().unwrap();
            m.submitted += 1;
            m.live.insert(
                id,
                ExecutionRecord {
                    id,
                    kind,
                    operation,
                    device,
                    status: ExecutionStatus::Queued,
                    queue_time: Duration::ZERO,
                    execution_time: Duration::ZERO,
                    queued_at: Instant::now(),
                    started_at: None,
                },
            );
        }
        tracing::debug!(
            event = "nmath.job.admitted",
            context = inner.id,
            execution = id,
            kind = kind.id()
        );
        let done = Arc::new(Mutex::new(Some(done)));
        let start: Start = {
            let weak = Arc::downgrade(inner);
            let token = token.clone();
            let done = done.clone();
            Box::new(move || {
                if let Some(inner) = weak.upgrade() {
                    run_job(inner, id, kind, device, token, work, done);
                }
            })
        };
        if start_now {
            sched.running.insert(id, token.clone());
            drop(sched);
            start();
        } else {
            let cancel: Cancel = {
                let done = done.clone();
                Box::new(move |err| {
                    if let Some(done) = done.lock().unwrap().take() {
                        done(Err(err));
                    }
                })
            };
            sched.queue.push_back(Queued {
                id,
                token: token.clone(),
                start,
                cancel,
            });
            drop(sched);
            let weak: Weak<Inner> = Arc::downgrade(inner);
            token.on_cancel(move || {
                if let Some(inner) = weak.upgrade() {
                    cancel_queued(&inner, id);
                }
            });
        }
        Ok(id)
    }

    /// Rejects new work, cancels queued and running work and waits for all jobs
    /// to release their resources.
    pub fn close(&self) {
        let inner = &self.inner;
        let (running, queued): (Vec<CancellationToken>, Vec<Queued>) = {
            let mut sched = inner.sched.lock().unwrap();
            sched.closed = true;
            (
                sched.running.values().cloned().collect(),
                sched.queue.drain(..).collect(),
            )
        };
        // Cancel outside the scheduler lock: a job that started from the queue
        // still has its `cancel_queued` callback registered, and that callback
        // locks the scheduler.
        for token in running {
            token.cancel();
        }
        for job in queued {
            finish_record(inner, job.id, ExecutionStatus::Cancelled);
            (job.cancel)(Error::new(ErrorCode::Cancelled, "The context was closed."));
        }
        let mut sched = inner.sched.lock().unwrap();
        while !sched.running.is_empty() {
            sched = inner.idle.wait(sched).unwrap();
        }
    }

    pub fn metrics(&self) -> Metrics {
        let m = self.inner.metrics.lock().unwrap();
        let now = Instant::now();
        let mut executions: Vec<ExecutionRecord> = m.recent.iter().cloned().collect();
        let mut live: Vec<ExecutionRecord> = m
            .live
            .values()
            .cloned()
            .map(|mut r| {
                match r.started_at {
                    None => r.queue_time = now - r.queued_at,
                    Some(s) => r.execution_time = now - s,
                }
                r
            })
            .collect();
        live.sort_by_key(|r| r.id);
        executions.extend(live);
        Metrics {
            executions,
            total_submitted: m.submitted,
            total_succeeded: m.succeeded,
            total_failed: m.failed,
            total_cancelled: m.cancelled,
            total_rejected: m.rejected,
            cpu_memory: self.inner.cpu.usage(),
            gpu_memory: self.inner.gpu.usage(),
        }
    }
}

fn cancel_queued(inner: &Arc<Inner>, id: u64) {
    let job = {
        let mut sched = inner.sched.lock().unwrap();
        let pos = sched.queue.iter().position(|q| q.id == id);
        pos.and_then(|p| sched.queue.remove(p))
    };
    if let Some(job) = job {
        finish_record(inner, id, ExecutionStatus::Cancelled);
        (job.cancel)(Error::new(
            ErrorCode::Cancelled,
            "The computation was cancelled while queued.",
        ));
    }
}

fn finish_record(inner: &Inner, id: u64, status: ExecutionStatus) {
    let mut m = inner.metrics.lock().unwrap();
    if let Some(mut record) = m.live.remove(&id) {
        let now = Instant::now();
        match record.started_at {
            None => record.queue_time = now - record.queued_at,
            Some(s) => record.execution_time = now - s,
        }
        record.status = status;
        match status {
            ExecutionStatus::Succeeded => m.succeeded += 1,
            ExecutionStatus::Failed => m.failed += 1,
            ExecutionStatus::Cancelled => m.cancelled += 1,
            _ => {}
        }
        m.recent.push_back(record);
        if m.recent.len() > RECENT_EXECUTIONS {
            m.recent.pop_front();
        }
    }
}

type SharedDone = Arc<Mutex<Option<Done>>>;

fn run_job(
    inner: Arc<Inner>,
    id: u64,
    kind: OperationKind,
    device: Device,
    token: CancellationToken,
    work: Work,
    done: SharedDone,
) {
    std::thread::Builder::new()
        .name(format!("nmath-job-{id}"))
        .spawn(move || {
            {
                let mut m = inner.metrics.lock().unwrap();
                if let Some(r) = m.live.get_mut(&id) {
                    let now = Instant::now();
                    r.queue_time = now - r.queued_at;
                    r.started_at = Some(now);
                    r.status = ExecutionStatus::Running;
                }
            }
            tracing::debug!(
                event = "nmath.job.started",
                context = inner.id,
                execution = id
            );
            // Snapshots and returned host arrays use CPU memory even for GPU
            // jobs.
            let mut env = JobEnv::new(device, inner.cpu.clone(), token.clone());
            env.gpu_budget = Some(inner.gpu.clone());
            let mut result = token.check().and_then(|()| work(&mut env));
            drop(env);
            // FileWrite work returns success only after atomic publication. The
            // writer checks cancellation before publication; a late token
            // cannot undo that commit. Reads and computations
            // retain the final check.
            if kind != OperationKind::FileWrite && result.is_ok() && token.is_cancelled() {
                result = Err(Error::new(
                    ErrorCode::Cancelled,
                    "The computation was cancelled.",
                ));
            }
            let status = match &result {
                Ok(_) => ExecutionStatus::Succeeded,
                Err(e) if e.code == ErrorCode::Cancelled => ExecutionStatus::Cancelled,
                Err(_) => ExecutionStatus::Failed,
            };
            finish_record(&inner, id, status);
            tracing::debug!(
                event = "nmath.job.finished",
                context = inner.id,
                execution = id,
                status = status.id()
            );
            if let Some(done) = done.lock().unwrap().take() {
                done(result);
            }
            let next = {
                let mut sched = inner.sched.lock().unwrap();
                sched.running.remove(&id);
                let next = if sched.closed {
                    None
                } else {
                    sched.queue.pop_front()
                };
                if let Some(job) = &next {
                    sched.running.insert(job.id, job.token.clone());
                }
                inner.idle.notify_all();
                next
            };
            if let Some(job) = next {
                (job.start)();
            }
        })
        .expect("failed to spawn an NMath worker thread");
}

#[path = "io_npz_context.rs"]
mod io_npz_context;

#[cfg(test)]
mod admission_tests {
    use std::{sync::mpsc, thread, time::Duration};

    use super::*;
    use crate::{Buffer, Scalar};

    fn context(queue: usize) -> Context {
        Context::new(ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: queue,
            cpu_memory_budget: 1 << 20,
            gpu_memory_budget: 0,
        })
        .unwrap()
    }

    fn completion() -> (Done, mpsc::Receiver<Result<JobOutput>>) {
        let (tx, rx) = mpsc::channel();
        (Box::new(move |result| tx.send(result).unwrap()), rx)
    }

    fn completed(rx: mpsc::Receiver<Result<JobOutput>>) {
        rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    }

    #[test]
    fn ticket_creation_and_admission_cannot_invert() {
        let ctx = context(2);
        let target = ctx.array(vec![1], Buffer::I32(vec![0])).unwrap();
        let (ticket_ready_tx, ticket_ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (first_done, first_rx) = completion();
        let first_ctx = ctx.clone();
        let first_target = target.clone();
        let first = thread::spawn(move || {
            first_ctx.submit_with_work(
                OperationKind::Write,
                "first",
                Device::Cpu,
                CancellationToken::new(),
                move || {
                    let ticket = first_target.write_ticket();
                    ticket_ready_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Box::new(move |env| {
                        let _turn = ticket.wait(&env.token)?;
                        Ok(JobOutput::Unit)
                    })
                },
                first_done,
            )
        });
        ticket_ready_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let (second_started_tx, second_started_rx) = mpsc::channel();
        let (second_done, second_rx) = completion();
        let second_ctx = ctx.clone();
        let second_target = target.clone();
        let second = thread::spawn(move || {
            second_started_tx.send(()).unwrap();
            second_ctx.submit_with_work(
                OperationKind::Write,
                "second",
                Device::Cpu,
                CancellationToken::new(),
                move || {
                    let ticket = second_target.write_ticket();
                    Box::new(move |env| {
                        let _turn = ticket.wait(&env.token)?;
                        Ok(JobOutput::Unit)
                    })
                },
                second_done,
            )
        });
        second_started_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        release_tx.send(()).unwrap();
        let first_id = first.join().unwrap().unwrap();
        let second_id = second.join().unwrap().unwrap();
        assert!(first_id < second_id);
        completed(first_rx);
        completed(second_rx);
        ctx.close();
    }

    #[test]
    fn assignment_rejection_and_cancellation_release_tickets() {
        let ctx = context(1);
        let target = ctx.array(vec![1], Buffer::I32(vec![0])).unwrap();
        let other = ctx.array(vec![1], Buffer::I32(vec![0])).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (block_done, block_rx) = completion();
        ctx.submit(
            OperationKind::Compute,
            "blocker",
            Device::Cpu,
            CancellationToken::new(),
            Box::new(move |_| {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(JobOutput::Unit)
            }),
            block_done,
        )
        .unwrap();
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let cancelled = CancellationToken::new();
        let (cancel_done, cancel_rx) = completion();
        ctx.assign(
            &target,
            vec![],
            &Expr::scalar(Scalar::Int(1)),
            cancelled.clone(),
            cancel_done,
        )
        .unwrap();
        let (reject_done, _) = completion();
        assert_eq!(
            ctx.assign(
                &target,
                vec![],
                &Expr::scalar(Scalar::Int(2)),
                CancellationToken::new(),
                reject_done,
            )
            .unwrap_err()
            .code,
            ErrorCode::SchedQueueFull
        );
        cancelled.cancel();
        assert_eq!(
            cancel_rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        let (next_done, next_rx) = completion();
        ctx.assign(
            &target,
            vec![],
            &Expr::scalar(Scalar::Int(3)),
            CancellationToken::new(),
            next_done,
        )
        .unwrap();
        release_tx.send(()).unwrap();
        completed(block_rx);
        completed(next_rx);
        let (other_done, other_rx) = completion();
        ctx.assign(
            &other,
            vec![],
            &Expr::scalar(Scalar::Int(4)),
            CancellationToken::new(),
            other_done,
        )
        .unwrap();
        completed(other_rx);
        ctx.close();
        assert_eq!(target.to_buffer(), Buffer::I32(vec![3]));
        assert_eq!(other.to_buffer(), Buffer::I32(vec![4]));
    }

    #[test]
    fn shared_generator_tickets_preserve_commit_on_success() {
        let ctx = context(2);
        let generator = ctx.generator(17);
        let initial = generator.current();
        let expr = Expr::random(ctx.id(), &generator, vec![1]);
        let target = ctx.array(vec![1], Buffer::F64(vec![0.0])).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (block_done, block_rx) = completion();
        ctx.submit(
            OperationKind::Compute,
            "blocker",
            Device::Cpu,
            CancellationToken::new(),
            Box::new(move |_| {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(JobOutput::Unit)
            }),
            block_done,
        )
        .unwrap();
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let cancelled = CancellationToken::new();
        let (cancel_done, cancel_rx) = completion();
        ctx.compute(&expr, Device::Cpu, cancelled.clone(), cancel_done)
            .unwrap();
        cancelled.cancel();
        assert_eq!(
            cancel_rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(generator.current(), initial);
        let (write_done, write_rx) = completion();
        ctx.assign(&target, vec![], &expr, CancellationToken::new(), write_done)
            .unwrap();
        release_tx.send(()).unwrap();
        completed(block_rx);
        completed(write_rx);
        let committed = generator.current();
        assert_ne!(committed, initial);
        let (bad_done, bad_rx) = completion();
        let bad_expr = Expr::random(ctx.id(), &generator, vec![2]);
        ctx.assign(
            &target,
            vec![],
            &bad_expr,
            CancellationToken::new(),
            bad_done,
        )
        .unwrap();
        assert_eq!(
            bad_rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err()
                .code,
            ErrorCode::ShapeBroadcast
        );
        assert_eq!(generator.current(), committed);
        let (compute_done, compute_rx) = completion();
        ctx.compute(&expr, Device::Cpu, CancellationToken::new(), compute_done)
            .unwrap();
        completed(compute_rx);
        assert_ne!(generator.current(), committed);
        ctx.close();
    }

    #[test]
    fn sequential_assignments_publish_in_order() {
        let ctx = context(2);
        let target = ctx.array(vec![1], Buffer::I32(vec![0])).unwrap();
        for value in [1, 2, 3] {
            let (done, rx) = completion();
            ctx.assign(
                &target,
                vec![],
                &Expr::scalar(Scalar::Int(value)),
                CancellationToken::new(),
                done,
            )
            .unwrap();
            completed(rx);
            assert_eq!(target.to_buffer(), Buffer::I32(vec![value as i32]));
        }
        ctx.close();
    }

    #[test]
    fn close_abandons_queued_generator_ticket() {
        let ctx = context(1);
        let generator = ctx.generator(23);
        let initial = generator.current();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (block_done, block_rx) = completion();
        ctx.submit(
            OperationKind::Compute,
            "blocker",
            Device::Cpu,
            CancellationToken::new(),
            Box::new(move |_| {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(JobOutput::Unit)
            }),
            block_done,
        )
        .unwrap();
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let (queued_done, queued_rx) = completion();
        ctx.compute(
            &Expr::random(ctx.id(), &generator, vec![1]),
            Device::Cpu,
            CancellationToken::new(),
            queued_done,
        )
        .unwrap();
        let closing = ctx.clone();
        let close_thread = thread::spawn(move || closing.close());
        assert_eq!(
            queued_rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        release_tx.send(()).unwrap();
        assert_eq!(
            block_rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap_err().code,
            ErrorCode::Cancelled
        );
        close_thread.join().unwrap();
        assert_eq!(generator.current(), initial);
        let ticket = generator.ticket();
        drop(ticket.wait(&CancellationToken::new()).unwrap());
    }
}

#[cfg(test)]
mod publication_tests {
    use std::{fs, sync::mpsc, time::Duration};

    use super::*;

    fn context() -> Context {
        Context::new(ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: 4,
            cpu_memory_budget: 128 << 20,
            gpu_memory_budget: 0,
        })
        .unwrap()
    }

    fn completion() -> (Done, mpsc::Receiver<Result<JobOutput>>) {
        let (tx, rx) = mpsc::channel();
        (Box::new(move |result| tx.send(result).unwrap()), rx)
    }

    #[test]
    fn published_file_success_survives_late_cancellation() {
        for mode in [WriteMode::CreateNew, WriteMode::Overwrite] {
            let ctx = context();
            let dir = std::env::temp_dir().join(format!(
                "nmath-publication-{}-{}",
                std::process::id(),
                ctx.id()
            ));
            fs::create_dir_all(&dir).unwrap();
            let path = dir.join("output.npy");
            if mode == WriteMode::Overwrite {
                fs::write(&path, b"original").unwrap();
            }
            let output = path.clone();
            let token = CancellationToken::new();
            let (done, rx) = completion();
            ctx.submit(
                OperationKind::FileWrite,
                "publication_test",
                Device::Cpu,
                token,
                Box::new(move |env| {
                    io::write_atomic(&output, b"committed", mode, &env.token)?;
                    // Deterministically cancel between publication and the
                    // scheduler's completion check, without
                    // sleeps or a timing-dependent race.
                    env.token.cancel();
                    Ok(JobOutput::Unit)
                }),
                done,
            )
            .unwrap();
            assert!(matches!(
                rx.recv_timeout(Duration::from_secs(5)).unwrap(),
                Ok(JobOutput::Unit)
            ));
            ctx.close();
            assert_eq!(fs::read(&path).unwrap(), b"committed");
            assert_eq!(ctx.metrics().total_succeeded, 1);
            assert_eq!(ctx.metrics().total_cancelled, 0);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn published_npz_success_survives_late_cancellation() {
        use crate::io_npz::{self, NpzCompression};
        for compression in [NpzCompression::Stored, NpzCompression::Deflated] {
            let ctx = context();
            let dir = std::env::temp_dir().join(format!(
                "nmath-npz-publication-{}-{}",
                std::process::id(),
                ctx.id()
            ));
            fs::create_dir_all(&dir).unwrap();
            let path = dir.join("output.npz");
            fs::write(&path, b"original").unwrap();
            let output = path.clone();
            let (done, rx) = completion();
            ctx.submit(
                OperationKind::FileWrite,
                "npz_publication_test",
                Device::Cpu,
                CancellationToken::new(),
                Box::new(move |env| {
                    io_npz::save_npz(
                        &output,
                        &[],
                        compression,
                        WriteMode::Overwrite,
                        &env.budget,
                        &env.token,
                    )?;
                    env.token.cancel();
                    Ok(JobOutput::Unit)
                }),
                done,
            )
            .unwrap();
            assert!(matches!(
                rx.recv_timeout(Duration::from_secs(5)).unwrap(),
                Ok(JobOutput::Unit)
            ));
            ctx.close();
            assert!(
                io_npz::load_npz(&path, &ctx.inner.cpu, &CancellationToken::new())
                    .unwrap()
                    .arrays
                    .is_empty()
            );
            assert_eq!(ctx.metrics().total_succeeded, 1);
            assert_eq!(ctx.metrics().total_cancelled, 0);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn prepublication_and_queued_cancellation_preserve_output() {
        let ctx = context();
        let dir = std::env::temp_dir().join(format!(
            "nmath-prepublication-{}-{}",
            std::process::id(),
            ctx.id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("output.npy");
        fs::write(&path, b"original").unwrap();
        let output = path.clone();
        let (done, rx) = completion();
        ctx.submit(
            OperationKind::FileWrite,
            "prepublication_test",
            Device::Cpu,
            CancellationToken::new(),
            Box::new(move |env| {
                env.token.cancel();
                io::write_atomic(&output, b"replacement", WriteMode::Overwrite, &env.token)?;
                Ok(JobOutput::Unit)
            }),
            done,
        )
        .unwrap();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done, blocker) = completion();
        ctx.submit(
            OperationKind::FileRead,
            "blocker",
            Device::Cpu,
            CancellationToken::new(),
            Box::new(move |_| {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(JobOutput::Unit)
            }),
            done,
        )
        .unwrap();
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let array = ctx.array(vec![1], crate::Buffer::I32(vec![42])).unwrap();
        let token = CancellationToken::new();
        let (done, rx) = completion();
        ctx.save_npy(
            &array,
            path.clone(),
            WriteMode::Overwrite,
            token.clone(),
            done,
        )
        .unwrap();
        token.cancel();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        release_tx.send(()).unwrap();
        blocker
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        ctx.close();
        assert_eq!(fs::read(path).unwrap(), b"original");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn late_cancellation_still_discards_uncommitted_results() {
        let ctx = context();
        let (done, rx) = completion();
        ctx.submit(
            OperationKind::FileRead,
            "cancelled_read",
            Device::Cpu,
            CancellationToken::new(),
            Box::new(|env| {
                env.token.cancel();
                Ok(JobOutput::Unit)
            }),
            done,
        )
        .unwrap();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        ctx.close();
        assert_eq!(ctx.metrics().total_cancelled, 1);
    }
}
