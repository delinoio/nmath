//! Cancellation tokens and request-ordered locks.

use std::{
    collections::BTreeSet,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::error::{Error, ErrorCode, Result};

type Callback = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct TokenInner {
    cancelled: AtomicBool,
    callbacks: Mutex<Vec<Callback>>,
}

/// Cooperative cancellation shared by the caller and the job.
#[derive(Clone, Default)]
pub struct CancellationToken {
    inner: Arc<TokenInner>,
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        if self.inner.cancelled.swap(true, Ordering::SeqCst) {
            return;
        }
        let callbacks = std::mem::take(&mut *self.inner.callbacks.lock().unwrap());
        for callback in callbacks {
            callback();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Returns `NMATH_CANCELLED` once cancellation was requested.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::new(
                ErrorCode::Cancelled,
                "The computation was cancelled.",
            ))
        } else {
            Ok(())
        }
    }

    /// Runs `callback` on cancellation, immediately if already cancelled.
    pub fn on_cancel(&self, callback: impl FnOnce() + Send + 'static) {
        let mut callbacks = self.inner.callbacks.lock().unwrap();
        if self.is_cancelled() {
            drop(callbacks);
            callback();
        } else {
            callbacks.push(Box::new(callback));
        }
    }
}

#[derive(Default)]
struct FifoState {
    next_ticket: u64,
    serving: u64,
    abandoned: BTreeSet<u64>,
}

#[derive(Default)]
struct FifoInner {
    state: Mutex<FifoState>,
    cv: Condvar,
}

impl FifoInner {
    fn advance(state: &mut FifoState) {
        state.serving += 1;
        while state.abandoned.remove(&state.serving) {
            state.serving += 1;
        }
    }
}

/// A lock granted in ticket (request) order. Used to serialize writes to one
/// storage and jobs that share a random generator.
#[derive(Clone, Default)]
pub struct FifoLock {
    inner: Arc<FifoInner>,
}

impl FifoLock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes a place in line. Call at request time so order follows requests.
    pub fn ticket(&self) -> Ticket {
        let mut state = self.inner.state.lock().unwrap();
        let number = state.next_ticket;
        state.next_ticket += 1;
        Ticket {
            inner: Some(self.inner.clone()),
            number,
        }
    }

    /// Takes a ticket and waits for it.
    pub fn lock(&self, token: &CancellationToken) -> Result<FifoGuard> {
        self.ticket().wait(token)
    }
}

/// A place in a [`FifoLock`] line. Dropping an unwaited ticket gives up the
/// place.
pub struct Ticket {
    inner: Option<Arc<FifoInner>>,
    number: u64,
}

impl Ticket {
    /// Waits for this ticket's turn. Waiting remains cancellable.
    pub fn wait(mut self, token: &CancellationToken) -> Result<FifoGuard> {
        let inner = self.inner.take().unwrap();
        {
            let waker = inner.clone();
            token.on_cancel(move || {
                let _guard = waker.state.lock().unwrap();
                waker.cv.notify_all();
            });
        }
        let mut state = inner.state.lock().unwrap();
        loop {
            if state.serving == self.number {
                drop(state);
                return Ok(FifoGuard { inner });
            }
            if token.is_cancelled() {
                give_up(&mut state, self.number);
                inner.cv.notify_all();
                return Err(Error::new(
                    ErrorCode::Cancelled,
                    "The waiting operation was cancelled.",
                ));
            }
            state = inner.cv.wait(state).unwrap();
        }
    }
}

fn give_up(state: &mut FifoState, number: u64) {
    if state.serving == number {
        FifoInner::advance(state);
    } else {
        state.abandoned.insert(number);
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let mut state = inner.state.lock().unwrap();
            give_up(&mut state, self.number);
            inner.cv.notify_all();
        }
    }
}

/// Holds a [`FifoLock`] until dropped.
pub struct FifoGuard {
    inner: Arc<FifoInner>,
}

impl Drop for FifoGuard {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock().unwrap();
        FifoInner::advance(&mut state);
        self.inner.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use super::*;

    #[test]
    fn grants_in_ticket_order() {
        let lock = FifoLock::new();
        let first = lock.ticket();
        let second = lock.ticket();
        let third = lock.ticket();
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();
        for (name, ticket) in [(3, third), (2, second)] {
            let tx = tx.clone();
            handles.push(thread::spawn(move || {
                let _g = ticket.wait(&CancellationToken::new()).unwrap();
                tx.send(name).unwrap();
            }));
        }
        thread::sleep(Duration::from_millis(20));
        let guard = first.wait(&CancellationToken::new()).unwrap();
        assert!(rx.try_recv().is_err());
        drop(guard);
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(rx.iter().take(2).collect::<Vec<_>>(), vec![2, 3]);
    }

    #[test]
    fn cancelled_waiter_gives_up_its_place() {
        let lock = FifoLock::new();
        let held = lock.lock(&CancellationToken::new()).unwrap();
        let waiting = lock.ticket();
        let after = lock.ticket();
        let token = CancellationToken::new();
        let t2 = token.clone();
        let h = thread::spawn(move || waiting.wait(&t2).err().map(|e| e.code));
        thread::sleep(Duration::from_millis(20));
        token.cancel();
        assert_eq!(h.join().unwrap(), Some(ErrorCode::Cancelled));
        drop(held);
        assert!(after.wait(&CancellationToken::new()).is_ok());
    }

    #[test]
    fn dropped_ticket_does_not_block() {
        let lock = FifoLock::new();
        drop(lock.ticket());
        assert!(lock.lock(&CancellationToken::new()).is_ok());
    }
}
