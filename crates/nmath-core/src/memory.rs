//! Per-context, per-device memory accounting.

use std::sync::{Arc, Mutex};

use crate::error::{Error, ErrorCode, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Gpu,
}

impl Device {
    pub fn id(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        }
    }

    pub fn from_id(id: &str) -> Result<Device> {
        match id {
            "cpu" => Ok(Device::Cpu),
            "gpu" => Ok(Device::Gpu),
            other => Err(Error::new(
                ErrorCode::ContextInvalidOptions,
                format!("Unknown device '{other}'. Use 'cpu' or 'gpu'."),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    pub limit: u64,
    pub used: u64,
    pub peak: u64,
    pub cumulative_reserved: u64,
    pub rejected: u64,
}

#[derive(Debug)]
struct BudgetState {
    usage: MemoryUsage,
}

/// A memory budget. Reservations fail before allocation when they do not fit
/// and are never retried.
#[derive(Clone, Debug)]
pub struct MemoryBudget {
    device: Device,
    state: Arc<Mutex<BudgetState>>,
}

impl MemoryBudget {
    pub fn new(device: Device, limit: u64) -> Self {
        Self {
            device,
            state: Arc::new(Mutex::new(BudgetState {
                usage: MemoryUsage {
                    limit,
                    ..MemoryUsage::default()
                },
            })),
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn usage(&self) -> MemoryUsage {
        self.state.lock().unwrap().usage
    }

    pub fn reserve(&self, bytes: u64) -> Result<Reservation> {
        let mut state = self.state.lock().unwrap();
        let usage = &mut state.usage;
        if usage.used.saturating_add(bytes) > usage.limit {
            usage.rejected += 1;
            tracing::debug!(
                event = "nmath.memory.rejected",
                device = self.device.id(),
                requested = bytes,
                used = usage.used,
                limit = usage.limit
            );
            return Err(Error::new(
                ErrorCode::MemoryBudgetExceeded,
                format!(
                    "Reserving {bytes} bytes on {} would exceed the budget ({} of {} bytes in \
                     use).",
                    self.device.id(),
                    usage.used,
                    usage.limit
                ),
            ));
        }
        usage.used += bytes;
        usage.cumulative_reserved += bytes;
        usage.peak = usage.peak.max(usage.used);
        Ok(Reservation {
            budget: Some(self.clone()),
            bytes,
        })
    }
}

/// RAII charge against a budget. Released on drop.
#[derive(Debug)]
pub struct Reservation {
    budget: Option<MemoryBudget>,
    bytes: u64,
}

impl Reservation {
    /// A reservation that charges nothing, used for values outside any context.
    pub fn none() -> Self {
        Self {
            budget: None,
            bytes: 0,
        }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if let Some(budget) = &self.budget {
            let mut state = budget.state.lock().unwrap();
            state.usage.used -= self.bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_before_allocation_and_releases_on_drop() {
        let budget = MemoryBudget::new(Device::Cpu, 100);
        let a = budget.reserve(60).unwrap();
        let err = budget.reserve(41).unwrap_err();
        assert_eq!(err.code, ErrorCode::MemoryBudgetExceeded);
        assert_eq!(budget.usage().used, 60);
        drop(a);
        let usage = budget.usage();
        assert_eq!(usage.used, 0);
        assert_eq!(usage.peak, 60);
        assert_eq!(usage.rejected, 1);
        budget.reserve(100).unwrap();
    }

    #[test]
    fn zero_budget_allows_only_empty_reservations() {
        let budget = MemoryBudget::new(Device::Gpu, 0);
        assert!(budget.reserve(0).is_ok());
        assert!(budget.reserve(1).is_err());
    }
}
