//! Stable error codes shared by every NMath layer.

use std::fmt;

/// Stable public error codes. See `docs/crates-nmath-runtime-contract.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    ContextInvalidOptions,
    ContextMismatch,
    ContextClosed,
    Disposed,
    SchedQueueFull,
    Cancelled,
    MemoryBudgetExceeded,
    ShapeBroadcast,
    ShapeMismatch,
    DTypeUnsupported,
    IndexOutOfBounds,
    IndexInvalid,
    GpuUnavailable,
    GpuUnsupportedOperation,
    GpuDeviceLost,
    GpuOutOfMemory,
    IoExists,
    IoFormat,
    IoModified,
    IoFailed,
    NativeLoad,
}

impl ErrorCode {
    pub const ALL: [ErrorCode; 21] = [
        ErrorCode::ContextInvalidOptions,
        ErrorCode::ContextMismatch,
        ErrorCode::ContextClosed,
        ErrorCode::Disposed,
        ErrorCode::SchedQueueFull,
        ErrorCode::Cancelled,
        ErrorCode::MemoryBudgetExceeded,
        ErrorCode::ShapeBroadcast,
        ErrorCode::ShapeMismatch,
        ErrorCode::DTypeUnsupported,
        ErrorCode::IndexOutOfBounds,
        ErrorCode::IndexInvalid,
        ErrorCode::GpuUnavailable,
        ErrorCode::GpuUnsupportedOperation,
        ErrorCode::GpuDeviceLost,
        ErrorCode::GpuOutOfMemory,
        ErrorCode::IoExists,
        ErrorCode::IoFormat,
        ErrorCode::IoModified,
        ErrorCode::IoFailed,
        ErrorCode::NativeLoad,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::ContextInvalidOptions => "NMATH_CONTEXT_INVALID_OPTIONS",
            ErrorCode::ContextMismatch => "NMATH_CONTEXT_MISMATCH",
            ErrorCode::ContextClosed => "NMATH_CONTEXT_CLOSED",
            ErrorCode::Disposed => "NMATH_DISPOSED",
            ErrorCode::SchedQueueFull => "NMATH_SCHED_QUEUE_FULL",
            ErrorCode::Cancelled => "NMATH_CANCELLED",
            ErrorCode::MemoryBudgetExceeded => "NMATH_MEMORY_BUDGET_EXCEEDED",
            ErrorCode::ShapeBroadcast => "NMATH_SHAPE_BROADCAST",
            ErrorCode::ShapeMismatch => "NMATH_SHAPE_MISMATCH",
            ErrorCode::DTypeUnsupported => "NMATH_DTYPE_UNSUPPORTED",
            ErrorCode::IndexOutOfBounds => "NMATH_INDEX_OUT_OF_BOUNDS",
            ErrorCode::IndexInvalid => "NMATH_INDEX_INVALID",
            ErrorCode::GpuUnavailable => "NMATH_GPU_UNAVAILABLE",
            ErrorCode::GpuUnsupportedOperation => "NMATH_GPU_UNSUPPORTED_OPERATION",
            ErrorCode::GpuDeviceLost => "NMATH_GPU_DEVICE_LOST",
            ErrorCode::GpuOutOfMemory => "NMATH_GPU_OUT_OF_MEMORY",
            ErrorCode::IoExists => "NMATH_IO_EXISTS",
            ErrorCode::IoFormat => "NMATH_IO_FORMAT",
            ErrorCode::IoModified => "NMATH_IO_MODIFIED",
            ErrorCode::IoFailed => "NMATH_IO_FAILED",
            ErrorCode::NativeLoad => "NMATH_NATIVE_LOAD",
        }
    }

    /// English recovery guidance appended to every message.
    pub fn recovery(self) -> &'static str {
        match self {
            ErrorCode::ContextInvalidOptions => {
                "Pass positive maxConcurrentJobs and non-negative queue and memory limits."
            }
            ErrorCode::ContextMismatch => {
                "Copy the value into the target context explicitly before combining it."
            }
            ErrorCode::ContextClosed => "Create a new context for new work.",
            ErrorCode::Disposed => "Do not use a value after calling dispose().",
            ErrorCode::SchedQueueFull => {
                "Wait for queued work to finish or create the context with a larger queue."
            }
            ErrorCode::Cancelled => "Submit a new computation if the result is still needed.",
            ErrorCode::MemoryBudgetExceeded => {
                "Reduce the input size or create the context with a larger memory budget."
            }
            ErrorCode::ShapeBroadcast | ErrorCode::ShapeMismatch => {
                "Check the operand shapes against the operation requirements."
            }
            ErrorCode::DTypeUnsupported => "Convert the operand to a supported dtype explicitly.",
            ErrorCode::IndexOutOfBounds | ErrorCode::IndexInvalid => {
                "Use an index inside the axis length or a valid selector."
            }
            ErrorCode::GpuUnavailable => "Run on device 'cpu' or use a supported GPU platform.",
            ErrorCode::GpuUnsupportedOperation => {
                "Use float32, int32 or uint32 with a supported GPU operation, or run on 'cpu'."
            }
            ErrorCode::GpuDeviceLost => "Submit the computation again after the device recovers.",
            ErrorCode::GpuOutOfMemory => "Reduce the GPU workload or free GPU memory.",
            ErrorCode::IoExists => "Pass the overwrite option to replace the existing file.",
            ErrorCode::IoFormat => "Check that the file is a valid file of the expected format.",
            ErrorCode::IoModified => "Keep input files unchanged while a computation runs.",
            ErrorCode::IoFailed => "Check file permissions and available disk space.",
            ErrorCode::NativeLoad => {
                "Reinstall the package for this platform or build the native module from source."
            }
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An NMath error with a stable code and an English message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

impl Error {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Message with recovery guidance, as surfaced to users.
    pub fn full_message(&self) -> String {
        format!("{}: {} {}", self.code, self.message, self.code.recovery())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.full_message())
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_and_prefixed() {
        let mut seen = std::collections::HashSet::new();
        for code in ErrorCode::ALL {
            assert!(code.as_str().starts_with("NMATH_"));
            assert!(seen.insert(code.as_str()));
            assert!(!code.recovery().is_empty());
        }
    }

    #[test]
    fn full_message_includes_code_and_recovery() {
        let err = Error::new(ErrorCode::SchedQueueFull, "The queue holds 2 jobs.");
        let text = err.full_message();
        assert!(text.starts_with("NMATH_SCHED_QUEUE_FULL: The queue holds 2 jobs."));
        assert!(text.contains("larger queue"));
    }
}
