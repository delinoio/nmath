//! NMath native runtime: arrays, dtypes, lazy expressions, kernels, contexts
//! and I/O.
//!
//! See `docs/crates-nmath-runtime-contract.md` for the behavior contract.

pub mod array;
pub mod buffer;
pub mod context;
pub mod dtype;
pub mod error;
pub mod expr;
pub mod gpu;
pub mod io;
pub mod io_npz;
pub mod kernels;
pub mod memory;
pub mod random;
pub mod shape;
pub mod sync;
pub mod temporal;

pub use array::NdArray;
pub use buffer::{Buffer, DenseArray, Scalar};
pub use context::{Context, ContextOptions, ExecutionStatus, JobOutput, Metrics, OperationKind};
pub use dtype::DType;
pub use error::{Error, ErrorCode, Result};
pub use expr::{Expr, Value};
pub use kernels::{BinaryOp, ReduceOp, UnaryOp};
pub use memory::Device;
pub use random::Generator;
pub use shape::Selector;
pub use sync::CancellationToken;

pub mod table;

pub mod io_columnar;
pub mod io_text;

pub mod fft;
pub mod linalg;
mod numerical;
