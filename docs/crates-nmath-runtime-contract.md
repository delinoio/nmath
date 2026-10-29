# NMath Runtime Crates Contract

## Scope

This contract owns the `nmath-core` and `nmath-node` components of the [`nmath` project](project-nmath.md):

- `crates/nmath-core/`: Rust library with arrays, dtypes, lazy expressions, CPU and GPU kernels, execution contexts, scheduling, memory accounting, local file I/O, random generators and measurements.
- `crates/nmath-node/`: napi-rs `cdylib` that exposes `nmath-core` to Node.js 24.

## Runtime and Language

- Rust, using the repository toolchain in `rust-toolchain` and the 2024 edition.
- `nmath-node` uses napi-rs with N-API version 9 or later and targets Node.js 24 only.
- GPU kernels use Rust `wgpu` inside the native module. The GPU path initializes only when a job requests `device: 'gpu'`.
- No Python interpreter or Python package is a runtime, build or link dependency.

## Users and Operators

- Library consumers who call `@delinoio/nmath` from JavaScript, TypeScript or compiled `.nts`.
- Maintainers who validate numerical behavior against NumPy 2.5.4, pandas 3.1.0 and Polars 2.1.0.

## Interfaces and Contracts

### Dtypes

`DType` is a Rust enum whose variants map one-to-one to the dtype IDs in the [project index](project-nmath.md#stable-identifiers). Type promotion follows NumPy 2.5.4 rules (NEP 50 semantics for Python-scalar-like JS numbers). The core crate exposes `DType::promote(a, b)` and `DType::id()`.

Small signed and unsigned integers use one-byte (`int8`, `uint8`) or two-byte (`int16`, `uint16`) storage. Arithmetic wraps at the result dtype width. Mixed signed/unsigned arrays promote to a signed width that can represent both ranges, or to `float64` when no such integer width exists. Weak integer operands must fit the promoted dtype. Integer true division and mean return `float64`; sum widens small signed integers to `int64` and small unsigned integers to `uint64`. NPY uses `|i1`, `|u1`, `<i2` and `<u2` descriptors for these dtypes.

`complex64` stores two IEEE-754 binary32 components in 8 bytes; `complex128` stores two binary64 components in 16 bytes. Components are ordered real, imaginary. Array promotion follows NumPy 2.5.4: `complex64` remains `complex64` with boolean, 8-bit/16-bit integers and `float32`, and promotes to `complex128` with 32-bit/64-bit integers, `float64` or `complex128`. A weak complex scalar promotes boolean/integer arrays to `complex128`, `float32` arrays to `complex64`, and preserves an existing complex dtype.

Complex CPU kernels support `add`, `sub`, `mul`, `div`, `pow`, `pos`, `neg`, `sum`, `mean`, lexicographic `min`/`max`, and the existing 1-D/2-D `matmul` forms. Complex remainder is unsupported and fails at expression construction. Complex extrema propagate NaNs and preserve the first equal complex element. Sum and mean preserve the complex dtype. Contiguous complex reductions use pairwise accumulation in component precision.

Basic indexing shares complex storage; advanced indexing and explicit context copies preserve both components. An explicit `astype` may discard the imaginary component when converting to a real dtype. Complex-to-boolean casts test both components. Assignment from complex values to real numeric storage requires an explicit cast before the write; assignment to boolean storage uses complex truth semantics. Failed writes preserve the target.

Boolean remainder and power use the NumPy `int8` result loop. Boolean addition and multiplication preserve `bool`; subtraction and both unary arithmetic operators reject boolean operands.

### Arrays and storage

- `Storage` is a reference-counted contiguous buffer owned by one context. Arrays are `(storage, offset, shape, strides, dtype)` views.
- A shared view reflects later changes to its storage. An execution snapshot copies or pins the visible bytes when a job starts, so later writes do not affect a running job.
- Every storage records its byte size against its context's CPU or GPU budget until the last reference drops.

### Lazy expressions

- `Expr` is an immutable graph node. Constructors: `Input(array)`, `Scalar(value)`, `Unary(op, expr)`, `Binary(op, lhs, rhs)`, `Reduce(op, expr, axes)`, `MatMul(lhs, rhs)`, `Index(expr, selectors)`, `Cast(expr, dtype)`.
- Binary operator IDs: `add`, `sub`, `mul`, `div`, `mod`, `pow`. Unary operator IDs: `pos`, `neg`. Reduction IDs: `sum`, `mean`, `min`, `max`.
- Shape and dtype of every node are inferred at construction. Invalid broadcasting fails at construction with `NMATH_SHAPE_BROADCAST`.
- Indexing selectors: integer, `Slice { start, stop, step }` with `stop` excluded and `None` for an omitted bound, integer-array selection and boolean-mask selection. Negative indexes count from the end; out-of-range integers fail with `NMATH_INDEX_OUT_OF_BOUNDS`.

### Contexts and scheduling

- `Context::new(ContextOptions { max_concurrent_jobs, max_queue_length, cpu_memory_budget, gpu_memory_budget })`. All four fields are required. A zero concurrency limit fails with `NMATH_CONTEXT_INVALID_OPTIONS`.
- Admission: if fewer than `max_concurrent_jobs` jobs run, the job starts. Otherwise it enters a FIFO queue. If the queue holds `max_queue_length` jobs, admission fails with `NMATH_SCHED_QUEUE_FULL`.
- Each job carries a `CancellationToken`. Cancelling a queued job removes it and completes it with `NMATH_CANCELLED`. Workers check cancellation before starting work. Kernels check the token between chunks; a cancelled read or computation does not return its result. File writers check cancellation before atomic publication. Successful publication is terminal success: cancellation received after publication does not change the completion or metrics to cancelled.
- `Context::close()` rejects new work with `NMATH_CONTEXT_CLOSED`, cancels queued work, cancels running work cooperatively and resolves after all jobs release their resources.
- Jobs on one storage that write are serialized in request order. A write commits only after it fully succeeds.

### Memory accounting

- `MemoryBudget` tracks `used`, `peak` and `limit` separately for `cpu` and `gpu`. `reserve(bytes)` fails with `NMATH_MEMORY_BUDGET_EXCEEDED` before any allocation when `used + bytes > limit`.
- Inputs, snapshots, intermediates and results are all charged. Charges release on drop. GPU execution charges snapshots and host results to the CPU budget, and upload, storage, parameter, status and readback buffers to the GPU budget. Submitted allocations stay charged until device work finishes, including after cancellation. Driver and compiler internal allocations are outside the managed-allocation accounting boundary.

Reduction and matrix-multiplication kernels borrow operands whose dtype already matches. Required operand conversions reserve their full byte size before allocation and release that reservation when the temporary drops. Reductions traverse output offsets without input-sized slot or integer scratch vectors. Min/max reserve their output-initialization flags. Casts and kernels check cancellation between chunks; failure and cancellation release temporary reservations.

### Random generation

- `Generator` implements NumPy's `PCG64` bit generator and `SeedSequence` seeding so the same seed produces the same raw 64-bit stream as NumPy 2.5.4.
- Jobs that share a generator serialize. A job reads state at start and commits the advanced state only on success.

### Measurements

`Context::metrics()` returns current and cumulative values: per-execution ID, operation kind, device, status, queue time and execution time, and NMath-managed memory usage. Measurements never contain input data, file paths or file contents.

### Error codes

| Code                              | Meaning                                                  |
| --------------------------------- | -------------------------------------------------------- |
| `NMATH_CONTEXT_INVALID_OPTIONS`   | Context options are missing or invalid.                  |
| `NMATH_CONTEXT_MISMATCH`          | Values from different contexts were combined directly.   |
| `NMATH_CONTEXT_CLOSED`            | The context is closed.                                   |
| `NMATH_DISPOSED`                  | The object was disposed.                                 |
| `NMATH_SCHED_QUEUE_FULL`          | The context queue is full.                               |
| `NMATH_CANCELLED`                 | The job was cancelled.                                   |
| `NMATH_MEMORY_BUDGET_EXCEEDED`    | An allocation does not fit the device budget.            |
| `NMATH_SHAPE_BROADCAST`           | Shapes cannot broadcast.                                 |
| `NMATH_SHAPE_MISMATCH`            | A shape does not match the operation requirement.        |
| `NMATH_DTYPE_UNSUPPORTED`         | The dtype is not supported for the operation or device.  |
| `NMATH_INDEX_OUT_OF_BOUNDS`       | An index is outside the axis.                            |
| `NMATH_INDEX_INVALID`             | A selector is malformed.                                 |
| `NMATH_GPU_UNAVAILABLE`           | No supported GPU adapter is available.                   |
| `NMATH_GPU_UNSUPPORTED_OPERATION` | The operation or dtype is outside the GPU subset.        |
| `NMATH_GPU_DEVICE_LOST`           | The GPU device was lost during execution.                |
| `NMATH_GPU_OUT_OF_MEMORY`         | The GPU driver reported memory exhaustion.               |
| `NMATH_IO_EXISTS`                 | The output file exists and overwrite was not requested.  |
| `NMATH_IO_FORMAT`                 | The input file is malformed or unsupported.              |
| `NMATH_IO_MODIFIED`               | A change to an input file was detected during execution. |
| `NMATH_IO_FAILED`                 | An operating-system I/O operation failed.                |
| `NMATH_NATIVE_LOAD`               | The prebuilt native binary could not load.               |

Every error message is English and includes recovery guidance.

### GPU subset

GPU jobs support `float32`, `int32` and `uint32` arrays for elementwise `add`, `sub`, `mul`, `div`, `mod`, `pow`, `pos`, `neg`, broadcasting, `sum`, `mean`, `min`, `max` and `matmul`. Any other dtype or operation fails with `NMATH_GPU_UNSUPPORTED_OPERATION` before queue admission when the condition is known statically.

Float32 GPU kernels carry binary32 values as integer words. Power uses Q32.96 logarithm/exponential arithmetic, explicit special-value handling, and exact integer and rational-root paths for rounding ties.

GPU initialization occurs only on an execution worker after admission. Native backend selection is restricted to Metal on macOS arm64, Vulkan on Linux x64 glibc and DX12 on Windows x64. Software adapters are rejected. Device discovery, allocation and device-loss errors remain distinct; execution never retries or falls back. Float32 execution requires preservation of subnormal arithmetic. A device that fails the precision probe is rejected; passing that probe alone is not numerical acceptance or an official hardware allowlist.

## Storage

- Local files only: CSV, JSON, Parquet, Arrow IPC, NPY and NPZ. This deviates from the repository R2 default because NMath is a local library; see the [project index](project-nmath.md#deviations-from-repository-defaults).
- Writers never overwrite an existing file unless the caller passes an explicit overwrite option. Output is completed in a temporary file in the destination directory and published atomically. Create-new publication uses a hard link that fails if the destination exists; replacement uses an atomic rename. Cancellation or error before publication removes the temporary file and preserves the original. Once publication succeeds, the write completes successfully even if cancellation arrives before the scheduler delivers completion.
- Readers record file size and modification time at job start and fail with `NMATH_IO_MODIFIED` when a change is detected. Detection is best effort.

Complex NPY output uses C order with `<c8` or `<c16` descriptors. Complex input accepts little- or big-endian components and C or Fortran order; the loader converts Fortran input to logical C order. Header, shape and payload-length validation applies before decoded allocation. These complex input conversions do not enable Fortran-order input for other dtypes. NPZ members use the same NPY codec and inherit its dtype and layout support.

## Security

- NMath uses the calling process's file permissions. It has no authentication, tenancy or sandbox.
- All file parsers validate lengths, offsets, shapes and dtypes before allocation and fail with `NMATH_IO_FORMAT` on malformed input.
- Unsafe Rust is limited to the napi boundary and audited buffer reinterpretation. Each `unsafe` block documents its invariant.

## Logging

- Use `tracing` with structured fields. Events: `nmath.job.admitted`, `nmath.job.started`, `nmath.job.finished`, `nmath.memory.rejected`, `nmath.gpu.init`.
- Logging is disabled unless the host enables it. Logs never contain input values, file paths or file contents.
- NMath sends no external telemetry.

## Build and Test

- `cargo test` from the repository root runs all crate unit tests.
- `nmath-node` builds with `pnpm --filter @delinoio/nmath build:native`, which invokes the napi-rs CLI.
- Differential tests against NumPy, pandas and Polars run from recorded fixtures. Fixture generation needs Python with the pinned reference versions; the runtime and its tests never import Python.
- Floating-point tolerances come from the matching upstream test for the same dtype and operation. A missing or relaxed tolerance remains a release blocker until an explicit decision approves it.

## Dependencies and Integrations

- `nmath-node` depends on `nmath-core`, `napi` and `napi-derive`.
- `nmath-core` may depend on `wgpu`, `arrow`, `parquet` and `rand_core`-compatible crates. Each dependency keeps its license; distributions include third-party notices.
- Downstream: [npm package contract](packages-nmath-package-contract.md).

## Change Triggers

- Changes to error codes, dtype IDs, operator IDs or execution semantics update the [npm package contract](packages-nmath-package-contract.md) and the [project index](project-nmath.md).
- Changes to the expression graph that the `.nts` transform targets update the [compiler crates contract](crates-nmath-compiler-contract.md).
- Update `AGENTS.md` only when development procedures, directory ownership or repository/domain development rules change; follow the [instruction-update policy](README.md#instruction-update-policy).

## References

- [NMath project index](project-nmath.md)
- [Repository defaults](repository-defaults.md)
- [Repository workflow](repository-workflow-contract.md)
- [Repository license](repository-license-contract.md)

## Numeric tables

A table plan contains numeric or boolean one-dimensional expressions and optional boolean validity expressions (`true` means valid). All source values and masks must have one context and matching lengths. An empty typed source is valid. String, temporal, categorical, complex, extended-precision and object columns are not supported by this table layer.

A table computation submits one CPU job to the context scheduler. It validates all source contexts before admission, freezes all source expressions and masks in one job environment before evaluation, and evaluates the complete plan on a native worker thread. The snapshot gate is also held by array mutation commits, so a mutation cannot split the table's snapshot phase. The gate is released before table kernels run. Repeated computations are independent. A queued job sees mutations completed before its snapshot phase. Cancellation and close use the existing scheduler; cancellation never returns a table result.

Plan nodes are source, named/positional projection, boolean-column filter, stable single-key sort, single-key inner/left equijoin and single-key group aggregation. Join keys must have identical numeric dtypes. Joins expand duplicate keys and preserve left-row and right-match order. Group operations are sum, mean, min, max and non-null count. Validity masks propagate through row operations; null filter values discard their rows. All-null groups produce zero for sum/count and invalid results for min/max/mean. Empty inputs produce typed empty results.

All buffers for source snapshots, evaluated columns, row selection, sort scratch space, join row mappings, group representatives, aggregation intermediates and results are reserved against the CPU budget before allocation. Native table results retain their data and validity reservations until their array handles are released. Metadata bookkeeping is not a process-wide memory cap. Integer key comparisons preserve all 64 bits.

Row-local numeric source expressions (inputs, scalars, unary/binary arithmetic and casts), projections and boolean-column filters execute in internal batches of at most 1,024 rows. Filters count against frozen inputs before reserving the exact completed result, then evaluate batches again to fill it. Snapshots still cover all source values and masks atomically at actual execution start. No public batch-stream API is exposed.

Stable sorting uses an iterative merge sort with cooperative cancellation. Joins and groups use pre-reserved randomized hash indexes that preserve first-seen group order, right-match order, exact integer keys, null equivalence, NaN equivalence and signed-zero equivalence. Group accumulation retains input order. Sort, join, group and non-row-local source expressions remain materialization boundaries; eligible input pipelines can use batches. Snapshots, hash indexes and completed results must fit the CPU budget. Spill-to-disk execution, streaming group/join state and full Polars compatibility remain implementation targets.

### NPZ archives

`Context::save_npz` admits one file-write job with ordered `(name, array)` entries, `NpzCompression::Stored` or `NpzCompression::Deflated`, a `WriteMode`, and a cancellation token. Every input must belong to that context. The worker snapshots inputs at actual execution start. Empty archives are valid. `Context::load_npz` admits one file-read job and returns ordered contextual arrays; rank-zero members remain arrays. Loading completes eagerly through this file job and does not construct a lazy multi-result expression.

One save freezes all input members under a single snapshot boundary. A mutation cannot commit between member copies, including when different member names refer to the same storage. The boundary ends before archive encoding or file work. Each member's offset scratch is reserved before its snapshot copy and released after that copy. During copying, peak usage includes completed member snapshots, the current member's offsets and payload, and metadata. During encoding, completed snapshots and metadata remain charged alongside codec workspace.

Members contain NPY data. Writers append `.npy` to each member name; readers remove that suffix once. Names must be nonempty UTF-8 components, must not be `.` or `..`, and must not contain `/`, `\`, `:`, or control characters. A name is limited to 65,531 UTF-8 bytes before its `.npy` suffix. Duplicate names, non-NPY members, encryption, pickle/object data, and unsupported compression methods fail with `NMATH_IO_FORMAT`. Members are never extracted to disk.

The codec supports stored and deflated ZIP members, including NumPy ZIP64 local member headers. It currently rejects multi-disk archives, ZIP64 central directories, 65,535 or more entries, and output capacity bounds at or above 4 GiB. Conservative output bounds can reject inputs before their actual compressed size reaches this limit. These are implementation limits, not agreed exclusions from compatibility requirements.

The reader validates central and local headers, member boundaries and overlap, lengths, CRC and decompression completion. It checks file size and modification time through decoding; modification detection remains best effort. The context charges input archives, member metadata, decompression buffers, codec workspace, decoded arrays, execution snapshots, output archives and publication pathname workspace before their allocations. Metadata stays charged until native results are converted to JS entries. Codec workspace reservations are conservative bounds for the pinned ZIP/DEFLATE versions. Cancellation is cooperative around shared NPY codec passes and between compressed I/O chunks; one NPY encode/decode pass is not internally interruptible.

NPZ supports the numeric and boolean types accepted by the shared NPY codec, including `complex64` and `complex128`. Complex output is little-endian C order. Complex input may be little- or big-endian and C or Fortran order and is converted to logical C order without changing component precision. Unsupported noncomplex endian/layout combinations, strings, datetime/timedelta, structured and extended dtypes remain compatibility gaps.

### Numeric CSV and JSON records

`io_text` supplies scheduled `Context::read_text` and `Context::write_text` file jobs. CSV uses UTF-8, a comma delimiter, one header row, doubled quote escaping, and LF/CRLF/CR records. JSON uses an array of numeric record objects. Record keys retain first-record order; subsequent records may reorder the same keys. Duplicate names, missing or extra keys, broken CSV quoting, inconsistent CSV widths, nested values and string-valued cells fail with `NMATH_IO_FORMAT`. Errors omit paths and contents.

Read schema entries are ordered `(name, dtype)` pairs for all columns. Explicit schemas preserve boolean, signed/unsigned integer widths, float widths and all-null column dtypes. Values retain separate validity masks. Empty CSV cells and JSON null become invalid values with zero storage. Schema-free inference chooses bool, int64, uint64 or float64; all-null columns become float64. Mixed signed/unsigned values that cannot fit one integer type fail instead of narrowing. Decimal/exponent syntax selects floating-point inference. Explicit integer schemas reject out-of-range values. JSON numeric tokens retain all 64 integer bits without passing through JS numbers. Empty JSON arrays need a schema to retain column names and dtypes.

Writers evaluate and freeze all numeric value/mask expressions in one CPU file-write job, using the shared table snapshot gate at execution start. Random-generator expressions remain unsupported. Writers emit numeric CSV cells or JSON records; JSON maps non-finite floats to null, while CSV emits numeric non-finite tokens. JSON has no native NaN/infinity representation. Neither format stores dtype metadata. Readers need the explicit schema for dtype-faithful round trips. CSV/JSON record output does not encode pandas index labels.

Input bytes, parse trees, numeric buffers, masks, serialization buffers and publication workspace receive CPU reservations before allocation. The current parser materializes the whole input and uses a conservative workspace allowance of 512 bytes per input byte plus 64 KiB. Output uses conservative per-cell and escaped-name bounds. These limits can reject files whose final table would fit. This is not the required internal streaming engine. JSON parsing retains the serde recursion limit and checks cancellation around its parser call; CSV checks cancellation during scanning and numeric conversion checks it in chunks. Size/mtime checks before reading and after decoding provide best-effort source-change detection. They do not provide a source snapshot.

All writes use the shared atomic publisher and require explicit overwrite authorization. The final fallible operation is publication; later cancellation cannot change successful file-write completion. Cancellation or a failure before publication preserves the destination.

Zero-column CSV and nonempty zero-column JSON records are rejected because the current table adapters cannot preserve their row/schema distinction. Custom delimiters, encoding options, compressed text, JSON orientations other than records, newline-delimited JSON, richer scalar dtypes, pandas missing-token/blank-line policies and the full upstream read/write option sets remain compatibility gaps, not exclusions. Public adapters describe the supported JS adaptation explicitly; these implementations do not establish full upstream API parity.
