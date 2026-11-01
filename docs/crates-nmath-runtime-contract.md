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

`float16` stores IEEE-754 binary16 values in two bytes. CPU arithmetic rounds each result to binary16; contiguous sum, mean and matrix multiplication use binary32 accumulation. Mean retains binary32 accumulators until division, including non-contiguous axes, and charges output-sized scratch before allocation. Non-contiguous sum rounds updates to the binary16 output. Min/max preserve the first equal value and propagate NaNs. Array promotion preserves float16 with boolean and 8-bit integers, widens to float32 with 16-bit integers, and widens to float64 with 32-bit/64-bit integers. Weak real scalars preserve float16; weak complex scalars promote it to complex64. GPU float16 expressions fail before queue admission. Float16 NPY output uses `<f2`; input accepts `<f2` and `>f2` in C or Fortran order. NPZ inherits the NPY codec. Recorded NumPy fixtures use exact finite bits and NaN classification; arbitrary power and long reduction/matrix accumulation tolerances remain uncovered numerical criteria.

Small signed and unsigned integers use one-byte (`int8`, `uint8`) or two-byte (`int16`, `uint16`) storage. Arithmetic wraps at the result dtype width. Mixed signed/unsigned arrays promote to a signed width that can represent both ranges, or to `float64` when no such integer width exists. Weak integer operands must fit the promoted dtype. Integer true division and mean return `float64`; sum widens small signed integers to `int64` and small unsigned integers to `uint64`. NPY uses `|i1`, `|u1`, `<i2` and `<u2` descriptors for these dtypes.

`complex64` stores two IEEE-754 binary32 components in 8 bytes; `complex128` stores two binary64 components in 16 bytes. Components are ordered real, imaginary. Array promotion follows NumPy 2.5.4: `complex64` remains `complex64` with boolean, 8-bit/16-bit integers and `float32`, and promotes to `complex128` with 32-bit/64-bit integers, `float64` or `complex128`. A weak complex scalar promotes boolean/integer arrays to `complex128`, `float32` arrays to `complex64`, and preserves an existing complex dtype.

Complex CPU kernels support `add`, `sub`, `mul`, `div`, `pow`, `pos`, `neg`, `sum`, `mean`, lexicographic `min`/`max`, and broadcast-batched `matmul` with vector promotion. Complex remainder is unsupported and fails at expression construction. Complex extrema propagate NaNs and preserve the first equal complex element. Sum and mean preserve the complex dtype. Contiguous complex reductions use pairwise accumulation in component precision.

Contiguous CPU `float32` sum and mean use pairwise accumulation with `float32` intermediates, following the pinned NumPy real reduction order. Mean divides the `float32` sum in `float32` precision, including overflow. Noncontiguous reductions retain their existing accumulation order.

CPU variance (`var`) and standard deviation (`std`) use a two-pass mean and squared-deviation computation over all axes or one axis. Integer and boolean inputs compute in float64; float16 keeps float16; float32 and complex64 return float32; float64 and complex128 return float64. Complex deviations use the squared magnitude, so the result is real. `ddof` is subtracted from the element count and the divisor is clamped at zero, matching NumPy: a zero divisor returns NaN for a zero numerator and infinity otherwise. Empty reductions therefore return NaN. String and temporal operands are rejected at expression construction, and GPU admission rejects variance with `NMATH_GPU_UNSUPPORTED_OPERATION`.

CPU cumulative scans (`cumsum`, `cumprod`) accumulate in the sum-promoted dtype. Without an axis they flatten the input to one dimension in C order. With an axis they accumulate inside that axis and keep the input shape. The first element of each run is copied without applying an identity, so signed-zero and NaN behavior is preserved. Empty inputs produce empty results. String and temporal scans are rejected at expression construction. GPU admission rejects scans with `NMATH_GPU_UNSUPPORTED_OPERATION`.

CPU product reductions (`prod`) reuse the sum dtype promotion and multiply in the promoted storage dtype. A zero-length reduced axis yields the multiplicative identity (`1`, `1+0j` or `true`) for each retained output slot, so empty products succeed while empty `min`/`max` fail. Temporal and string products are rejected at expression construction. Index reductions (`argmin`, `argmax`) return `int64` positions, accept no axis or exactly one axis, and use the flat C-order offset when no axis is given. They keep the first equal value's index and make NaNs sticky with the first NaN index. Complex values compare lexicographically by real then imaginary component. A zero-length reduced axis fails with `NMATH_SHAPE_MISMATCH` before selection; a nonempty reduced axis whose retained output shape is empty returns an empty result. String and temporal index reductions are rejected at expression construction. GPU admission rejects product and index reductions with `NMATH_GPU_UNSUPPORTED_OPERATION` without fallback.

Basic indexing shares complex storage; advanced indexing and explicit context copies preserve both components. An explicit `astype` may discard the imaginary component when converting to a real dtype. Complex-to-boolean casts test both components. Assignment from complex values to real numeric storage requires an explicit cast before the write; assignment to boolean storage uses complex truth semantics. Failed writes preserve the target.

Boolean remainder and power use the NumPy `int8` result loop. Boolean addition and multiplication preserve `bool`; subtraction and both unary arithmetic operators reject boolean operands.

CPU boolean `matmul` uses OR of AND products. Each output stops scanning its inner dimension after the first true product. Long unsuccessful scans check cancellation between bounded chunks.

### Arrays and storage

- `Storage` is a reference-counted contiguous buffer owned by one context. Arrays are `(storage, offset, shape, strides, dtype)` views.
- A shared view reflects later changes to its storage. An execution snapshot copies or pins the visible bytes when a job starts, so later writes do not affect a running job.
- Every storage records its byte size against its context's CPU or GPU budget until the last reference drops.

### Shape transformations

`Expr::transpose` permutes all axes, reversing them when axes are omitted. Explicit axes may be negative but must normalize to a complete permutation. `Expr::reshape` supports C logical traversal and one inferred `-1` dimension. It preserves element count, dtype metadata and context. Shapes and stride arithmetic are checked before execution, including empty shapes. An empty input reshapes to `[-1]` as `[0]`; an inferred dimension with a zero known product is ambiguous and fails.

`CopyPolicy::IfNeeded` shares storage when signed strides permit the reshape; `Always` returns independent storage; `Never` rejects a copy-required layout. Regular stepped and reversed runs can reshape without copying. Pure chains of basic indexing, transpose and compatible reshape return live shared views. Writes through these views retain original-storage serialization and lifetime rules. Shape transforms over computed values retain their internal layouts, while all materialization and arithmetic read execution-start snapshots. Intermediate and copy-required results reserve managed CPU bytes before allocation and check cancellation during logical-order copies without element-sized offsets scratch.

C-order reshape and transpose are CPU operations. Other reshape traversal orders remain compatibility targets. GPU shape-transform nodes reject before queue admission without fallback.

### Lazy expressions

- `Expr` is an immutable graph node. Constructors: `Input(array)`, `Scalar(value)`, `Unary(op, expr)`, `Binary(op, lhs, rhs)`, `Reduce(op, expr, axes)`, `MatMul(lhs, rhs)`, `Index(expr, selectors)`, `Cast(expr, dtype)`, `Transpose(expr, axes)`, `Reshape(expr, copy_policy)`.
- Metadata discovery and input freezing visit each distinct expression node once per traversal, in first-encounter order. A shared node keeps its identity across incoming edges. Numerical evaluation remains a separate operation.
- Binary operator IDs: `add`, `sub`, `mul`, `div`, `mod`, `pow`. Unary operator IDs: `pos`, `neg`. Reduction IDs: `sum`, `mean`, `min`, `max`, `prod`, `argmin`, `argmax`. Cumulative scan IDs: `cumsum`, `cumprod`. Variance IDs: `var`, `std`.
- Shape and dtype of every node are inferred at construction. Invalid broadcasting fails at construction with `NMATH_SHAPE_BROADCAST`.
- Indexing selectors: integer, `Slice { start, stop, step }` with `stop` excluded and `None` for an omitted bound, integer-array selection and boolean-mask selection. Negative indexes count from the end; out-of-range integers fail with `NMATH_INDEX_OUT_OF_BOUNDS`.

### Matrix products

`matmul` accepts numeric and boolean operands of rank at least one. It contracts the left operand's last axis with the right operand's second-last axis (or its only axis for a vector), broadcasts leading batch axes, and removes the temporary row or column axis used to promote a vector. Scalar operands and unequal contraction dimensions fail with `NMATH_SHAPE_MISMATCH`; incompatible leading axes fail with `NMATH_SHAPE_BROADCAST` at expression construction.

CPU execution traverses broadcast batches without expanding repeated operands or allocating per-batch results. It checks the complete output element and byte counts before allocation and reserves the output once. Empty batch/row/column axes return empty output; an empty contraction returns zeros. Integer arithmetic wraps in the promoted dtype. Boolean products retain OR-of-AND short-circuit behavior, float16 retains float32 accumulation, and complex products do not conjugate either operand. Cancellation checks run between batches and kernel chunks. GPU execution uses the same inferred shape with its existing dtype and platform restrictions; shader compilation does not establish real-device numerical acceptance.

### Contexts and scheduling

- `Context::new(ContextOptions { max_concurrent_jobs, max_queue_length, cpu_memory_budget, gpu_memory_budget })`. All four fields are required. A zero concurrency limit fails with `NMATH_CONTEXT_INVALID_OPTIONS`.
- Admission: if fewer than `max_concurrent_jobs` jobs run, the job starts. Otherwise it enters a FIFO queue. If the queue holds `max_queue_length` jobs, admission fails with `NMATH_SCHED_QUEUE_FULL`.
- Each job carries a `CancellationToken`. Cancelling a queued job removes it and completes it with `NMATH_CANCELLED`. Workers check cancellation before starting work. Kernels check the token between chunks; a cancelled read or computation does not return its result. File writers check cancellation before atomic publication. Successful publication is terminal success: cancellation received after publication does not change the completion or metrics to cancelled.
- `Context::close()` rejects new work with `NMATH_CONTEXT_CLOSED`, cancels queued work, cancels running work cooperatively and resolves after all jobs release their resources.
- A terminal job records its status, releases its execution slot, and starts the next queued job before it invokes its completion callback. The callback runs without the scheduler or callback-storage lock. It can close its context or submit and wait for another job, subject to normal admission and cancellation rules.
- Jobs on one storage that write are serialized in admission order. Ticket creation and scheduler admission use one critical section, so a running context job never waits for a ticket held by a queued successor. A write commits only after it fully succeeds. Rejected and cancelled jobs give up unused tickets.

### Memory accounting

- `MemoryBudget` tracks `used`, `peak` and `limit` separately for `cpu` and `gpu`. `reserve(bytes)` fails with `NMATH_MEMORY_BUDGET_EXCEEDED` before any allocation when `used + bytes > limit`.
- Inputs, snapshots, intermediates and results are all charged. Charges release on drop. GPU execution charges snapshots and host results to the CPU budget, and upload, storage, parameter, status and readback buffers to the GPU budget. Submitted allocations stay charged until device work finishes, including after cancellation. Driver and compiler internal allocations are outside the managed-allocation accounting boundary.

Reduction and matrix-multiplication kernels borrow operands whose dtype already matches. Required operand conversions reserve their full byte size before allocation and release that reservation when the temporary drops. Reductions traverse output offsets without input-sized slot or integer scratch vectors. Reduction rank metadata and min/max output-initialization flags remain charged for their live lifetimes; index reductions charge output-sized best-value and seen scratch before allocation; variance charges output-sized accumulators before allocation; product reductions and cumulative scans need no extra scratch beyond a promoted operand copy. Result data and shape remain charged until the result drops. Reduction byte sizes are checked before allocation. Casts and kernels check cancellation between chunks; failure and cancellation release temporary reservations.

### Random generation

- `Generator` implements NumPy's `PCG64` bit generator and `SeedSequence` seeding so the same seed produces the same raw 64-bit stream as NumPy 2.5.4.
- Jobs that share a generator serialize in admission order. The same critical section coordinates generator tickets with scheduler admission. A job reads state at start and commits the advanced state only on success.
- Scalar-parameter random operations support `random` in float32/float64, `integers` in bool and signed/unsigned 8/16/32/64-bit dtypes, `uniform`, float32/float64 `standard_normal`, `normal`, and `exponential`. Uniform, normal and exponential return float64. Integer high bounds are exclusive unless `endpoint` is true. Full-width 64-bit bounds are represented without float conversion.
- PCG64 state includes the cached upper uint32 word and its validity flag. Small-integer distribution buffers are local to one call. Each execution reads the last successful state and independently advances it. FIFO guards remain held through the scheduler's final cancellation decision and state commit. Error or pre-completion cancellation discards pending state. An array assignment that has published its atomic write completes successfully; a later cancellation cannot undo the array or generator commit.
- Normal and exponential use the pinned NumPy Ziggurat routines and constants. Arithmetic preserves the pinned oracle's contracted multiply-add behavior. Exact cross-platform transcendental results still require platform-specific acceptance.
- Array-valued parameter broadcasting, `out`, alternative bit generators, SeedSequence spawning/array entropy, state import/export and other distributions remain compatibility targets. The supported seeding interface accepts an explicit nonnegative integer of at most 128 bits.
- CPU random evaluation checks the element count, dtype-specific byte count and vector capacity before reserving or allocating the result. Unrepresentable sizes fail through the job callback with `NMATH_MEMORY_BUDGET_EXCEEDED`; a zero-length dimension produces an empty result even when other dimensions are large. Failed jobs do not advance generator state.

### Measurements

`Context::metrics()` returns current and cumulative values: per-execution ID, operation kind, device, status, queue time and execution time, and NMath-managed memory usage. Measurements never contain input data, file paths or file contents.

### Error codes

| Code                              | Meaning                                                              |
| --------------------------------- | -------------------------------------------------------------------- |
| `NMATH_CONTEXT_INVALID_OPTIONS`   | Context options are missing or invalid.                              |
| `NMATH_CONTEXT_MISMATCH`          | Values from different contexts were combined directly.               |
| `NMATH_CONTEXT_CLOSED`            | The context is closed.                                               |
| `NMATH_DISPOSED`                  | The object was disposed.                                             |
| `NMATH_SCHED_QUEUE_FULL`          | The context queue is full.                                           |
| `NMATH_CANCELLED`                 | The job was cancelled.                                               |
| `NMATH_MEMORY_BUDGET_EXCEEDED`    | An allocation does not fit the device budget.                        |
| `NMATH_SHAPE_BROADCAST`           | Shapes cannot broadcast.                                             |
| `NMATH_SHAPE_MISMATCH`            | A shape does not match the operation requirement.                    |
| `NMATH_LINALG_SINGULAR`           | A square matrix cannot be solved or inverted because it is singular. |
| `NMATH_DTYPE_UNSUPPORTED`         | The dtype is not supported for the operation or device.              |
| `NMATH_INDEX_OUT_OF_BOUNDS`       | An index is outside the axis.                                        |
| `NMATH_INDEX_INVALID`             | A selector is malformed.                                             |
| `NMATH_GPU_UNAVAILABLE`           | No supported GPU adapter is available.                               |
| `NMATH_GPU_UNSUPPORTED_OPERATION` | The operation or dtype is outside the GPU subset.                    |
| `NMATH_GPU_DEVICE_LOST`           | The GPU device was lost during execution.                            |
| `NMATH_GPU_OUT_OF_MEMORY`         | The GPU driver reported memory exhaustion.                           |
| `NMATH_IO_EXISTS`                 | The output file exists and overwrite was not requested.              |
| `NMATH_IO_FORMAT`                 | The input file is malformed or unsupported.                          |
| `NMATH_IO_MODIFIED`               | A change to an input file was detected during execution.             |
| `NMATH_IO_FAILED`                 | An operating-system I/O operation failed.                            |
| `NMATH_NATIVE_LOAD`               | The prebuilt native binary could not load.                           |

Every error message is English and includes recovery guidance.

### GPU subset

GPU jobs support `float32`, `int32` and `uint32` arrays for elementwise `add`, `sub`, `mul`, `div`, `mod`, `pow`, `pos`, `neg`, broadcasting, `sum`, `mean`, `min`, `max` and `matmul`. Any other dtype or operation fails with `NMATH_GPU_UNSUPPORTED_OPERATION` before queue admission when the condition is known statically.

Float32 GPU kernels carry binary32 values as integer words. Power uses Q32.96 logarithm/exponential arithmetic, explicit special-value handling, and exact integer and rational-root paths for rounding ties.

GPU initialization occurs only on an execution worker after admission. Native backend selection is restricted to Metal on macOS arm64, Vulkan on Linux x64 glibc and DX12 on Windows x64. Software adapters are rejected. Device discovery, allocation and device-loss errors remain distinct; execution never retries or falls back. Float32 execution requires preservation of subnormal arithmetic. A device that fails the precision probe is rejected; passing that probe alone is not numerical acceptance or an official hardware allowlist.

## Storage

- Local files only: CSV, JSON, Parquet, Arrow IPC, NPY and NPZ. This deviates from the repository R2 default because NMath is a local library; see the [project index](project-nmath.md#deviations-from-repository-defaults).
- Writers never overwrite an existing file unless the caller passes an explicit overwrite option. Output is completed in a temporary file in the destination directory and published atomically. Create-new publication uses a hard link that fails if the destination exists; replacement uses an atomic rename. Cancellation or error before publication removes the temporary file and preserves the original. Once publication succeeds, the write completes successfully even if cancellation arrives before the scheduler delivers completion.
- Readers record file size and modification time at job start and fail with `NMATH_IO_MODIFIED` when a change is detected. Detection is best effort.

Complex NPY output uses C order with `<c8` or `<c16` descriptors. Complex input accepts little- or big-endian components and C or Fortran order; the loader converts Fortran input to logical C order. Header, shape and payload-length validation applies before decoded allocation. Float16 also accepts both byte orders and Fortran input. Other real dtypes retain their existing input restrictions. NPZ members use the same NPY codec and inherit its dtype and layout support.

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

A table plan contains numeric, boolean, fixed-width Unicode or temporal one-dimensional expressions and optional boolean validity expressions (`true` means valid). All source values and masks must have one context and matching lengths. An empty typed source is valid. Categorical, complex, extended-precision and object columns are not supported by this table layer. String and temporal sorting, joins and groups remain unsupported.

A table computation submits one CPU job to the context scheduler. It validates all source contexts before admission, freezes all source expressions and masks in one job environment before evaluation, and evaluates the complete plan on a native worker thread. The snapshot gate is also held by array mutation commits, so a mutation cannot split the table's snapshot phase. The gate is released before table kernels run. Repeated computations are independent. A queued job sees mutations completed before its snapshot phase. Cancellation and close use the existing scheduler; cancellation never returns a table result.

Plan nodes are source, named/positional projection, boolean-column filter, stable single-key sort, single-key inner/left equijoin and single-key group aggregation. Join keys must have identical numeric dtypes. Joins expand duplicate keys and preserve left-row and right-match order. Group operations are sum, mean, min, max and non-null count. Validity masks propagate through row operations; null filter values discard their rows. All-null groups produce zero for sum/count and invalid results for min/max/mean. Empty inputs produce typed empty results.

Pandas grouping drops invalid keys and float32/float64 NaN keys when `dropna` is true. When `dropna` is false, it retains both kinds of keys. Polars grouping retains NaN keys and follows validity masks for null groups. Retained keys keep their existing equality and order rules.

All buffers for source snapshots, evaluated columns, row selection, sort scratch space, join row mappings, group representatives, aggregation intermediates and results are reserved against the CPU budget before allocation. Native table results retain their data and validity reservations until their array handles are released. Metadata bookkeeping is not a process-wide memory cap. Integer key comparisons preserve all 64 bits.

Row-local numeric source expressions (inputs, scalars, unary/binary arithmetic and casts), projections and boolean-column filters execute in internal batches of at most 1,024 rows. Filters count against frozen inputs before reserving the exact completed result, then evaluate batches again to fill it. Snapshots still cover all source values and masks atomically at actual execution start. No public batch-stream API is exposed.

Stable sorting uses an iterative merge sort with cooperative cancellation. Joins and groups use pre-reserved randomized hash indexes that preserve first-seen group order, right-match order, exact integer keys, null equivalence, NaN equivalence and signed-zero equivalence. Group accumulation retains input order. Sort, join, group and non-row-local source expressions remain materialization boundaries; eligible input pipelines can use batches. Snapshots, hash indexes and completed results must fit the CPU budget. Spill-to-disk execution, streaming group/join state and full Polars compatibility remain implementation targets.

### NPZ archives

`Context::save_npz` admits one file-write job with ordered `(name, array)` entries, `NpzCompression::Stored` or `NpzCompression::Deflated`, a `WriteMode`, and a cancellation token. Every input must belong to that context. The worker snapshots inputs at actual execution start. Empty archives are valid. `Context::load_npz` admits one file-read job and returns ordered contextual arrays; rank-zero members remain arrays. Loading completes eagerly through this file job and does not construct a lazy multi-result expression.

One save freezes all input members under a single snapshot boundary. A mutation cannot commit between member copies, including when different member names refer to the same storage. The boundary ends before archive encoding or file work. Each member's offset scratch is reserved before its snapshot copy and released after that copy. During copying, peak usage includes completed member snapshots, the current member's offsets and payload, and metadata. During encoding, completed snapshots and metadata remain charged alongside codec workspace.

Members contain NPY data. Writers append `.npy` to each member name; readers remove that suffix once. Names must be nonempty UTF-8 components, must not be `.` or `..`, and must not contain `/`, `\`, `:`, or control characters. A name is limited to 65,531 UTF-8 bytes before its `.npy` suffix. Duplicate names, non-NPY members, encryption, pickle/object data, and unsupported compression methods fail with `NMATH_IO_FORMAT`. Members are never extracted to disk.

The codec supports stored and deflated ZIP members, including NumPy ZIP64 local member headers. It currently rejects multi-disk archives, ZIP64 central directories, 65,535 or more entries, and output capacity bounds at or above 4 GiB. Conservative output bounds can reject inputs before their actual compressed size reaches this limit. These are implementation limits, not agreed exclusions from compatibility requirements.

The reader validates central and local headers, member boundaries and overlap, lengths, CRC and decompression completion. It checks file size and modification time through decoding; modification detection remains best effort. The context charges input archives, member metadata, decompression buffers, codec workspace, decoded arrays, execution snapshots, output archives and publication pathname workspace before their allocations. Metadata stays charged until native results are converted to JS entries. Codec workspace reservations are conservative bounds for the pinned ZIP/DEFLATE versions. Cancellation is cooperative around shared NPY codec passes and between compressed I/O chunks; one NPY encode/decode pass is not internally interruptible.

NPZ supports the numeric and boolean types accepted by the shared NPY codec, including `complex64` and `complex128`. Complex output is little-endian C order. Complex input may be little- or big-endian and C or Fortran order and is converted to logical C order without changing component precision. Fixed-width Unicode and unit-bearing temporal NPY members are supported. Remaining unsupported endian/layout combinations, byte/variable strings, generic temporal units, structured and extended dtypes remain compatibility gaps.

### Numeric CSV and JSON records

`io_text` supplies scheduled `Context::read_text` and `Context::write_text` file jobs. CSV uses UTF-8, a comma delimiter, one header row, doubled quote escaping, and LF/CRLF/CR records. JSON uses an array of numeric record objects. Record keys retain first-record order; subsequent records may reorder the same keys. Duplicate names, missing or extra keys, broken CSV quoting, inconsistent CSV widths, nested values and string-valued cells fail with `NMATH_IO_FORMAT`. Errors omit paths and contents.

Read schema entries are ordered `(name, dtype)` pairs for all columns. Explicit schemas preserve boolean, signed/unsigned integer widths, float widths and all-null column dtypes. Values retain separate validity masks. Empty CSV cells and JSON null become invalid values with zero storage. Schema-free inference chooses bool, int64, uint64 or float64; all-null columns become float64. Mixed signed/unsigned values that cannot fit one integer type fail instead of narrowing. Decimal/exponent syntax selects floating-point inference. Explicit integer schemas reject out-of-range values. JSON numeric tokens retain all 64 integer bits without passing through JS numbers. Empty JSON arrays need a schema to retain column names and dtypes.

Writers evaluate and freeze all numeric value/mask expressions in one CPU file-write job, using the shared table snapshot gate at execution start. Random-generator expressions remain unsupported. Writers emit numeric CSV cells or JSON records; JSON maps non-finite floats to null, while CSV emits numeric non-finite tokens. JSON has no native NaN/infinity representation. Neither format stores dtype metadata. Readers need the explicit schema for dtype-faithful round trips. CSV/JSON record output does not encode pandas index labels.

Input bytes, parse trees, numeric buffers, masks, serialization buffers and publication workspace receive CPU reservations before allocation. The current parser materializes the whole input and uses a conservative workspace allowance of 512 bytes per input byte plus 64 KiB. Output uses conservative per-cell and escaped-name bounds. These limits can reject files whose final table would fit. This is not the required internal streaming engine. JSON parsing retains the serde recursion limit and checks cancellation around its parser call; CSV checks cancellation during scanning and numeric conversion checks it in chunks. Size/mtime checks before reading and after decoding provide best-effort source-change detection. They do not provide a source snapshot.

All writes use the shared atomic publisher and require explicit overwrite authorization. The final fallible operation is publication; later cancellation cannot change successful file-write completion. Cancellation or a failure before publication preserves the destination.

Zero-column CSV and nonempty zero-column JSON records are rejected because the current table adapters cannot preserve their row/schema distinction. Custom delimiters, encoding options, compressed text, JSON orientations other than records, newline-delimited JSON, richer scalar dtypes, pandas missing-token/blank-line policies and the full upstream read/write option sets remain compatibility gaps, not exclusions. Public adapters describe the supported JS adaptation explicitly; these implementations do not establish full upstream API parity.

### Arrow IPC and Parquet files

The numeric file adapter submits one CPU `FileRead` or `FileWrite` job through the context scheduler. File writes freeze expressions and validity masks together at actual execution start. The adapter preserves column order, names, numeric/boolean widths, 64-bit integer bits and declared nullability. Complex, string, temporal, categorical, nested, dictionary and extension schemas fail explicitly. Duplicate column names and custom schema/field metadata fail explicitly, including pandas index/dtype metadata that the current adapter cannot preserve. These limits remain compatibility targets.

Arrow IPC supports the IPC file container (including uncompressed Feather V2), not the stream container or Feather V1. IPC compression is rejected. IPC schemas are limited to 1,024 flat fields and 4,096 UTF-8 bytes per field name. Parquet supports uncompressed and Snappy pages; output uses Snappy with dictionary encoding disabled. Other codecs remain targets. Embedded Arrow schema hints must agree with the decoded Parquet schema.

Reads reserve serialized input and conservative metadata storage before allocation. Allocation-free Compact Thrift scanning validates container lengths and nesting before Parquet metadata parsing; IPC block, buffer and expanded-schema lengths are checked before codec use. Parquet page output sizes and value counts are reserved before decompression. Decoding proceeds in internal batches into one completed, charged table. The serialized input is currently buffered in full. No public batch iterator is exposed. Codec reservations are conservative bounds for Arrow/Parquet 60.1.0, not a process-wide allocator limit. Oversized work fails without retry.

Writes serialize 4,096-row batches into a sibling temporary file, flush each Parquet row group, sync and atomically publish using the shared create-new/overwrite semantics. Snapshot, conversion, codec and metadata workspace are reserved before codec execution. Cancellation is checked between input chunks, columns, codec batches and before publication. A codec call itself cannot be interrupted. Publication success is terminal. Size/mtime checks compare both the open input handle and destination path after decoding; detection remains best effort.

Runtime dependencies use Arrow component crates and Parquet 60.1.0 with default features disabled. Parquet enables `arrow` and `snap` only. `base64` validates embedded Arrow metadata. Python and PyArrow are fixture-generation tools only.

Float16 file interchange currently uses NPY and NPZ. The CSV/JSON and Arrow/Parquet readers do not accept float16 schemas; callers must explicitly cast to a supported type for those formats.

## Square linear algebra and one-axis transforms

CPU lazy nodes provide `solve`, `inv`, `det`, `fft`, `ifft`, `rfft`, and `irfft`. Linear algebra uses the final two square axes and broadcasts batch axes. Only a rank-one solve right-hand side is a vector; all other right-hand sides are matrices. Empty matrices and batches retain NumPy shapes. Float16 linear algebra is rejected. Integer and boolean inputs compute as float64; float32/complex64 results retain their dtype unless an operand requires double precision. Singular solve/inverse fails with `NMATH_LINALG_SINGULAR`; singular determinants return zero.

Transforms accept one axis, optional positive length, and backward/forward/ortho normalization. They crop or zero-pad the selected axis. `irfft` on float16 returns float16. Other float16/float32/complex64 transforms use single-precision output; other supported numeric inputs use double precision. `rfft` rejects complex inputs. `irfft` defaults to twice the input axis length minus two, ignores the DC imaginary part and, for even output lengths, the Nyquist imaginary part, and reconstructs conjugate frequencies.

`fftshift` and `ifftshift` are CPU reindexing nodes over any dtype. `fftshift` rolls each selected axis forward by half its length; `ifftshift` applies the inverse roll. Default axes are all axes, repeated axes fail with `NMATH_INDEX_INVALID`, and out-of-range axes fail with `NMATH_INDEX_OUT_OF_BOUNDS`. The output is charged before allocation and cancellation is checked per chunk. Multi-axis transforms are composed from the one-axis node at expression construction: `fft2`/`ifft2` default to the final two axes and `fftn`/`ifftn` to all axes, with per-axis length cropping or zero-padding. Per-axis normalization composes to the NumPy total scale. Frequency helpers are package-level array construction and do not add a native node.

The kernels execute through the existing scheduler and graph snapshots, with precharged output and workspace. Linear algebra uses partial-pivot elimination and double-component workspace. Transforms use radix-two butterflies and Bluestein convolution with double-component workspace. Native loops check cancellation cooperatively. GPU submission fails before admission. Other factorizations, FFT convenience functions, output buffers, extended precision and full-family compatibility remain targets. Non-finite and extreme-scale numerical criteria remain pending.

`num-complex` 0.4.6 supplies scalar complex arithmetic with default features disabled and `std` enabled. Its MIT/Apache-2.0 license terms remain part of the native distribution license bundle.

### Strings and temporal storage

`str` uses fixed-width UTF-32 Unicode storage. Its width counts Unicode scalar values, including non-BMP characters as one value. Padding uses zero code points; scalar export removes trailing zero padding and preserves embedded zeros. String concatenation broadcasts and allocates the sum of the operand widths. Explicit string-to-string casts can change width and truncate; assignment rejects truncation unless the caller first casts explicitly. Object-valued storage is not supported. Numeric/string casts and unsupported string operators fail explicitly.

`datetime64` and `timedelta64` store signed 64-bit ticks and retain a positive unit multiplier with a base unit from `Y`, `M`, `W`, `D`, `h`, `m`, `s`, `ms`, `us`, `ns`, `ps`, `fs`, and `as`. The minimum signed 64-bit value represents NaT. Date unit casts use the proleptic Gregorian calendar; duration calendar casts use the Gregorian 400-year average. Negative unit conversion rounds down. Temporal arithmetic and conversions reject overflow, including a non-missing result that would collide with NaT. Unit-factor and negative-bound checks follow the fixed NumPy reference. Explicit int64 casts interchange raw ticks. Temporal assignment requires matching kind and unit; callers cast before assignment when a conversion is required.

CPU temporal kernels support datetime plus/minus duration, datetime subtraction with a duration result, duration addition/subtraction/remainder, duration-to-duration ratios, duration multiplication/division by integers, duration unary operators, datetime extrema, and duration sum/mean/extrema. Broadcasting preserves result unit metadata. Calendar and fixed-unit arithmetic require a supported common unit; unsupported combinations fail explicitly. NaT propagates through temporal arithmetic and reductions. Duration ratios return float64 and represent missing ratios as NaN. Duration multiplication/division with uint64 arrays uses the fixed NumPy reference’s bit-preserving int64 operand conversion; result overflow and NaT collisions still fail.

String widths and temporal units remain part of the dtype through views, indexing, snapshots, copies, mutation and disposal. Their buffer sizes participate in the existing pre-allocation budget checks. Assignment conversion buffers remain charged until the atomic write finishes. NPY and NPZ preserve these descriptors, including supported big-endian and Fortran-order input. Unicode NPY descriptors use `U`; temporal descriptors use `M8[unit]` and `m8[unit]`. Invalid Unicode scalar encodings are rejected.

Table materialization, named selection, boolean filtering and access preserve these columns and their metadata. Specialized string/temporal table sorting, join keys and group aggregation require further kernels and currently fail explicitly. This is an implementation limit, not an exclusion from the compatibility target.

### Polynomial evaluation

`Expr::polynomial` evaluates coefficients and points through a CPU-only lazy graph node. `PolynomialOrder::Ascending` follows `numpy.polynomial.polynomial.polyval`; `Descending` follows `numpy.polyval`. Coefficients occupy the first axis. Ascending evaluation accepts a rank-zero coefficient as a constant, rejects an empty coefficient axis, and converts boolean/integer coefficients to float64. Its initial `points * 0` preserves non-finite behavior and promotes boolean points to int64 before coefficient promotion. Descending evaluation requires a coefficient axis and preserves ordinary array dtype promotion, integer wrapping and 64-bit precision. An empty descending coefficient axis returns zeros with the points' dtype and shape.

Ascending tensor evaluation concatenates the remaining coefficient axes with the points' shape. Non-tensor and descending evaluation broadcast those remaining axes with the points. Shape and dtype errors fail at construction. Horner multiplication and addition round separately in the promoted dtype; no fused operation or f64 transport replaces integer or single-precision arithmetic. Inputs use actual-start snapshots. Results, operand conversions and rank-sized traversal workspace reserve CPU budget before allocation, with cooperative cancellation during traversal and long coefficient loops. GPU requests reject before admission without fallback. Other polynomial operations and polynomial object classes remain compatibility targets.

### Polynomial derivatives

`Expr::polynomial_derivative` creates a CPU-only lazy derivative node. Ascending coefficients follow `numpy.polynomial.polynomial.polyder`: scalar coefficients become a length-one axis, the selected axis accepts negative indexes, and boolean/integer coefficients convert to float64 even at order zero. Other numeric dtypes retain their precision. Every derivative stage multiplies by the scalar scale, rounds in the coefficient dtype, multiplies by the degree, and rounds again. Real coefficients require an explicit complex cast for active complex scaling. Zero order copies the converted coefficients. Orders at least as large as the axis length return the first coefficient slice multiplied by zero, preserving NaN effects; an empty axis stays empty.

Descending derivatives follow the rank-one array form of `numpy.polyder`. Positive orders promote coefficients against int64 degree factors and shorten the axis to `max(length - order, 0)`. Integer products wrap in their result dtype. Zero order returns the original expression, preserving its view behavior. Multidimensional legacy derivatives, polynomial object inputs and array-valued scales remain compatibility targets. Orders use nonnegative integer values; scalar scales can be real or complex.

Derivative inputs use execution-start snapshots. Output data, shape metadata and required dtype conversions reserve CPU budget before allocation. Kernels check cancellation during output traversal and repeated differentiation. High orders do not allocate or iterate once per order when the result is already constant or empty. Derivative nodes reject GPU requests before admission; the legacy zero-order identity retains its original expression and device support. Integration remains a compatibility target: NumPy's zero-polynomial shortcut makes its result shape depend on coefficient values, so padded coefficients must not substitute for the original shape semantics.
