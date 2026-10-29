// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::ContextOptions;
fn budget() -> MemoryBudget {
    MemoryBudget::new(Device::Cpu, 1 << 30)
}
fn fixture(rows: usize, b: &MemoryBudget) -> (SchemaRef, Table) {
    let types = [
        DType::Bool,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int64,
        DType::UInt8,
        DType::UInt16,
        DType::UInt32,
        DType::UInt64,
        DType::Float32,
        DType::Float64,
    ];
    let fields = types
        .iter()
        .map(|&d| Field::new(d.id(), arrow_dtype(d).unwrap(), true))
        .collect::<Vec<_>>();
    let mut columns = Vec::new();
    for d in types {
        let charge = b.reserve(rows as u64 * d.item_size() as u64).unwrap();
        let value = match d {
            DType::Int64 => Scalar::Int(i64::MIN),
            DType::UInt64 => Scalar::UInt(u64::MAX),
            _ => Scalar::Int(1),
        };
        let values =
            DenseArray::new(vec![rows], Buffer::filled(d, rows, value).unwrap(), charge).unwrap();
        let valid = DenseArray::new(
            vec![rows],
            Buffer::Bool((0..rows).map(|i| i % 3 != 1).collect()),
            b.reserve(rows as u64).unwrap(),
        )
        .unwrap();
        columns.push(MaterializedColumn { values, valid });
    }
    (Arc::new(Schema::new(fields)), Table { columns, rows })
}
#[test]
fn roundtrip_all_numeric_types_nulls_empty_and_multiple_batches() {
    let dir = tempfile::tempdir().unwrap();
    let b = budget();
    let token = CancellationToken::new();
    for f in [Format::ArrowIpc, Format::Parquet] {
        for rows in [0, 3, 8193] {
            let (schema, t) = fixture(rows, &b);
            let path = dir.path().join("data");
            write(
                &path,
                f,
                schema.clone(),
                &t,
                WriteMode::Overwrite,
                &b,
                &token,
            )
            .unwrap();
            let out = read(&path, f, &b, &token).unwrap();
            assert_eq!(out.schema, schema);
            assert_eq!(out.table.rows, rows);
            for (a, c) in out.table.columns.iter().zip(&t.columns) {
                assert_eq!(a.values.dtype(), c.values.dtype());
                for i in 0..rows {
                    assert_eq!(a.valid.buffer.get(i), c.valid.buffer.get(i));
                    if c.valid.buffer.get(i) == Scalar::Bool(true) {
                        assert_eq!(a.values.buffer.get(i), c.values.buffer.get(i));
                    }
                }
            }
        }
    }
    assert_eq!(b.usage().used, 0);
}
#[test]
fn malformed_truncated_budget_and_cancel_preserve_destination() {
    let dir = tempfile::tempdir().unwrap();
    let b = budget();
    let token = CancellationToken::new();
    for f in [Format::ArrowIpc, Format::Parquet] {
        let path = dir.path().join("data");
        let (schema, t) = fixture(3, &b);
        write(
            &path,
            f,
            schema.clone(),
            &t,
            WriteMode::Overwrite,
            &b,
            &token,
        )
        .unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            write(
                &path,
                f,
                schema.clone(),
                &t,
                WriteMode::CreateNew,
                &b,
                &token
            )
            .unwrap_err()
            .code,
            ErrorCode::IoExists
        );
        let small = MemoryBudget::new(Device::Cpu, 1);
        assert_eq!(
            write(
                &path,
                f,
                schema.clone(),
                &t,
                WriteMode::Overwrite,
                &small,
                &token
            )
            .unwrap_err()
            .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(
            read(&path, f, &small, &token).err().unwrap().code,
            ErrorCode::MemoryBudgetExceeded
        );
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert_eq!(
            write(&path, f, schema, &t, WriteMode::Overwrite, &b, &cancelled)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(
            read(&path, f, &b, &cancelled).err().unwrap().code,
            ErrorCode::Cancelled
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        let baseline = b.usage().used;
        for n in [0, 1, 8, bytes.len() / 2, bytes.len() - 1] {
            fs::write(&path, &bytes[..n]).unwrap();
            assert!(read(&path, f, &b, &token).is_err());
            assert_eq!(b.usage().used, baseline);
        }
        fs::write(&path, b"not a columnar file").unwrap();
        assert_eq!(
            read(&path, f, &b, &token).err().unwrap().code,
            ErrorCode::IoFormat
        );
    }
}
#[test]
fn malformed_thrift_container_never_enters_native_parser() {
    // A list claiming 2^28 entries in a five-byte footer.
    let mut bytes = b"PAR1".to_vec();
    let footer = [0x19, 0xf5, 0xff, 0xff, 0xff, 0x7f];
    bytes.extend(footer);
    bytes.extend((footer.len() as u32).to_le_bytes());
    bytes.extend(b"PAR1");
    assert!(parquet_footer(&bytes).is_err());
}
#[test]
fn scheduler_charges_results_and_records_file_operations() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 4,
        cpu_memory_budget: 1 << 28,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let path = dir.path().join("file");
    let (tx, rx) = std::sync::mpsc::channel();
    schedule_write(
        &ctx,
        vec![],
        path.clone(),
        Format::ArrowIpc,
        WriteMode::CreateNew,
        CancellationToken::new(),
        Box::new(move |r| tx.send(r.map(|_| ())).unwrap()),
    )
    .unwrap();
    rx.recv().unwrap().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    schedule_read(
        &ctx,
        path,
        Format::ArrowIpc,
        CancellationToken::new(),
        Box::new(move |r| tx.send(r).unwrap()),
    )
    .unwrap();
    let result = rx.recv().unwrap().unwrap();
    assert_eq!(result.table.rows, 0);
    let m = ctx.metrics();
    assert!(m.cpu_memory.used > 0);
    drop(result);
    assert_eq!(ctx.metrics().cpu_memory.used, 0);
}
#[test]
fn oracle_numeric_files_and_explicit_schema_errors() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/io-columnar");
    let b = budget();
    let token = CancellationToken::new();
    for (file, format) in [
        ("polars.arrow", Format::ArrowIpc),
        ("polars.parquet", Format::Parquet),
    ] {
        let result = read(&root.join(file), format, &b, &token).unwrap();
        assert_eq!(result.table.rows, 3);
        assert_eq!(
            result.table.columns[0].values.buffer.get(0),
            Scalar::Int(i64::MIN)
        );
        assert_eq!(
            result.table.columns[1].values.buffer.get(2),
            Scalar::UInt(u64::MAX)
        );
        assert_eq!(
            result.table.columns[0].valid.buffer.get(1),
            Scalar::Bool(false)
        );
    }
    for (file, format) in [
        ("pandas.parquet", Format::Parquet),
        ("strings.arrow", Format::ArrowIpc),
        ("compressed.arrow", Format::ArrowIpc),
    ] {
        assert!(read(&root.join(file), format, &b, &token).is_err());
    }
}

#[test]
fn publication_is_terminal_even_when_cancelled_before_completion() {
    for format in [Format::ArrowIpc, Format::Parquet] {
        let ctx = Context::new(ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: 4,
            cpu_memory_budget: 1 << 28,
            gpu_memory_budget: 0,
        })
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output");
        let output = path.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        ctx.submit(
            OperationKind::FileWrite,
            "columnar_publication_test",
            Device::Cpu,
            CancellationToken::new(),
            Box::new(move |env| {
                let (schema, table) = fixture(3, &env.budget);
                write(
                    &output,
                    format,
                    schema,
                    &table,
                    WriteMode::CreateNew,
                    &env.budget,
                    &env.token,
                )?;
                env.token.cancel();
                Ok(JobOutput::Unit)
            }),
            Box::new(move |r| tx.send(r.map(|_| ())).unwrap()),
        )
        .unwrap();
        rx.recv().unwrap().unwrap();
        ctx.close();
        assert_eq!(ctx.metrics().total_succeeded, 1);
        assert_eq!(ctx.metrics().total_cancelled, 0);
        assert_eq!(
            read(&path, format, &budget(), &CancellationToken::new())
                .unwrap()
                .table
                .rows,
            3
        );
    }
}
#[test]
fn queued_cancellation_does_not_publish() {
    let ctx = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 4,
        cpu_memory_budget: 1 << 28,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let (release, wait) = std::sync::mpsc::channel();
    let (started, ready) = std::sync::mpsc::channel();
    ctx.submit(
        OperationKind::Compute,
        "block",
        Device::Cpu,
        CancellationToken::new(),
        Box::new(move |_| {
            started.send(()).unwrap();
            wait.recv().unwrap();
            Ok(JobOutput::Unit)
        }),
        Box::new(|_| {}),
    )
    .unwrap();
    ready.recv().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output");
    let token = CancellationToken::new();
    let (tx, rx) = std::sync::mpsc::channel();
    schedule_write(
        &ctx,
        vec![],
        path.clone(),
        Format::ArrowIpc,
        WriteMode::CreateNew,
        token.clone(),
        Box::new(move |r| tx.send(r.map(|_| ())).unwrap()),
    )
    .unwrap();
    token.cancel();
    release.send(()).unwrap();
    assert_eq!(rx.recv().unwrap().unwrap_err().code, ErrorCode::Cancelled);
    ctx.close();
    assert!(!path.exists());
    assert_eq!(ctx.metrics().cpu_memory.used, 0);
}

#[test]
fn detects_external_size_change_on_open_handle_and_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data");
    fs::write(&path, b"before").unwrap();
    let file = File::open(&path).unwrap();
    let before = file.metadata().unwrap();
    check_unchanged(&path, &file, &before).unwrap();
    fs::write(&path, b"changed length").unwrap();
    assert_eq!(
        check_unchanged(&path, &file, &before).unwrap_err().code,
        ErrorCode::IoModified
    );
}
#[test]
fn queued_write_snapshots_at_actual_start() {
    let ctx = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 4,
        cpu_memory_budget: 1 << 28,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let (release, wait) = std::sync::mpsc::channel();
    let (started, ready) = std::sync::mpsc::channel();
    ctx.submit(
        OperationKind::Compute,
        "block",
        Device::Cpu,
        CancellationToken::new(),
        Box::new(move |_| {
            started.send(()).unwrap();
            wait.recv().unwrap();
            Ok(JobOutput::Unit)
        }),
        Box::new(|_| {}),
    )
    .unwrap();
    ready.recv().unwrap();
    let array = ctx.array(vec![1], Buffer::I64(vec![1])).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output");
    let (tx, rx) = std::sync::mpsc::channel();
    schedule_write(
        &ctx,
        vec![InputColumn {
            name: "value".into(),
            values: Expr::input(&array),
            valid: None,
        }],
        path.clone(),
        Format::Parquet,
        WriteMode::CreateNew,
        CancellationToken::new(),
        Box::new(move |r| tx.send(r.map(|_| ())).unwrap()),
    )
    .unwrap();
    let token = CancellationToken::new();
    let turn = array.write_ticket().wait(&token).unwrap();
    let dense = DenseArray::new(vec![1], Buffer::I64(vec![i64::MAX]), Reservation::none()).unwrap();
    array.assign(&turn, &[], &dense, &token).unwrap();
    drop(turn);
    release.send(()).unwrap();
    rx.recv().unwrap().unwrap();
    ctx.close();
    let output = read(&path, Format::Parquet, &budget(), &token).unwrap();
    assert_eq!(
        output.table.columns[0].values.buffer.get(0),
        Scalar::Int(i64::MAX)
    );
}

#[test]
fn writer_rejects_unreadable_schema_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output");
    let b = budget();
    let token = CancellationToken::new();
    let fields = (0..1024)
        .map(|i| Field::new(i.to_string(), DataType::Int32, false))
        .collect::<Vec<_>>();
    check_schema(&Schema::new(fields.clone())).unwrap();
    check_schema(&Schema::new(vec![Field::new(
        "é".repeat(2048),
        DataType::Int32,
        false,
    )]))
    .unwrap();
    let mut excessive = fields;
    excessive.push(Field::new("extra", DataType::Int32, false));
    for schema in [
        Schema::new(excessive),
        Schema::new(vec![Field::new("é".repeat(2049), DataType::Int32, false)]),
    ] {
        for format in [Format::ArrowIpc, Format::Parquet] {
            fs::write(&path, b"original").unwrap();
            let error = write(
                &path,
                format,
                Arc::new(schema.clone()),
                &Table {
                    columns: vec![],
                    rows: 0,
                },
                WriteMode::Overwrite,
                &b,
                &token,
            )
            .unwrap_err();
            assert_eq!(error.code, ErrorCode::DTypeUnsupported);
            assert_eq!(fs::read(&path).unwrap(), b"original");
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }
}
