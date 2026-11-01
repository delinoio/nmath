// SPDX-License-Identifier: Apache-2.0
use super::*;
fn budget() -> MemoryBudget {
    MemoryBudget::new(Device::Cpu, 16 << 20)
}
#[test]
fn exact_nullable_roundtrip_and_atomic_policy() {
    let dir = tempfile::tempdir().unwrap();
    let b = budget();
    let token = CancellationToken::new();
    let source = dir.path().join("source.csv");
    std::fs::write(
        &source,
        "\"a,\"\"x\"\n-9223372036854775808\n\"\"\n9223372036854775807\n",
    )
    .unwrap();
    let schema = vec![("a,\"x".into(), DType::Int64)];
    let t = read(&source, Format::Csv, &schema, &b, &token).unwrap();
    assert_eq!(
        t.table.columns[0].values.buffer,
        Buffer::I64(vec![i64::MIN, 0, i64::MAX])
    );
    assert_eq!(
        t.table.columns[0].valid.buffer,
        Buffer::Bool(vec![true, false, true])
    );
    for f in [Format::Csv, Format::Json] {
        let path = dir.path().join(format!("{f:?}"));
        write(
            &path,
            f,
            &t.names,
            &t.table,
            WriteMode::CreateNew,
            &b,
            &token,
        )
        .unwrap();
        let copy = read(&path, f, &schema, &b, &token).unwrap();
        assert_eq!(
            copy.table.columns[0].values.buffer,
            t.table.columns[0].values.buffer
        );
        assert_eq!(
            copy.table.columns[0].valid.buffer,
            t.table.columns[0].valid.buffer
        );
        assert_eq!(
            write(
                &path,
                f,
                &t.names,
                &t.table,
                WriteMode::CreateNew,
                &b,
                &token
            )
            .unwrap_err()
            .code,
            ErrorCode::IoExists
        );
        let previous = std::fs::read(&path).unwrap();
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert_eq!(
            write(
                &path,
                f,
                &t.names,
                &t.table,
                WriteMode::Overwrite,
                &b,
                &cancelled
            )
            .unwrap_err()
            .code,
            ErrorCode::Cancelled
        );
        assert_eq!(std::fs::read(&path).unwrap(), previous);
        assert_eq!(
            write(
                &path,
                f,
                &t.names,
                &t.table,
                WriteMode::Overwrite,
                &MemoryBudget::new(Device::Cpu, 0),
                &token
            )
            .unwrap_err()
            .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(std::fs::read(&path).unwrap(), previous);
        write(
            &path,
            f,
            &t.names,
            &t.table,
            WriteMode::Overwrite,
            &b,
            &token,
        )
        .unwrap();
    }
}
#[test]
fn malformed_and_budget_release() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad");
    let b = budget();
    let token = CancellationToken::new();
    for (format, input) in [
        (Format::Csv, "a,b\n1\n"),
        (Format::Csv, "a\n\"broken"),
        (Format::Csv, "a\n\"1\"oops"),
        (Format::Csv, "a,a\n1,2"),
        (Format::Json, "[{\"a\":1,\"a\":2}]"),
        (Format::Json, "[{\"a\":{\"nested\":1}}]"),
        (Format::Json, "[{\"a\":1},{\"b\":2}]"),
        (Format::Json, "[NaN]"),
    ] {
        std::fs::write(&path, input).unwrap();
        assert!(read(&path, format, &vec![], &b, &token).is_err());
        assert_eq!(b.usage().used, 0);
    }
    std::fs::write(&path, "a\n1\n").unwrap();
    assert!(matches!(
        read(
            &path,
            Format::Csv,
            &vec![],
            &MemoryBudget::new(Device::Cpu, 1),
            &token
        ),
        Err(Error {
            code: ErrorCode::MemoryBudgetExceeded,
            ..
        })
    ));
    token.cancel();
    assert!(matches!(
        read(&path, Format::Csv, &vec![], &b, &token),
        Err(Error {
            code: ErrorCode::Cancelled,
            ..
        })
    ));
}
#[test]
fn json_uint64_and_empty_schema() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("uint.json");
    let b = budget();
    let token = CancellationToken::new();
    std::fs::write(&path, "[{\"u\":18446744073709551615},{\"u\":null}]").unwrap();
    let t = read(&path, Format::Json, &vec![], &b, &token).unwrap();
    assert_eq!(
        t.table.columns[0].values.buffer,
        Buffer::U64(vec![u64::MAX, 0])
    );
    std::fs::write(&path, "[]").unwrap();
    let t = read(
        &path,
        Format::Json,
        &vec![("u".into(), DType::UInt64)],
        &b,
        &token,
    )
    .unwrap();
    assert_eq!(t.table.columns[0].values.buffer, Buffer::U64(vec![]));
}

#[test]
fn pinned_original_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/io-text");
    let schema = vec![
        ("signed".into(), DType::Int64),
        ("unsigned".into(), DType::UInt64),
        ("small".into(), DType::Int8),
        ("flag".into(), DType::Bool),
    ];
    for original in ["pandas", "polars"] {
        for (format, ext) in [(Format::Csv, "csv"), (Format::Json, "json")] {
            let b = budget();
            let t = read(
                &root.join(format!("{original}.{ext}")),
                format,
                &schema,
                &b,
                &CancellationToken::new(),
            )
            .unwrap();
            assert_eq!(
                t.table.columns[0].values.buffer,
                Buffer::I64(vec![i64::MIN, 0, i64::MAX])
            );
            assert_eq!(
                t.table.columns[1].values.buffer,
                Buffer::U64(vec![u64::MAX, 0, 0])
            );
            assert_eq!(t.table.columns[2].values.buffer, Buffer::I8(vec![1, 0, -2]));
            assert_eq!(
                t.table.columns[3].values.buffer,
                Buffer::Bool(vec![true, false, false])
            );
            assert_eq!(
                t.table.columns[0].valid.buffer,
                Buffer::Bool(vec![true, false, true])
            );
            drop(t);
            assert_eq!(b.usage().used, 0);
        }
    }
}
#[test]
fn rejection_preserves_destination_and_cleans_temp() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("destination");
    std::fs::create_dir(&path).unwrap();
    let b = budget();
    let token = CancellationToken::new();
    let source = dir.path().join("source");
    std::fs::write(&source, "a\n1\n").unwrap();
    let t = read(&source, Format::Csv, &vec![], &b, &token).unwrap();
    assert!(
        write(
            &path,
            Format::Csv,
            &t.names,
            &t.table,
            WriteMode::Overwrite,
            &b,
            &token
        )
        .is_err()
    );
    assert!(path.is_dir());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
}

#[test]
fn source_change_and_running_cancel_are_detected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source");
    let b = budget();
    let token = CancellationToken::new();
    std::fs::write(&path, "a\n1\n").unwrap();
    let result = read_with(&path, Format::Csv, &vec![], &b, &token, || {
        std::fs::write(&path, "a\n12\n").unwrap()
    });
    assert!(matches!(
        result,
        Err(Error {
            code: ErrorCode::IoModified,
            ..
        })
    ));
    assert_eq!(b.usage().used, 0);
    let result = read_with(&path, Format::Csv, &vec![], &b, &token, || token.cancel());
    assert!(matches!(
        result,
        Err(Error {
            code: ErrorCode::Cancelled,
            ..
        })
    ));
    assert_eq!(b.usage().used, 0);
}

#[test]
fn unsupported_or_overflowing_values_do_not_narrow() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source");
    let b = budget();
    let token = CancellationToken::new();
    for input in [
        "[{\"a\":18446744073709551616}]",
        "[{\"a\":-9223372036854775809}]",
        "[{\"a\":\"secret-value\"}]",
    ] {
        std::fs::write(&path, input).unwrap();
        let error = read(&path, Format::Json, &vec![], &b, &token)
            .err()
            .unwrap();
        assert_eq!(error.code, ErrorCode::IoFormat);
        assert!(!error.message.contains("secret-value"));
    }
    std::fs::write(&path, "a\n128\n").unwrap();
    assert!(
        read(
            &path,
            Format::Csv,
            &vec![("a".into(), DType::Int8)],
            &b,
            &token
        )
        .is_err()
    );
}

#[test]
fn every_real_dtype_and_float_extremes_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let b = budget();
    let token = CancellationToken::new();
    let buffers = vec![
        Buffer::Bool(vec![true, false]),
        Buffer::I8(vec![i8::MIN, i8::MAX]),
        Buffer::I16(vec![i16::MIN, i16::MAX]),
        Buffer::I32(vec![i32::MIN, i32::MAX]),
        Buffer::I64(vec![i64::MIN, i64::MAX]),
        Buffer::U8(vec![0, u8::MAX]),
        Buffer::U16(vec![0, u16::MAX]),
        Buffer::U32(vec![0, u32::MAX]),
        Buffer::U64(vec![0, u64::MAX]),
        Buffer::F32(vec![f32::MIN_POSITIVE, f32::MAX]),
        Buffer::F64(vec![f64::from_bits(1), f64::MAX]),
    ];
    for buffer in buffers {
        let dtype = buffer.dtype();
        let names = vec!["value".to_owned()];
        let table = Table {
            rows: 2,
            pivot_columns: None,
            columns: vec![MaterializedColumn {
                values: DenseArray::new(vec![2], buffer.clone(), Reservation::none()).unwrap(),
                valid: DenseArray::new(
                    vec![2],
                    Buffer::Bool(vec![true, true]),
                    Reservation::none(),
                )
                .unwrap(),
            }],
        };
        for f in [Format::Csv, Format::Json] {
            let path = dir.path().join("out");
            write(&path, f, &names, &table, WriteMode::Overwrite, &b, &token).unwrap();
            let copy = read(&path, f, &vec![("value".into(), dtype)], &b, &token).unwrap();
            assert_eq!(copy.table.columns[0].values.buffer, buffer);
        }
    }
}
#[test]
fn queued_file_jobs_obey_admission_and_snapshot_at_start() {
    use std::{sync::mpsc, time::Duration};
    let ctx = Context::new(crate::ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 1,
        cpu_memory_budget: 16 << 20,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let a = ctx.array(vec![1], Buffer::I64(vec![1])).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out");
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    ctx.submit(
        OperationKind::Compute,
        "text_blocker",
        Device::Cpu,
        CancellationToken::new(),
        Box::new(move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(JobOutput::Unit)
        }),
        Box::new(|_| {}),
    )
    .unwrap();
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let (done_tx, done_rx) = mpsc::channel();
    ctx.write_text(
        path.clone(),
        Format::Csv,
        vec![("a".into(), Expr::input(&a), None)],
        WriteMode::CreateNew,
        CancellationToken::new(),
        Box::new(move |r| done_tx.send(r).unwrap()),
    )
    .unwrap();
    assert!(
        ctx.read_text(
            path.clone(),
            Format::Csv,
            vec![],
            CancellationToken::new(),
            Box::new(|_| {})
        )
        .is_err()
    );
    let replacement = DenseArray::new(vec![], Buffer::I64(vec![7]), Reservation::none()).unwrap();
    let token = CancellationToken::new();
    {
        let turn = a.write_ticket().wait(&token).unwrap();
        a.assign(&turn, &[crate::Selector::Index(0)], &replacement, &token)
            .unwrap();
    }
    release_tx.send(()).unwrap();
    done_rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(std::fs::read_to_string(path).unwrap(), "a\n7\n");
    ctx.close();
}
