use std::{
    fs,
    io::{Cursor, Write},
    sync::mpsc,
    time::Duration,
};

use nmath_core::{
    Buffer, CancellationToken, Context, ContextOptions, Device, ErrorCode, Expr, Scalar,
    buffer::DenseArray,
    context::JobOutput,
    io::WriteMode,
    io_npz::{self, NamedArray, NpzCompression},
    memory::{MemoryBudget, Reservation},
};
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

use crate as nmath_core;

const STORED: &[u8] = include_bytes!("../tests/fixtures/io-npz/stored.npz");
const DEFLATED: &[u8] = include_bytes!("../tests/fixtures/io-npz/deflated.npz");
fn budget() -> MemoryBudget {
    MemoryBudget::new(Device::Cpu, 128 << 20)
}
fn token() -> CancellationToken {
    CancellationToken::new()
}
fn named(name: &str, shape: Vec<usize>, buffer: Buffer) -> NamedArray {
    NamedArray {
        name: name.into(),
        array: DenseArray::new(shape, buffer, Reservation::none()).unwrap(),
    }
}
fn zip_members(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    for (name, bytes) in entries {
        zip.start_file(
            *name,
            SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
        )
        .unwrap();
        zip.write_all(bytes).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

#[test]
fn exact_numpy_254_fixtures_and_both_roundtrips() {
    for fixture in [STORED, DEFLATED] {
        let budget = budget();
        let archive = io_npz::decode_npz(fixture, &budget, &token()).unwrap();
        assert_eq!(archive.arrays.len(), 6);
        assert_eq!(archive.arrays[0].array.shape, vec![2, 3]);
        assert_eq!(
            archive.arrays[0].array.buffer,
            Buffer::I8(vec![-128, 0, 127, 1, -2, 3])
        );
        assert_eq!(
            archive.arrays[1].array.buffer,
            Buffer::U64(vec![0, u64::MAX])
        );
        let Buffer::F32(v) = &archive.arrays[2].array.buffer else {
            panic!()
        };
        assert_eq!(
            v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![
                1.5f32.to_bits(),
                (-0.0f32).to_bits(),
                f32::INFINITY.to_bits()
            ]
        );
        assert_eq!(archive.arrays[3].array.shape, vec![2, 0, 3]);
        assert_eq!(archive.arrays[3].array.buffer, Buffer::I16(vec![]));
        assert_eq!(archive.arrays[4].array.shape, Vec::<usize>::new());
        assert_eq!(archive.arrays[4].array.buffer, Buffer::Bool(vec![true]));
        assert_eq!(archive.arrays[5].name, "한글");
        assert_eq!(archive.arrays[5].array.buffer, Buffer::U16(vec![65535, 0]));
        for compression in [NpzCompression::Stored, NpzCompression::Deflated] {
            let (bytes, charge) =
                io_npz::encode_npz(&archive.arrays, compression, &budget, &token()).unwrap();
            if let Some(directory) = std::env::var_os("NMATH_NPZ_ORACLE_OUTPUT") {
                let suffix = if compression == NpzCompression::Stored {
                    "stored"
                } else {
                    "deflated"
                };
                fs::write(
                    std::path::Path::new(&directory).join(format!("rust-{suffix}.npz")),
                    &bytes,
                )
                .unwrap();
            }
            let decoded = io_npz::decode_npz(&bytes, &budget, &token()).unwrap();
            for (a, b) in archive.arrays.iter().zip(&decoded.arrays) {
                assert_eq!(a.name, b.name);
                assert_eq!(a.array.shape, b.array.shape);
                assert_eq!(a.array.buffer, b.array.buffer);
            }
            drop(decoded);
            drop(bytes);
            drop(charge);
        }
        drop(archive);
        assert_eq!(budget.usage().used, 0);
    }
    let budget = budget();
    let archive = io_npz::decode_npz(
        include_bytes!("../tests/fixtures/io-npz/positional.npz"),
        &budget,
        &token(),
    )
    .unwrap();
    assert_eq!(archive.arrays[0].name, "arr_0");
    assert_eq!(archive.arrays[1].name, "arr_1");
}

#[test]
fn rejects_truncation_crc_duplicates_traversal_pickle_and_false_lengths_without_leaks() {
    let budget = budget();
    for n in 0..STORED.len() {
        assert!(io_npz::decode_npz(&STORED[..n], &budget, &token()).is_err());
        assert_eq!(budget.usage().used, 0);
    }
    let npy = nmath_core::io::encode_npy(&[1], &Buffer::I32(vec![7])).unwrap();
    for name in [
        "../escape.npy",
        "/root.npy",
        "C:drive.npy",
        "dir/name.npy",
        "dir\\name.npy",
        "wrong.txt",
        ".npy",
    ] {
        assert_eq!(
            io_npz::decode_npz(&zip_members(&[(name, &npy)]), &budget, &token())
                .unwrap_err()
                .code,
            ErrorCode::IoFormat
        );
        assert_eq!(budget.usage().used, 0);
    }
    let mut duplicate = zip_members(&[("aa.npy", &npy), ("bb.npy", &npy)]);
    for p in 0..duplicate.len() - 6 {
        if &duplicate[p..p + 6] == b"bb.npy" {
            duplicate[p..p + 6].copy_from_slice(b"aa.npy");
        }
    }
    assert!(io_npz::decode_npz(&duplicate, &budget, &token()).is_err());
    let mut corrupt = zip_members(&[("a.npy", &npy)]);
    let data = corrupt.windows(6).position(|w| w == b"\x93NUMPY").unwrap();
    corrupt[data + 15] ^= 1;
    assert!(io_npz::decode_npz(&corrupt, &budget, &token()).is_err());
    let mut length = zip_members(&[("a.npy", &npy)]);
    let central = length.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    length[central + 24..central + 28].copy_from_slice(&1u32.to_le_bytes());
    assert!(io_npz::decode_npz(&length, &budget, &token()).is_err());
    let pickle = io_npz::decode_npz(
        include_bytes!("../tests/fixtures/io-npz/object.npz"),
        &budget,
        &token(),
    )
    .unwrap_err();
    assert_eq!(pickle.code, ErrorCode::IoFormat);
    assert!(!pickle.message.contains("forbidden"));
    assert_eq!(budget.usage().used, 0);
}

#[test]
fn budgets_archive_metadata_decompression_results_and_output_before_allocation() {
    for limit in [0, 100, 4 << 20, (4 << 20) + 100_000] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        let error = io_npz::decode_npz(
            include_bytes!("../tests/fixtures/io-npz/bomb.npz"),
            &budget,
            &token(),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::MemoryBudgetExceeded);
        assert_eq!(budget.usage().used, 0);
    }
    let budget = budget();
    let archive = io_npz::decode_npz(STORED, &budget, &token()).unwrap();
    let peak = budget.usage().peak;
    drop(archive);
    let tight = MemoryBudget::new(Device::Cpu, peak - 1);
    assert_eq!(
        io_npz::decode_npz(STORED, &tight, &token())
            .unwrap_err()
            .code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(tight.usage().used, 0);
    let tight = MemoryBudget::new(Device::Cpu, 4 << 20);
    assert_eq!(
        io_npz::encode_npz(
            &[named("a", vec![1], Buffer::I32(vec![4]))],
            NpzCompression::Deflated,
            &tight,
            &token()
        )
        .unwrap_err()
        .code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(tight.usage().used, 0);
}

#[test]
fn empty_names_extensions_and_atomic_failure_preserve_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output.npz");
    let budget = budget();
    let arrays = [named("a.npy", vec![1], Buffer::I16(vec![4]))];
    io_npz::save_npz(
        &path,
        &arrays,
        NpzCompression::Stored,
        WriteMode::CreateNew,
        &budget,
        &token(),
    )
    .unwrap();
    let old = fs::read(&path).unwrap();
    assert_eq!(
        io_npz::load_npz(&path, &budget, &token()).unwrap().arrays[0].name,
        "a.npy"
    );
    assert_eq!(
        io_npz::save_npz(
            &path,
            &[],
            NpzCompression::Deflated,
            WriteMode::CreateNew,
            &budget,
            &token()
        )
        .unwrap_err()
        .code,
        ErrorCode::IoExists
    );
    let cancelled = token();
    cancelled.cancel();
    assert_eq!(
        io_npz::save_npz(
            &path,
            &[],
            NpzCompression::Stored,
            WriteMode::Overwrite,
            &budget,
            &cancelled
        )
        .unwrap_err()
        .code,
        ErrorCode::Cancelled
    );
    let tight = MemoryBudget::new(Device::Cpu, 1);
    assert!(
        io_npz::save_npz(
            &path,
            &arrays,
            NpzCompression::Deflated,
            WriteMode::Overwrite,
            &tight,
            &token()
        )
        .is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), old);
    io_npz::save_npz(
        &path,
        &[],
        NpzCompression::Deflated,
        WriteMode::Overwrite,
        &budget,
        &token(),
    )
    .unwrap();
    assert!(
        io_npz::load_npz(&path, &budget, &token())
            .unwrap()
            .arrays
            .is_empty()
    );
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    assert_eq!(budget.usage().used, 0);
}

fn context() -> Context {
    Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 4,
        cpu_memory_budget: 256 << 20,
        gpu_memory_budget: 0,
    })
    .unwrap()
}
fn done() -> (
    nmath_core::context::Done,
    mpsc::Receiver<nmath_core::Result<JobOutput>>,
) {
    let (tx, rx) = mpsc::channel();
    (Box::new(move |v| tx.send(v).unwrap()), rx)
}
fn occupy(
    ctx: &Context,
) -> (
    nmath_core::sync::FifoGuard,
    mpsc::Receiver<nmath_core::Result<JobOutput>>,
) {
    let a = ctx.array(vec![1], Buffer::I32(vec![0])).unwrap();
    let guard = a.write_ticket().wait(&token()).unwrap();
    let (done, rx) = done();
    ctx.assign(&a, vec![], &Expr::scalar(Scalar::Int(1)), token(), done)
        .unwrap();
    (guard, rx)
}
#[test]
fn context_queued_cancel_start_snapshots_cross_context_and_result_release() {
    let ctx = context();
    let other = context();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.npz");
    let a = ctx.array(vec![1], Buffer::I32(vec![1])).unwrap();
    let (callback, _rx) = done();
    assert_eq!(
        other
            .save_npz(
                vec![("a".into(), a.clone())],
                path.clone(),
                NpzCompression::Stored,
                WriteMode::CreateNew,
                token(),
                callback
            )
            .unwrap_err()
            .code,
        ErrorCode::ContextMismatch
    );
    let (guard, block) = occupy(&ctx);
    let cancelled = token();
    let (callback, rx) = done();
    ctx.save_npz(
        vec![("a".into(), a.clone())],
        path.clone(),
        NpzCompression::Stored,
        WriteMode::CreateNew,
        cancelled.clone(),
        callback,
    )
    .unwrap();
    cancelled.cancel();
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    assert!(!path.exists());
    let (callback, rx) = done();
    ctx.save_npz(
        vec![("a".into(), a.clone())],
        path.clone(),
        NpzCompression::Deflated,
        WriteMode::CreateNew,
        token(),
        callback,
    )
    .unwrap();
    let source = other.array(vec![1], Buffer::I32(vec![9])).unwrap();
    let snap = source.snapshot(&budget()).unwrap();
    let ticket = a.write_ticket();
    let turn = ticket.wait(&token()).unwrap();
    a.assign(&turn, &[], &snap, &token()).unwrap();
    drop(turn);
    drop(guard);
    block.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    let (tx, rx) = mpsc::channel();
    ctx.load_npz(path, token(), Box::new(move |v| tx.send(v).unwrap()))
        .unwrap();
    let result = rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    assert_eq!(result.arrays[0].1.context_id(), ctx.id());
    assert_eq!(result.arrays[0].1.to_buffer(), Buffer::I32(vec![9]));
    drop(result);
    drop(a);
    ctx.close();
    other.close();
    assert_eq!(ctx.metrics().cpu_memory.used, 0);
}

#[test]
fn running_compression_cancel_releases_memory_and_preserves_destination() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.npz");
    fs::write(&path, b"original").unwrap();
    let budget = budget();
    let worker_budget = budget.clone();
    let cancelled = token();
    let worker_token = cancelled.clone();
    let output = path.clone();
    let arrays = vec![named(
        "large",
        vec![16 << 20],
        Buffer::U8(vec![13; 16 << 20]),
    )];
    let worker = std::thread::spawn(move || {
        io_npz::save_npz(
            &output,
            &arrays,
            NpzCompression::Deflated,
            WriteMode::Overwrite,
            &worker_budget,
            &worker_token,
        )
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while budget.usage().used < 16 << 20 {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    cancelled.cancel();
    assert_eq!(
        worker.join().unwrap().unwrap_err().code,
        ErrorCode::Cancelled
    );
    assert_eq!(fs::read(&path).unwrap(), b"original");
    assert_eq!(budget.usage().used, 0);
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn modification_during_decode_fails_and_releases_decoded_arrays() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("input.npz");
    fs::write(&path, STORED).unwrap();
    let budget = budget();
    let result = super::load_npz_with_decoder(&path, &budget, &token(), |bytes, budget, token| {
        let decoded = io_npz::decode_npz(bytes, budget, token)?;
        fs::write(&path, b"changed while decoding").unwrap();
        Ok(decoded)
    });
    assert_eq!(result.unwrap_err().code, ErrorCode::IoModified);
    assert_eq!(budget.usage().used, 0);
}

#[test]
fn rejects_header_disagreement_encryption_oversized_directories_and_invalid_npy() {
    let budget = budget();
    let npy = nmath_core::io::encode_npy(&[1], &Buffer::I32(vec![3])).unwrap();
    let original = zip_members(&[("a.npy", &npy)]);
    for local_offset in [6, 8, 14, 18, 22, 30] {
        let mut data = original.clone();
        data[local_offset] ^= 1;
        assert!(
            io_npz::decode_npz(&data, &budget, &token()).is_err(),
            "offset {local_offset}"
        );
        assert_eq!(budget.usage().used, 0);
    }
    let mut data = original.clone();
    let end = data.len() - 22;
    data[end + 10..end + 12].copy_from_slice(&65535u16.to_le_bytes());
    assert_eq!(
        io_npz::decode_npz(&data, &budget, &token())
            .unwrap_err()
            .code,
        ErrorCode::IoFormat
    );
    let mut npy = npy;
    npy.pop();
    assert_eq!(
        io_npz::decode_npz(&zip_members(&[("a.npy", &npy)]), &budget, &token())
            .unwrap_err()
            .code,
        ErrorCode::IoFormat
    );
    assert_eq!(budget.usage().used, 0);
}

#[test]
fn running_decode_cancel_and_queued_read_cancel_return_no_arrays() {
    let budget = budget();
    let cancelled = token();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("input.npz");
    fs::write(&path, STORED).unwrap();
    let result =
        super::load_npz_with_decoder(&path, &budget, &cancelled, |bytes, budget, token| {
            token.cancel();
            io_npz::decode_npz(bytes, budget, token)
        });
    assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
    assert_eq!(budget.usage().used, 0);
    let ctx = context();
    let (guard, block) = occupy(&ctx);
    let cancelled = token();
    let (tx, rx) = mpsc::channel();
    ctx.load_npz(
        path,
        cancelled.clone(),
        Box::new(move |v| tx.send(v).unwrap()),
    )
    .unwrap();
    cancelled.cancel();
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    drop(guard);
    block.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    ctx.close();
    assert_eq!(ctx.metrics().cpu_memory.used, 0);
}

#[test]
fn invalid_output_and_publication_failure_leave_no_temporary_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.npz");
    fs::write(&path, b"old").unwrap();
    let budget = budget();
    for arrays in [
        vec![named("../bad", vec![1], Buffer::I8(vec![1]))],
        vec![
            named("a", vec![1], Buffer::I8(vec![1])),
            named("a", vec![1], Buffer::I8(vec![2])),
        ],
    ] {
        assert_eq!(
            io_npz::save_npz(
                &path,
                &arrays,
                NpzCompression::Stored,
                WriteMode::Overwrite,
                &budget,
                &token()
            )
            .unwrap_err()
            .code,
            ErrorCode::IoFormat
        );
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(budget.usage().used, 0);
    }
    let destination = dir.path().join("directory");
    fs::create_dir(&destination).unwrap();
    assert_eq!(
        io_npz::save_npz(
            &destination,
            &[],
            NpzCompression::Stored,
            WriteMode::Overwrite,
            &budget,
            &token()
        )
        .unwrap_err()
        .code,
        ErrorCode::IoFailed
    );
    assert!(destination.is_dir());
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    assert_eq!(budget.usage().used, 0);
    let tight = MemoryBudget::new(Device::Cpu, (STORED.len() - 1) as u64);
    fs::write(&path, STORED).unwrap();
    assert_eq!(
        io_npz::load_npz(&path, &tight, &token()).unwrap_err().code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(tight.usage().used, 0);
}

#[test]
fn complex_numpy_interchange_preserves_component_bits() {
    for fixture in [
        include_bytes!("../tests/fixtures/io-npz/complex-stored.npz").as_slice(),
        include_bytes!("../tests/fixtures/io-npz/complex-deflated.npz").as_slice(),
    ] {
        let budget = budget();
        let archive = io_npz::decode_npz(fixture, &budget, &token()).unwrap();
        assert_eq!(archive.arrays.len(), 5);
        assert_eq!(
            archive.arrays[0].array.buffer.dtype(),
            nmath_core::DType::Complex64
        );
        assert_eq!(
            archive.arrays[1].array.buffer.dtype(),
            nmath_core::DType::Complex128
        );
        assert_eq!(archive.arrays[2].array.shape, Vec::<usize>::new());
        assert_eq!(archive.arrays[3].array.shape, vec![2, 0, 3]);
        assert_eq!(archive.arrays[4].array.shape, vec![2, 2]);
        for compression in [NpzCompression::Stored, NpzCompression::Deflated] {
            let (bytes, charge) =
                io_npz::encode_npz(&archive.arrays, compression, &budget, &token()).unwrap();
            let decoded = io_npz::decode_npz(&bytes, &budget, &token()).unwrap();
            for (a, b) in archive.arrays.iter().zip(&decoded.arrays) {
                assert_eq!(a.name, b.name);
                // Compare encoded NPY bytes to preserve NaN payloads and signed
                // zeros; floating equality would not test this contract.
                assert_eq!(
                    nmath_core::io::encode_npy(&a.array.shape, &a.array.buffer).unwrap(),
                    nmath_core::io::encode_npy(&b.array.shape, &b.array.buffer).unwrap(),
                );
            }
            if let Some(directory) = std::env::var_os("NMATH_NPZ_ORACLE_OUTPUT") {
                let suffix = if compression == NpzCompression::Stored {
                    "stored"
                } else {
                    "deflated"
                };
                fs::write(
                    std::path::Path::new(&directory).join(format!("rust-complex-{suffix}.npz")),
                    &bytes,
                )
                .unwrap();
            }
            drop(decoded);
            drop(bytes);
            drop(charge);
        }
        drop(archive);
        assert_eq!(budget.usage().used, 0);
    }
}
