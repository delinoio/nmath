// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    buffer::Buffer,
    dtype::DType,
    memory::{Device, MemoryBudget},
    random::{Pcg64, RandomOperation},
    sync::CancellationToken,
};

fn bytes(buffer: &Buffer) -> Vec<u8> {
    macro_rules! flatten {
        ($v:expr) => {
            $v.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
    }
    match buffer {
        Buffer::Bool(v) => v.iter().map(|v| u8::from(*v)).collect(),
        Buffer::I8(v) => flatten!(v),
        Buffer::I16(v) => flatten!(v),
        Buffer::I32(v) => flatten!(v),
        Buffer::I64(v) => flatten!(v),
        Buffer::U8(v) => v.clone(),
        Buffer::U16(v) => flatten!(v),
        Buffer::U32(v) => flatten!(v),
        Buffer::U64(v) => flatten!(v),
        Buffer::F32(v) => flatten!(v),
        Buffer::F64(v) => flatten!(v),
        _ => unreachable!(),
    }
}

#[test]
fn pinned_mixed_calls_match_bytes_and_complete_pcg_state() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/nmath/test/fixtures/random");
    let mut rng = Pcg64::from_seed(12345);
    for line in std::fs::read_to_string(root.join("mixed.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<_> = line.split('|').collect();
        let dtype = DType::from_id(f[1]).unwrap();
        let op = match f[0] {
            "random" => RandomOperation::Random(dtype),
            "integers" => RandomOperation::Integers {
                low: f[2].parse().unwrap(),
                high: f[3].parse().unwrap(),
                endpoint: f[4] == "1",
                dtype,
            },
            "uniform" => RandomOperation::Uniform {
                low: f[2].parse().unwrap(),
                high: f[3].parse().unwrap(),
            },
            "standard_normal" => RandomOperation::StandardNormal(dtype),
            "normal" => RandomOperation::Normal {
                loc: f[2].parse().unwrap(),
                scale: f[3].parse().unwrap(),
            },
            "exponential" => RandomOperation::Exponential {
                scale: f[2].parse().unwrap(),
            },
            _ => unreachable!(),
        };
        let result = op
            .generate(
                &mut rng,
                &[f[5].parse().unwrap()],
                &MemoryBudget::new(Device::Cpu, 1_000_000),
                &CancellationToken::new(),
            )
            .unwrap();
        let expected = std::fs::read(root.join(f[6])).unwrap();
        let actual = bytes(&result.buffer);
        let mismatch = actual.iter().zip(&expected).position(|(a, b)| a != b);
        assert_eq!(actual.len(), expected.len());
        assert_eq!(
            mismatch, None,
            "{} {}: first differing byte {:?}",
            f[0], f[1], mismatch
        );
        assert_eq!(rng.state.to_string(), f[7]);
        assert_eq!(rng.inc.to_string(), f[8]);
        assert_eq!(u8::from(rng.has_uint32).to_string(), f[9]);
        assert_eq!(rng.uinteger.to_string(), f[10]);
    }
}

#[test]
fn invalid_bounds_budget_and_cancellation_preserve_local_state() {
    let mut rng = Pcg64::from_seed(42);
    rng.next_u32();
    let initial = rng;
    let budget = MemoryBudget::new(Device::Cpu, 100);
    for op in [
        RandomOperation::Integers {
            low: 0,
            high: 257,
            endpoint: false,
            dtype: DType::UInt8,
        },
        RandomOperation::Normal {
            loc: 0.0,
            scale: -1.0,
        },
        RandomOperation::Uniform {
            low: 2.0,
            high: 1.0,
        },
    ] {
        assert!(
            op.generate(&mut rng, &[3], &budget, &CancellationToken::new())
                .is_err()
        );
        assert_eq!(rng, initial);
    }
    let op = RandomOperation::Random(DType::Float32);
    assert!(
        op.generate(&mut rng, &[100], &budget, &CancellationToken::new())
            .is_err()
    );
    let token = CancellationToken::new();
    token.cancel();
    assert!(op.generate(&mut rng, &[3], &budget, &token).is_err());
    assert_eq!(rng, initial);
    assert_eq!(budget.usage().used, 0);
}
