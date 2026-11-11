// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DenseArray, Device,
    linalg::{self, LinalgOp},
    memory::{MemoryBudget, Reservation},
};

fn budget() -> MemoryBudget {
    MemoryBudget::new(Device::Cpu, u64::MAX)
}

fn dense(shape: Vec<usize>, buffer: Buffer) -> DenseArray {
    DenseArray::new(shape, buffer, Reservation::none()).unwrap()
}

#[test]
fn linalg_division_keeps_finite_extreme_solutions() {
    let budget = budget();
    let token = CancellationToken::new();
    for scale in [2f64.powi(600), 2f64.powi(-600)] {
        let reciprocal = 1.0 / scale;
        let one = dense(vec![1, 1], Buffer::F64(vec![scale]));
        let rhs = dense(vec![1], Buffer::F64(vec![scale]));
        assert_eq!(
            linalg::execute(LinalgOp::Inv, &one, None, &budget, &token)
                .unwrap()
                .buffer,
            Buffer::F64(vec![reciprocal])
        );
        assert_eq!(
            linalg::execute(LinalgOp::Solve, &one, Some(&rhs), &budget, &token)
                .unwrap()
                .buffer,
            Buffer::F64(vec![1.0])
        );

        let diagonal = dense(vec![2, 2], Buffer::F64(vec![scale, 0.0, 0.0, scale]));
        let rhs = dense(vec![2], Buffer::F64(vec![scale, scale]));
        assert_eq!(
            linalg::execute(LinalgOp::Inv, &diagonal, None, &budget, &token)
                .unwrap()
                .buffer,
            Buffer::F64(vec![reciprocal, 0.0, 0.0, reciprocal])
        );
        assert_eq!(
            linalg::execute(LinalgOp::Solve, &diagonal, Some(&rhs), &budget, &token)
                .unwrap()
                .buffer,
            Buffer::F64(vec![1.0, 1.0])
        );

        let z = [scale, scale];
        let complex_one = dense(vec![1, 1], Buffer::C128(vec![z]));
        let complex_rhs = dense(vec![1], Buffer::C128(vec![z]));
        let solution = linalg::execute(
            LinalgOp::Solve,
            &complex_one,
            Some(&complex_rhs),
            &budget,
            &token,
        )
        .unwrap();
        let [real, imag] = solution.buffer.get(0).as_complex();
        assert!((real - 1.0).abs() < 1e-12 && imag.abs() < 1e-12);

        let lower = dense(vec![2, 2], Buffer::C128(vec![z, [0.0, 0.0], z, z]));
        let rhs = dense(vec![2], Buffer::C128(vec![z, [2.0 * scale, 2.0 * scale]]));
        let solution =
            linalg::execute(LinalgOp::Solve, &lower, Some(&rhs), &budget, &token).unwrap();
        for i in 0..2 {
            let [real, imag] = solution.buffer.get(i).as_complex();
            assert!((real - 1.0).abs() < 1e-12 && imag.abs() < 1e-12);
        }
    }
    assert_eq!(budget.usage().used, 0);
}
