// SPDX-License-Identifier: Apache-2.0
//! Single-matrix factorizations in double-precision complex workspace:
//! Cholesky, Householder QR, Hermitian Jacobi eigendecomposition, one-sided
//! Jacobi SVD and LU-based log-determinant. Matrices are row-major. The batch
//! driver in `linalg` owns dtype conversion, budgets and output layout.
//!
//! QR, eigenvector and singular-vector signs and phases are not unique, as in
//! LAPACK; callers must rely only on the factorization invariants.

use crate::{
    error::{Error, ErrorCode, Result},
    numerical::C,
    sync::CancellationToken,
};

const ZERO: C = C::new(0.0, 0.0);
const ONE: C = C::new(1.0, 0.0);
/// Jacobi sweeps before reporting non-convergence. Cyclic Jacobi converges
/// quadratically; well-conditioned inputs need well under 20 sweeps.
const MAX_SWEEPS: usize = 100;

fn no_convergence() -> Error {
    Error::new(
        ErrorCode::LinalgNoConvergence,
        "The factorization did not converge.",
    )
}

/// Lower Cholesky factor from the lower triangle of `a` (n×n). The strict
/// upper triangle of the result is zero.
pub fn cholesky(a: &[C], n: usize, token: &CancellationToken) -> Result<Vec<C>> {
    let mut l = vec![ZERO; n * n];
    for j in 0..n {
        token.check()?;
        let mut d = a[j * n + j].re;
        for k in 0..j {
            d -= l[j * n + k].norm_sqr();
        }
        if d.is_nan() || d <= 0.0 {
            return Err(Error::new(
                ErrorCode::LinalgNotPositiveDefinite,
                "The matrix is not positive definite.",
            ));
        }
        let d = d.sqrt();
        l[j * n + j] = C::new(d, 0.0);
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k].conj();
            }
            l[i * n + j] = v / d;
        }
    }
    Ok(l)
}

/// Householder QR of `a` (m×n). Returns `Q` (m×m) and `R` (m×n, upper
/// triangular).
pub fn qr(a: &[C], m: usize, n: usize, token: &CancellationToken) -> Result<(Vec<C>, Vec<C>)> {
    let mut r = a.to_vec();
    let mut q = vec![ZERO; m * m];
    for i in 0..m {
        q[i * m + i] = ONE;
    }
    let mut v = vec![ZERO; m];
    for k in 0..m.min(n) {
        token.check()?;
        let norm = (k..m).map(|i| r[i * n + k].norm_sqr()).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let x0 = r[k * n + k];
        let phase = if x0 == ZERO { ONE } else { x0 / x0.norm() };
        let alpha = -phase * norm;
        for i in k..m {
            v[i] = r[i * n + k];
        }
        v[k] -= alpha;
        let vnorm = (k..m).map(|i| v[i].norm_sqr()).sum::<f64>().sqrt();
        if vnorm == 0.0 {
            continue;
        }
        for i in k..m {
            v[i] /= vnorm;
        }
        // R <- (I - 2 v v^H) R
        for j in 0..n {
            let dot: C = (k..m).map(|i| v[i].conj() * r[i * n + j]).sum();
            for i in k..m {
                r[i * n + j] -= v[i] * dot * 2.0;
            }
        }
        // Q <- Q (I - 2 v v^H)
        for i in 0..m {
            let dot: C = (k..m).map(|j| q[i * m + j] * v[j]).sum();
            for j in k..m {
                q[i * m + j] -= dot * v[j].conj() * 2.0;
            }
        }
        for i in k + 1..m {
            r[i * n + k] = ZERO;
        }
    }
    Ok((q, r))
}

/// Eigendecomposition of the Hermitian matrix described by one triangle of
/// `a` (n×n): the lower triangle, or the upper triangle when `upper`. The
/// imaginary parts of the diagonal are ignored, as in LAPACK. Returns
/// ascending eigenvalues and the eigenvectors as columns (n×n).
pub fn eigh(
    a: &[C],
    n: usize,
    upper: bool,
    token: &CancellationToken,
) -> Result<(Vec<f64>, Vec<C>)> {
    let mut h = vec![ZERO; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = if upper {
                a[j * n + i].conj()
            } else {
                a[i * n + j]
            };
            h[i * n + j] = v;
            h[j * n + i] = v.conj();
        }
        h[i * n + i] = C::new(h[i * n + i].re, 0.0);
    }
    let mut vecs = vec![ZERO; n * n];
    for i in 0..n {
        vecs[i * n + i] = ONE;
    }
    let total: f64 = h.iter().map(|z| z.norm_sqr()).sum();
    if !total.is_finite() {
        return Err(no_convergence());
    }
    let tolerance = f64::EPSILON * f64::EPSILON * total;
    let mut converged = false;
    for _ in 0..MAX_SWEEPS {
        token.check()?;
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| h[i * n + j].norm_sqr())
            .sum();
        if off <= tolerance {
            converged = true;
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let hpq = h[p * n + q];
                let abs = hpq.norm();
                if abs == 0.0 {
                    continue;
                }
                // Scale column/row q so that h[p][q] becomes real and positive.
                let d = (hpq / abs).conj();
                for k in 0..n {
                    h[k * n + q] *= d;
                    h[q * n + k] *= d.conj();
                    vecs[k * n + q] *= d;
                }
                let (hpp, hqq) = (h[p * n + p].re, h[q * n + q].re);
                let theta = (hqq - hpp) / (2.0 * abs);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (kp, kq) = (h[k * n + p], h[k * n + q]);
                    h[k * n + p] = kp * c - kq * s;
                    h[k * n + q] = kp * s + kq * c;
                }
                for k in 0..n {
                    let (pk, qk) = (h[p * n + k], h[q * n + k]);
                    h[p * n + k] = pk * c - qk * s;
                    h[q * n + k] = pk * s + qk * c;
                }
                h[p * n + q] = ZERO;
                h[q * n + p] = ZERO;
                for k in 0..n {
                    let (kp, kq) = (vecs[k * n + p], vecs[k * n + q]);
                    vecs[k * n + p] = kp * c - kq * s;
                    vecs[k * n + q] = kp * s + kq * c;
                }
            }
        }
    }
    if !converged {
        return Err(no_convergence());
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| h[x * n + x].re.total_cmp(&h[y * n + y].re));
    let values = order.iter().map(|&i| h[i * n + i].re).collect();
    let mut sorted = vec![ZERO; n * n];
    for (col, &src) in order.iter().enumerate() {
        for row in 0..n {
            sorted[row * n + col] = vecs[row * n + src];
        }
    }
    Ok((values, sorted))
}

/// Extends the first `filled` orthonormal columns of `basis` (rows×cols) to
/// an orthonormal set by Gram-Schmidt over the standard basis vectors.
fn complete(basis: &mut [C], rows: usize, cols: usize, filled: &[bool]) {
    let mut candidate = 0;
    for col in 0..cols {
        if filled[col] {
            continue;
        }
        loop {
            let mut v = vec![ZERO; rows];
            v[candidate % rows] = ONE;
            candidate += 1;
            // Two Gram-Schmidt passes keep the basis orthogonal to rounding.
            for _ in 0..2 {
                for other in 0..cols {
                    if other == col || !(filled[other] || other < col) {
                        continue;
                    }
                    let dot: C = (0..rows)
                        .map(|r| basis[r * cols + other].conj() * v[r])
                        .sum();
                    for (r, x) in v.iter_mut().enumerate() {
                        *x -= basis[r * cols + other] * dot;
                    }
                }
            }
            let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if norm > 0.5 {
                for (r, x) in v.into_iter().enumerate() {
                    basis[r * cols + col] = x / norm;
                }
                break;
            }
            if candidate > 2 * rows + cols {
                break;
            }
        }
    }
}

/// Singular value decomposition of `a` (m×n), m ≥ n, by one-sided Jacobi.
/// Returns `U` (m×m when `full`, otherwise m×n), descending singular values
/// (n) and `V` (n×n) with right singular vectors as columns.
fn svd_tall(
    a: &[C],
    m: usize,
    n: usize,
    full: bool,
    token: &CancellationToken,
) -> Result<(Vec<C>, Vec<f64>, Vec<C>)> {
    let mut u = a.to_vec();
    let mut v = vec![ZERO; n * n];
    for i in 0..n {
        v[i * n + i] = ONE;
    }
    let total: f64 = u.iter().map(|z| z.norm_sqr()).sum();
    if !total.is_finite() {
        return Err(no_convergence());
    }
    let mut converged = n < 2;
    for _ in 0..MAX_SWEEPS {
        if converged {
            break;
        }
        token.check()?;
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let alpha: f64 = (0..m).map(|i| u[i * n + p].norm_sqr()).sum();
                let beta: f64 = (0..m).map(|i| u[i * n + q].norm_sqr()).sum();
                let gamma: C = (0..m).map(|i| u[i * n + p].conj() * u[i * n + q]).sum();
                let g = gamma.norm();
                if g == 0.0 || g <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let phase = (gamma / g).conj();
                let zeta = (beta - alpha) / (2.0 * g);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (x, y) = (u[i * n + p], u[i * n + q] * phase);
                    u[i * n + p] = x * c - y * s;
                    u[i * n + q] = x * s + y * c;
                }
                for i in 0..n {
                    let (x, y) = (v[i * n + p], v[i * n + q] * phase);
                    v[i * n + p] = x * c - y * s;
                    v[i * n + q] = x * s + y * c;
                }
            }
        }
        converged = !rotated;
    }
    if !converged {
        return Err(no_convergence());
    }
    let norms: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| u[i * n + j].norm_sqr()).sum::<f64>().sqrt())
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| norms[y].total_cmp(&norms[x]));
    let cols = if full { m } else { n };
    let mut left = vec![ZERO; m * cols];
    let mut filled = vec![false; cols];
    let largest = norms.iter().copied().fold(0.0, f64::max);
    let mut right = vec![ZERO; n * n];
    let mut values = Vec::with_capacity(n);
    for (col, &src) in order.iter().enumerate() {
        let sigma = norms[src];
        values.push(sigma);
        for row in 0..n {
            right[row * n + col] = v[row * n + src];
        }
        if sigma > largest * f64::EPSILON * m as f64 && sigma > 0.0 {
            for row in 0..m {
                left[row * cols + col] = u[row * n + src] / sigma;
            }
            filled[col] = true;
        }
    }
    complete(&mut left, m, cols, &filled);
    Ok((left, values, right))
}

/// Singular value decomposition of `a` (m×n). Returns `U` (m×m or m×k),
/// descending singular values (k = min(m, n)) and `Vh` (n×n or k×n).
pub fn svd(
    a: &[C],
    m: usize,
    n: usize,
    full: bool,
    token: &CancellationToken,
) -> Result<(Vec<C>, Vec<f64>, Vec<C>)> {
    if m >= n {
        let (u, s, v) = svd_tall(a, m, n, full, token)?;
        let mut vh = vec![ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                vh[i * n + j] = v[j * n + i].conj();
            }
        }
        return Ok((u, s, vh));
    }
    // A = (A^H)^H: factor the tall conjugate transpose and swap the factors.
    let mut b = vec![ZERO; n * m];
    for i in 0..m {
        for j in 0..n {
            b[j * m + i] = a[i * n + j].conj();
        }
    }
    let (ub, s, vb) = svd_tall(&b, n, m, full, token)?;
    // U_A = V_b (m×m); Vh_A = U_b^H (cols×n).
    let cols = if full { n } else { m };
    let mut vh = vec![ZERO; cols * n];
    for i in 0..cols {
        for j in 0..n {
            vh[i * n + j] = ub[j * cols + i].conj();
        }
    }
    Ok((vb, s, vh))
}

/// Sign (a unit complex value, or zero) and natural log of |det(a)| by
/// partial-pivot LU. A singular matrix returns `(0, -inf)`.
pub fn slogdet(a: &[C], n: usize, token: &CancellationToken) -> Result<(C, f64)> {
    let mut lu = a.to_vec();
    let mut sign = ONE;
    let mut log = 0.0;
    for k in 0..n {
        token.check()?;
        let pivot = (k..n)
            .max_by(|&x, &y| lu[x * n + k].norm().total_cmp(&lu[y * n + k].norm()))
            .unwrap_or(k);
        let p = lu[pivot * n + k];
        if p == ZERO {
            return Ok((ZERO, f64::NEG_INFINITY));
        }
        if pivot != k {
            for j in 0..n {
                lu.swap(k * n + j, pivot * n + j);
            }
            sign = -sign;
        }
        let abs = p.norm();
        sign *= p / abs;
        log += abs.ln();
        for i in k + 1..n {
            let factor = lu[i * n + k] / p;
            for j in k + 1..n {
                let x = lu[k * n + j];
                lu[i * n + j] -= factor * x;
            }
        }
    }
    Ok((sign, log))
}

/// Eigenvalues and right eigenvectors of a general matrix `a` (n×n) through
/// Householder Hessenberg reduction and shifted complex QR iterations to
/// Schur form. Eigenvectors come from back substitution on the triangular
/// factor and have unit Euclidean norm. Eigenvalue order follows the Schur
/// deflation and is not sorted, as in LAPACK.
pub fn eig(a: &[C], n: usize, token: &CancellationToken) -> Result<(Vec<C>, Vec<C>)> {
    let mut h = a.to_vec();
    if h.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(no_convergence());
    }
    let mut z = vec![ZERO; n * n];
    for i in 0..n {
        z[i * n + i] = ONE;
    }
    // Hessenberg reduction: H <- P H P with P = I - 2 v v^H.
    let mut v = vec![ZERO; n];
    for k in 0..n.saturating_sub(2) {
        token.check()?;
        let norm = (k + 1..n)
            .map(|i| h[i * n + k].norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            continue;
        }
        let x0 = h[(k + 1) * n + k];
        let phase = if x0 == ZERO { ONE } else { x0 / x0.norm() };
        for i in k + 1..n {
            v[i] = h[i * n + k];
        }
        v[k + 1] += phase * norm;
        let vnorm = (k + 1..n).map(|i| v[i].norm_sqr()).sum::<f64>().sqrt();
        for i in k + 1..n {
            v[i] /= vnorm;
        }
        for j in 0..n {
            let dot: C = (k + 1..n).map(|i| v[i].conj() * h[i * n + j]).sum();
            for i in k + 1..n {
                h[i * n + j] -= v[i] * dot * 2.0;
            }
        }
        for i in 0..n {
            let dot: C = (k + 1..n).map(|j| h[i * n + j] * v[j]).sum();
            for j in k + 1..n {
                h[i * n + j] -= dot * v[j].conj() * 2.0;
            }
            let dot: C = (k + 1..n).map(|j| z[i * n + j] * v[j]).sum();
            for j in k + 1..n {
                z[i * n + j] -= dot * v[j].conj() * 2.0;
            }
        }
        for i in k + 2..n {
            h[i * n + k] = ZERO;
        }
    }
    // Shifted QR iterations with Wilkinson shifts and deflation.
    let mut hi = n.saturating_sub(1);
    let mut iterations = 0;
    let mut since = 0;
    let limit = 30 * n.max(1);
    let mut rotations: Vec<(C, C)> = Vec::with_capacity(n);
    while hi > 0 {
        token.check()?;
        let mut l = hi;
        while l > 0 {
            let s = h[(l - 1) * n + l - 1].norm() + h[l * n + l].norm();
            if h[l * n + l - 1].norm() <= f64::EPSILON * s {
                h[l * n + l - 1] = ZERO;
                break;
            }
            l -= 1;
        }
        if l == hi {
            hi -= 1;
            since = 0;
            continue;
        }
        iterations += 1;
        since += 1;
        if iterations > limit {
            return Err(no_convergence());
        }
        let shift = if since % 10 == 0 {
            // Exceptional shift breaks rare cycles.
            h[hi * n + hi] + C::new(h[hi * n + hi - 1].norm(), 0.0)
        } else {
            let (a, b, c, d) = (
                h[(hi - 1) * n + hi - 1],
                h[(hi - 1) * n + hi],
                h[hi * n + hi - 1],
                h[hi * n + hi],
            );
            let tr = (a + d) * 0.5;
            let disc = ((a - d) * 0.5 * ((a - d) * 0.5) + b * c).sqrt();
            let (e1, e2) = (tr + disc, tr - disc);
            if (e1 - d).norm() <= (e2 - d).norm() {
                e1
            } else {
                e2
            }
        };
        for i in l..=hi {
            h[i * n + i] -= shift;
        }
        rotations.clear();
        for k in l..hi {
            let (x, y) = (h[k * n + k], h[(k + 1) * n + k]);
            let r = (x.norm_sqr() + y.norm_sqr()).sqrt();
            let (c, s) = if r == 0.0 {
                (ONE, ZERO)
            } else {
                (x / r, y / r)
            };
            for j in k..n {
                let (p, q) = (h[k * n + j], h[(k + 1) * n + j]);
                h[k * n + j] = c.conj() * p + s.conj() * q;
                h[(k + 1) * n + j] = -s * p + c * q;
            }
            rotations.push((c, s));
        }
        for (offset, &(c, s)) in rotations.iter().enumerate() {
            let k = l + offset;
            for i in 0..=(k + 2).min(hi) {
                let (p, q) = (h[i * n + k], h[i * n + k + 1]);
                h[i * n + k] = p * c + q * s;
                h[i * n + k + 1] = -p * s.conj() + q * c.conj();
            }
            for i in 0..n {
                let (p, q) = (z[i * n + k], z[i * n + k + 1]);
                z[i * n + k] = p * c + q * s;
                z[i * n + k + 1] = -p * s.conj() + q * c.conj();
            }
        }
        for i in l..=hi {
            h[i * n + i] += shift;
        }
    }
    let values: Vec<C> = (0..n).map(|i| h[i * n + i]).collect();
    let scale = h
        .iter()
        .map(|z| z.norm())
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let small = f64::EPSILON * scale;
    let mut vectors = vec![ZERO; n * n];
    let mut y = vec![ZERO; n];
    for k in 0..n {
        token.check()?;
        y.fill(ZERO);
        y[k] = ONE;
        for i in (0..k).rev() {
            let sum: C = (i + 1..=k).map(|j| h[i * n + j] * y[j]).sum();
            let mut d = h[i * n + i] - values[k];
            if d.norm() < small {
                // LAPACK perturbs zero pivots of repeated eigenvalues.
                d = C::new(small, 0.0);
            }
            y[i] = -sum / d;
        }
        let mut norm = 0.0;
        for i in 0..n {
            let x: C = (0..=k).map(|j| z[i * n + j] * y[j]).sum();
            vectors[i * n + k] = x;
            norm += x.norm_sqr();
        }
        let norm = norm.sqrt();
        for i in 0..n {
            vectors[i * n + k] /= norm;
        }
    }
    Ok((values, vectors))
}
