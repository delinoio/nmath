//! WGSL generation uses inferred shapes and dtypes, never input values.

use super::{elements, unsupported};
use crate::{
    dtype::DType,
    error::Result,
    expr::{Expr, ExprKind},
    kernels::{self, BinaryOp, ReduceOp, UnaryOp},
};

fn ty(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "f32",
        DType::Int32 => "i32",
        _ => "u32",
    }
}

// Expand a dense linear index into a broadcast operand's dense linear index.
fn index(shape: &[usize], output: &[usize], linear: &str) -> String {
    let mut terms = Vec::new();
    let mut src_stride = 1usize;
    let mut dst_stride = 1usize;
    for axis in (0..output.len()).rev() {
        if let Some(i) = axis.checked_sub(output.len() - shape.len()) {
            if shape[i] != 1 && shape[i] != 0 {
                terms.push(format!(
                    "(({linear} / {dst_stride}u) % {}u) * {src_stride}u",
                    output[axis].max(1)
                ));
            }
            src_stride *= shape[i];
        }
        dst_stride *= output[axis].max(1);
    }
    if terms.is_empty() {
        "0u".into()
    } else {
        terms.join(" + ")
    }
}

const FLOAT_HELPERS: &str = r#"
fn nan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn inf(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }
fn zero(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0u; }
fn qnan() -> f32 { return bitcast<f32>(0x7fc00000u); }
fn signedzero(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) & 0x80000000u); }
fn divf(x: f32, y: f32) -> f32 {
    let sign = (bitcast<u32>(x) ^ bitcast<u32>(y)) & 0x80000000u;
    if nan(x) || nan(y) || (zero(x) && zero(y)) || (inf(x) && inf(y)) { return qnan(); }
    if zero(y) || inf(x) { return bitcast<f32>(sign | 0x7f800000u); }
    if zero(x) || inf(y) { return bitcast<f32>(sign); }
    return x / y;
}
fn modf(x: f32, y: f32) -> f32 {
    if nan(x) || nan(y) || zero(y) || inf(x) { return qnan(); }
    var r = x % y;
    if inf(y) { r = x; }
    if zero(r) { return signedzero(y); }
    if (r < 0.0) != (y < 0.0) { r += y; }
    return r;
}
fn powf(x: f32, y: f32) -> f32 {
    if zero(y) || x == 1.0 { return 1.0; }
    if nan(x) || nan(y) { return qnan(); }
    let odd = abs(y) < 16777216.0 && (abs(y) % 2.0) == 1.0;
    if x < 0.0 && !inf(x) && !inf(y) && trunc(y) != y { return qnan(); }
    if abs(x) == 1.0 && inf(y) { return 1.0; }
    var r = pow(abs(x), y);
    if zero(x) { r = select(bitcast<f32>(0x7f800000u), 0.0, y > 0.0); }
    if inf(x) { r = select(0.0, bitcast<f32>(0x7f800000u), y > 0.0); }
    if inf(y) { r = select(0.0, bitcast<f32>(0x7f800000u), (abs(x) > 1.0) == (y > 0.0)); }
    if (bitcast<u32>(x) & 0x80000000u) != 0u && odd { r = -r; }
    return r;
}
fn minf(x: f32, y: f32) -> f32 {
    if nan(x) { return x; } if nan(y) { return y; }
    return select(y, x, x < y);
}
fn maxf(x: f32, y: f32) -> f32 {
    if nan(x) { return x; } if nan(y) { return y; }
    return select(y, x, x > y);
}
"#;

fn binary(op: BinaryOp, dtype: DType) -> String {
    let t = ty(dtype);
    let body = match op {
        BinaryOp::Add => "return x + y;".into(),
        BinaryOp::Sub => "return x - y;".into(),
        BinaryOp::Mul => "return x * y;".into(),
        BinaryOp::Div => "return divf(x, y);".into(),
        BinaryOp::Mod if dtype == DType::Float32 => "return modf(x, y);".into(),
        BinaryOp::Pow if dtype == DType::Float32 => "return powf(x, y);".into(),
        BinaryOp::Mod if dtype == DType::Int32 => {
            "if y == 0 || (x == bitcast<i32>(0x80000000u) && y == -1) { return 0; } var r = x % y; \
             if r != 0 && ((r < 0) != (y < 0)) { r += y; } return r;"
                .into()
        }
        BinaryOp::Mod => "if y == 0u { return 0u; } return x % y;".into(),
        BinaryOp::Pow => format!(
            "{} var p = u32(y); var base = x; var result = {t}(1); while p != 0u {{ if (p & 1u) \
             != 0u {{ result *= base; }} p >>= 1u; base *= base; }} return result;",
            if dtype == DType::Int32 {
                "if y < 0 { atomicStore(&status, 1u); return 0; }"
            } else {
                ""
            }
        ),
    };
    format!("fn operation(x: {t}, y: {t}) -> {t} {{ {body} }}")
}

pub(super) fn generate(expr: &Expr) -> Result<String> {
    let t = ty(expr.dtype());
    let count = elements(expr.shape())?;
    let mut source = format!(
        "@group(0) @binding(0) var<storage, read> a: array<{t}>;\n@group(0) @binding(2) \
         var<storage, read_write> out: array<{t}>;\n@group(0) @binding(3) var<storage, \
         read_write> status: atomic<u32>;\n@group(0) @binding(4) var<storage, read> params: \
         array<u32>;\n"
    );
    if expr.dtype() == DType::Float32 {
        source.push_str(FLOAT_HELPERS);
    }
    let body = match expr.kind() {
        ExprKind::Unary(op, _) => format!(
            "out[i] = {}a[i];",
            if *op == UnaryOp::Neg {
                if expr.dtype() == DType::UInt32 {
                    "0u - "
                } else {
                    "-"
                }
            } else {
                ""
            }
        ),
        ExprKind::Binary { op, lhs, rhs, .. } => {
            source.push_str(&format!(
                "@group(0) @binding(1) var<storage, read> b: array<{t}>;\n"
            ));
            source.push_str(&binary(*op, expr.dtype()));
            format!(
                "out[i] = operation(a[{}], b[{}]);",
                index(lhs.shape(), expr.shape(), "i"),
                index(rhs.shape(), expr.shape(), "i")
            )
        }
        ExprKind::Reduce(op, input, axes) => {
            let axes = kernels::reduce_axes(input.shape().len(), axes.as_deref())?;
            let n: usize = axes
                .iter()
                .enumerate()
                .filter(|(_, reduce)| **reduce)
                .map(|(i, _)| input.shape()[i])
                .product();
            let mut base = Vec::new();
            let mut red = Vec::new();
            let mut input_stride = 1usize;
            let mut output_stride = 1usize;
            let mut reduced_stride = 1usize;
            for axis in (0..input.shape().len()).rev() {
                let d = input.shape()[axis];
                if axes[axis] {
                    red.push(format!(
                        "((j / {reduced_stride}u) % {}u) * {input_stride}u",
                        d.max(1)
                    ));
                    reduced_stride *= d.max(1);
                } else {
                    base.push(format!(
                        "((i / {output_stride}u) % {}u) * {input_stride}u",
                        d.max(1)
                    ));
                    output_stride *= d.max(1);
                }
                input_stride *= d.max(1);
            }
            let base = if base.is_empty() {
                "0u".into()
            } else {
                base.join(" + ")
            };
            let red = if red.is_empty() {
                "0u".into()
            } else {
                red.join(" + ")
            };
            let init = match op {
                ReduceOp::Min | ReduceOp::Max => "a[base]".into(),
                _ => format!("{t}(0)"),
            };
            let update = match op {
                ReduceOp::Sum | ReduceOp::Mean => "v += x;".into(),
                ReduceOp::Min => format!("v = {}(v, x);", if t == "f32" { "minf" } else { "min" }),
                ReduceOp::Max => format!("v = {}(v, x);", if t == "f32" { "maxf" } else { "max" }),
            };
            let finish = if *op == ReduceOp::Mean {
                format!("out[i] = v; if params[3] == {n}u {{ out[i] = divf(v, f32({n}u)); }}")
            } else {
                "out[i] = v;".into()
            };
            format!(
                "let base = {base}; var v = {init}; if params[2] != 0u {{ v = out[i]; }} for (var \
                 j = params[2]; j < params[3]; j++) {{ let x = a[base + {red}]; {update} }} \
                 {finish}"
            )
        }
        ExprKind::MatMul(lhs, rhs) => {
            source.push_str(&format!(
                "@group(0) @binding(1) var<storage, read> b: array<{t}>;\n"
            ));
            let a = lhs.shape();
            let b = rhs.shape();
            let k = *a
                .last()
                .ok_or_else(|| unsupported("matmul requires rank at least one."))?;
            let m = if a.len() > 1 { a[a.len() - 2] } else { 1 };
            let n = if b.len() > 1 { b[b.len() - 1] } else { 1 };
            let ab = &a[..a.len().saturating_sub(2)];
            let bb = &b[..b.len().saturating_sub(2)];
            let batch = crate::shape::broadcast_shapes(ab, bb)?;
            let ai = index(ab, &batch, "batch");
            let bi = index(bb, &batch, "batch");
            format!(
                "let batch = i / {}u; let row = (i / {}u) % {}u; let col = i % {}u; let abase = \
                 ({ai}) * {}u; let bbase = ({bi}) * {}u; var v = {t}(0); if params[2] != 0u {{ v \
                 = out[i]; }} for (var j = params[2]; j < params[3]; j++) {{ v += a[abase + row * \
                 {k}u + j] * b[bbase + j * {n}u + col]; }} out[i] = v;",
                (m * n).max(1),
                n.max(1),
                m.max(1),
                n.max(1),
                m * k,
                k * n
            )
        }
        _ => return Err(unsupported("No shader exists for this operation.")),
    };
    // The status read keeps binding 3 present even when a kernel cannot fail.
    source.push_str(&format!(
        "\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>, \
         @builtin(num_workgroups) groups: vec3<u32>) {{ let i = gid.x + params[0]; if i >= \
         params[1] || i >= {count}u || atomicLoad(&status) != 0u {{ return; }} {body} }}"
    ));
    Ok(source)
}
