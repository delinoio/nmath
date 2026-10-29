// SPDX-License-Identifier: Apache-2.0
// Power uses unsigned Q32.96 magnitudes, with signs handled separately.
// All limbs are little endian. Native floating operations may flush both
// inputs and outputs on Metal; keep this integer path until a replacement
// passes the same subnormal and special-value tests. No host evaluation is used.
const FX_ONE = vec4<u32>(0u, 0u, 0u, 1u);
// floor(ln(2) * 2^96).
const FX_LN2 = vec4<u32>(0xc9e3b398u, 0xd1cf79abu, 0xb17217f7u, 0u);
fn fx_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var r = vec4<u32>(0u); var carry = 0u;
    for (var i = 0u; i < 4u; i++) {
        let s = a[i] + b[i]; let t = s + carry;
        carry = select(0u, 1u, s < a[i] || t < s); r[i] = t;
    }
    return r;
}
fn fx_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var r = vec4<u32>(0u); var borrow = 0u;
    for (var i = 0u; i < 4u; i++) {
        let s = a[i] - b[i]; let t = s - borrow;
        borrow = select(0u, 1u, a[i] < b[i] || s < borrow); r[i] = t;
    }
    return r;
}
fn fx_less(a: vec4<u32>, b: vec4<u32>) -> bool {
    for (var i = 3; i >= 0; i--) { if a[i] != b[i] { return a[i] < b[i]; } }
    return false;
}
fn fx_right(a: vec4<u32>) -> vec4<u32> {
    return (a >> vec4<u32>(1u)) | vec4<u32>(a.y << 31u, a.z << 31u, a.w << 31u, 0u);
}
fn fx_left(a: vec4<u32>) -> vec4<u32> {
    return (a << vec4<u32>(1u)) | vec4<u32>(0u, a.x >> 31u, a.y >> 31u, a.z >> 31u);
}
fn fx_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    // Base-2^16 schoolbook product. Each sum fits exactly in u32, including
    // the previous digit and carry. Discard only the low 96 fractional bits.
    var digits: array<u32, 16>;
    for (var i = 0u; i < 8u; i++) {
        let ai = (a[i / 2u] >> ((i % 2u) * 16u)) & 65535u;
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let bj = (b[j / 2u] >> ((j % 2u) * 16u)) & 65535u;
            let s = ai * bj + digits[i + j] + carry;
            digits[i + j] = s & 65535u; carry = s >> 16u;
        }
        digits[i + 8u] = carry;
    }
    return vec4<u32>(digits[6] | (digits[7] << 16u), digits[8] | (digits[9] << 16u),
                     digits[10] | (digits[11] << 16u), digits[12] | (digits[13] << 16u));
}
fn fx_div_small(a: vec4<u32>, d: u32) -> vec4<u32> {
    var result = vec4<u32>(0u); var remainder = 0u;
    for (var i = 7; i >= 0; i--) {
        let shift = u32(i % 2) * 16u;
        let digit = (a[i / 2] >> shift) & 65535u;
        let n = (remainder << 16u) | digit;
        result[i / 2] |= (n / d) << shift; remainder = n % d;
    }
    return result;
}
fn fx_ratio(n: u32, d: u32) -> vec4<u32> {
    var r = n; var q = vec4<u32>(0u);
    for (var i = 0u; i < 96u; i++) {
        r <<= 1u; q = fx_left(q);
        if r >= d { r -= d; q.x |= 1u; }
    }
    return q;
}
fn fx_log_mantissa(m: u32) -> vec4<u32> {
    // ln(m / 2^23) = 2 * atanh((m - 2^23) / (m + 2^23)).
    // z < 1/3; 32 terms leave a series tail below 2^-105.
    let z = fx_ratio(m - 0x800000u, m + 0x800000u);
    let z2 = fx_mul(z, z); var term = z; var sum = z;
    for (var k = 1u; k < 32u; k++) {
        term = fx_mul(term, z2); sum = fx_add(sum, fx_div_small(term, 2u * k + 1u));
    }
    return fx_left(sum);
}
fn fx_exp(r: vec4<u32>) -> vec4<u32> {
    // 0 <= r < ln(2). The omitted Taylor tail after term 28 is below 2^-117.
    var term = FX_ONE; var sum = FX_ONE;
    for (var k = 1u; k <= 28u; k++) {
        term = fx_div_small(fx_mul(term, r), k); sum = fx_add(sum, term);
    }
    return sum;
}
fn fx_pack(value: vec4<u32>, exponent: i32, sign: u32) -> u32 {
    let sticky = select(0u, 1u, value.x != 0u || value.y != 0u || (value.z & 63u) != 0u);
    return pack(sign, (value.w << 26u) | (value.z >> 6u) | sticky, exponent);
}
fn pow_small_integer(m: u32, e: i32, power: u32, sign: u32) -> u32 {
    // Positive integer powers below 128 can contain exact binary32 rounding
    // ties (for example 11^7). Retain the exact low bits instead of passing
    // these ties through an approximation to ln/exp.
    var base = vec4<u32>(0u, 0u, (m & 0x7fffffu) << 9u, 1u);
    var base_exponent = e; var result = FX_ONE; var exponent = 0; var n = power;
    while n != 0u {
        if (n & 1u) != 0u {
            result = fx_mul(result, base); exponent += base_exponent;
            if result.w >= 2u { result = fx_right(result); exponent += 1; }
        }
        n >>= 1u;
        base = fx_mul(base, base); base_exponent *= 2;
        if base.w >= 2u { base = fx_right(base); base_exponent += 1; }
    }
    return fx_pack(result, exponent, sign);
}
fn integer_sqrt(n: u32) -> u32 {
    // The input is an odd significand below 2^24. Integer binary search
    // avoids a native sqrt and its device-specific rounding behavior.
    var low = 0u; var high = 4096u;
    while high - low > 1u {
        let middle = (high + low) / 2u;
        if middle * middle <= n { low = middle; } else { high = middle; }
    }
    return low;
}
fn powf(x: u32, y: u32) -> u32 {
    let ax = x & 0x7fffffffu; let ay = y & 0x7fffffffu;
    if ay == 0u || x == 0x3f800000u { return 0x3f800000u; }
    if nan(x) || nan(y) { return qnan(); }
    if inf(y) {
        if ax == 0x3f800000u { return 0x3f800000u; }
        return select(0u, 0x7f800000u, (ax > 0x3f800000u) == (y >> 31u == 0u));
    }
    let uy = unpack(y);
    var integral = false; var odd = false;
    if uy.y >= 24 { integral = true; }
    else if uy.y >= 0 {
        let shift = u32(23 - uy.y); let mask = (1u << shift) - 1u;
        integral = (u32(uy.x) & mask) == 0u;
        odd = integral && ((u32(uy.x) >> shift) & 1u) != 0u;
    }
    let sign = select(0u, x & 0x80000000u, odd);
    if ax == 0u { return sign | select(0u, 0x7f800000u, y >> 31u != 0u); }
    if inf(x) { return sign | select(0x7f800000u, 0u, y >> 31u != 0u); }
    if x >> 31u != 0u && !integral { return qnan(); }
    if ax == 0x3f800000u { return sign | 0x3f800000u; }
    if y == 0x3f800000u { return x; }
    if y == 0xbf800000u { return divf(0x3f800000u, x); }
    if y == 0x40000000u { return mulf(x, x); }
    let negative = (ax < 0x3f800000u) != (y >> 31u != 0u);
    // For binary32 x != 1, |ln(x)| > 2^-25 and < 104. These bounds
    // determine the rounded result without constructing a huge/tiny product.
    if uy.y > 31 { return sign | select(0x7f800000u, 0u, negative); }
    if uy.y < -40 { return 0x3f800000u; }
    let ux = unpack(x);
    if integral && y >> 31u == 0u && uy.y < 7 {
        return pow_small_integer(u32(ux.x), ux.y, u32(uy.x) >> u32(23 - uy.y), sign);
    }
    // A fractional dyadic exponent can also produce an exact rounding tie,
    // e.g. 121^3.5 = 11^7. Extract a rational root only when it is exact.
    // An odd numerator >= 3 raised to 2^4 exceeds 24 bits, so at most three
    // square roots are needed for a non-power-of-two binary32 input.
    if !integral && y >> 31u == 0u {
        let yz = countTrailingZeros(u32(uy.x));
        let numerator = u32(uy.x) >> yz;
        let denominator_bits = 23 - uy.y - i32(yz);
        if denominator_bits > 0 && denominator_bits <= 3 && numerator < 128u {
            let xz = countTrailingZeros(u32(ux.x));
            var odd_part = u32(ux.x) >> xz;
            let root_exponent = ux.y - 23 + i32(xz);
            var exact = (root_exponent % (1 << u32(denominator_bits))) == 0;
            for (var k = 0; k < denominator_bits; k++) {
                let root = integer_sqrt(odd_part);
                exact = exact && root * root == odd_part; odd_part = root;
            }
            if exact {
                let root = unpack(pack(0u, odd_part << 3u, root_exponent / (1 << u32(denominator_bits)) + 23));
                return pow_small_integer(u32(root.x), root.y, numerator, sign);
            }
        }
    }
    // Exact powers of two include the tie 2^-150, which rounds to zero.
    // Keep these boundaries independent of the transcendental approximation.
    if ux.x == 0x800000 {
        let product = u32(abs(ux.y)) * u32(uy.x);
        let shift = 23 - uy.y;
        if shift >= 0 && shift < 32 {
            let mask = (1u << u32(shift)) - 1u;
            if (product & mask) == 0u {
                let n = product >> u32(shift);
                if n > 150u { return sign | select(0x7f800000u, 0u, negative); }
                return pack(sign, 0x4000000u, select(i32(n), -i32(n), negative));
            }
        }
    }
    let lm = fx_log_mantissa(u32(ux.x));
    let le = fx_mul(FX_LN2, vec4<u32>(0u, 0u, 0u, u32(abs(ux.y))));
    var logarithm = fx_add(le, lm);
    if ux.y < 0 { logarithm = fx_sub(le, lm); }
    // Multiply by the exponent's normalized significand, then scale. Clamp
    // before each left shift so overflow cannot wrap back into the finite range.
    var t = fx_mul(logarithm, vec4<u32>(0u, 0u, (u32(uy.x) & 0x7fffffu) << 9u, 1u));
    for (var i = 0; i < uy.y; i++) {
        if t.w >= 128u { return sign | select(0x7f800000u, 0u, negative); }
        t = fx_left(t);
    }
    for (var i = 0; i < -uy.y; i++) { t = fx_right(t); }
    if t.w >= 105u { return sign | select(0x7f800000u, 0u, negative); }
    var exponent = 0;
    while !fx_less(t, FX_LN2) { t = fx_sub(t, FX_LN2); exponent += 1; }
    if negative {
        exponent = -exponent;
        if any(t != vec4<u32>(0u)) { t = fx_sub(FX_LN2, t); exponent -= 1; }
    }
    return fx_pack(fx_exp(t), exponent, sign);
}
