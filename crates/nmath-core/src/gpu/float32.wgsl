// IEEE binary32 values stay in integer registers and storage. WGSL permits
// native arithmetic to flush subnormals, including normal-input underflow.
// These kernels use round-to-nearest, ties-to-even with guard/round/sticky
// bits. Keep this path until a native replacement passes exact edge oracles.
fn nan(x: u32) -> bool { return (x & 0x7fffffffu) > 0x7f800000u; }
fn inf(x: u32) -> bool { return (x & 0x7fffffffu) == 0x7f800000u; }
fn zero(x: u32) -> bool { return (x & 0x7fffffffu) == 0u; }
fn qnan() -> u32 { return 0x7fc00000u; }
fn jam(x: u32, distance: u32) -> u32 {
    if distance == 0u { return x; }
    if distance >= 32u { return select(0u, 1u, x != 0u); }
    return (x >> distance) | select(0u, 1u, (x << (32u - distance)) != 0u);
}
// Nonzero finite magnitude: 24-bit normalized significand and unbiased exponent.
fn unpack(x: u32) -> vec2<i32> {
    var m = x & 0x7fffffu;
    var e = i32((x >> 23u) & 255u) - 127;
    if e == -127 {
        e = -126;
        while m < 0x800000u { m <<= 1u; e -= 1; }
    } else { m |= 0x800000u; }
    return vec2<i32>(i32(m), e);
}
fn pack(sign: u32, significand: u32, exponent: i32) -> u32 {
    var m = significand;
    var e = exponent;
    if m == 0u { return sign; }
    while m >= 0x8000000u { m = jam(m, 1u); e += 1; }
    while m < 0x4000000u { m <<= 1u; e -= 1; }
    if e < -126 { m = jam(m, u32(-126 - e)); e = -126; }
    let tail = m & 7u;
    m = (m >> 3u) + select(0u, 1u, tail > 4u || (tail == 4u && (m & 8u) != 0u));
    if m >= 0x1000000u { m >>= 1u; e += 1; }
    if e > 127 { return sign | 0x7f800000u; }
    if m < 0x800000u { return sign | m; }
    return sign | (u32(e + 127) << 23u) | (m & 0x7fffffu);
}
fn addf(x: u32, y: u32) -> u32 {
    if nan(x) || nan(y) { return qnan(); }
    if inf(x) { if inf(y) && ((x ^ y) >> 31u) != 0u { return qnan(); } return x; }
    if inf(y) { return y; }
    if zero(x) && zero(y) { return x & y; }
    if zero(x) { return y; } if zero(y) { return x; }
    var a = x; var b = y;
    if (a & 0x7fffffffu) < (b & 0x7fffffffu) { a = y; b = x; }
    let ua = unpack(a); let ub = unpack(b);
    let ma = u32(ua.x) << 3u;
    let mb = jam(u32(ub.x) << 3u, u32(ua.y - ub.y));
    if ((a ^ b) >> 31u) == 0u { return pack(a & 0x80000000u, ma + mb, ua.y); }
    if ma == mb { return 0u; }
    return pack(a & 0x80000000u, ma - mb, ua.y);
}
fn mulf(x: u32, y: u32) -> u32 {
    let sign = (x ^ y) & 0x80000000u;
    if nan(x) || nan(y) || (inf(x) && zero(y)) || (zero(x) && inf(y)) { return qnan(); }
    if inf(x) || inf(y) { return sign | 0x7f800000u; }
    if zero(x) || zero(y) { return sign; }
    let a = unpack(x); let b = unpack(y);
    let am = u32(a.x); let bm = u32(b.x);
    // Exact 24 x 24 -> 48 multiplication in two 32-bit words.
    let low = (am & 65535u) * (bm & 65535u);
    let middle = (am >> 16u) * (bm & 65535u) + (bm >> 16u) * (am & 65535u) + (low >> 16u);
    let high = (am >> 16u) * (bm >> 16u) + (middle >> 16u);
    let bottom = (middle << 16u) | (low & 65535u);
    let m = (high << 12u) | (bottom >> 20u) | select(0u, 1u, (bottom & 0xfffffu) != 0u);
    return pack(sign, m, a.y + b.y);
}
fn divf(x: u32, y: u32) -> u32 {
    let sign = (x ^ y) & 0x80000000u;
    if nan(x) || nan(y) || (zero(x) && zero(y)) || (inf(x) && inf(y)) { return qnan(); }
    if zero(y) || inf(x) { return sign | 0x7f800000u; }
    if zero(x) || inf(y) { return sign; }
    let a = unpack(x); let b = unpack(y);
    var r = u32(a.x); let d = u32(b.x);
    var q = 0u;
    // One extra bit for ratios below one prevents loss on normalization.
    for (var i = 0u; i < 28u; i++) {
        q <<= 1u;
        if r >= d { r -= d; q |= 1u; }
        r <<= 1u;
    }
    q |= select(0u, 1u, r != 0u);
    return pack(sign, q, a.y - b.y - 1);
}
fn lessf(x: u32, y: u32) -> bool {
    if zero(x) && zero(y) { return false; }
    if ((x ^ y) >> 31u) != 0u { return (x >> 31u) != 0u; }
    return select((x < y), (x > y), (x >> 31u) != 0u);
}
fn minf(x: u32, y: u32) -> u32 {
    if zero(x) && zero(y) { return x | y; }
    if nan(x) { return x; } if nan(y) { return y; }
    return select(y, x, lessf(x, y));
}
fn maxf(x: u32, y: u32) -> u32 {
    if zero(x) && zero(y) { return x & y; }
    if nan(x) { return x; } if nan(y) { return y; }
    return select(y, x, lessf(y, x));
}
fn modf(x: u32, y: u32) -> u32 {
    if nan(x) || nan(y) || zero(y) || inf(x) { return qnan(); }
    if zero(x) { return y & 0x80000000u; }
    var r = x;
    if !inf(y) && (x & 0x7fffffffu) >= (y & 0x7fffffffu) {
        let a = unpack(x); let b = unpack(y);
        var m = u32(a.x); let d = u32(b.x);
        for (var e = a.y; e > b.y; e--) { if m >= d { m -= d; } m <<= 1u; }
        if m >= d { m -= d; }
        r = pack(x & 0x80000000u, m << 3u, b.y);
    }
    if zero(r) { return y & 0x80000000u; }
    if ((r ^ y) >> 31u) != 0u { return addf(r, y); }
    return r;
}
