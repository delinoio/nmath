# SPDX-License-Identifier: Apache-2.0
"""Development-only NumPy 2.5.4 oracle; no Python runtime/build dependency."""
from pathlib import Path
import struct
import numpy as np

assert np.__version__ == "2.5.4"
rng = np.random.default_rng(271828)

def word(x):
    return int(np.float32(x).view(np.uint32))

edge = [0, 1, 2, 3, 0x007fffff, 0x00800000, 0x00800001,
        0x3f000000, 0x3f7fffff, 0x3f800000, 0x3f800001, 0x40000000,
        0x40400000, 0x7f7fffff, 0x7f800000, 0x7fc00000, 0x7f800001]
edge += [x | 0x80000000 for x in edge]
exponents = edge + [word(x) for x in [-150, -149, -127, -126, -25, -24, -3,
                                     -2, -1, -.5, .25, .5, 1.5, 3, 4, 5,
                                     23, 24, 25, 126, 127, 128, 149, 150,
                                     16777215, 16777216, 16777218]]
pairs = [(a, b) for a in edge for b in exponents]
pairs += [(int(a), int(b)) for a, b in rng.integers(0, 2**32, (4096, 2), dtype=np.uint32)]
# Finite fractional powers with normal and subnormal results.
a = rng.integers(1, 0x7f800000, 4096, dtype=np.uint32)
b = rng.uniform(-4, 4, 4096).astype(np.float32).view(np.uint32)
pairs += [(int(x), int(y)) for x, y in zip(a, b)]
# Near-one bases and large exponents amplify logarithm errors.
for a in range(0x3f800000 - 32, 0x3f800000 + 33):
    for y in [-1e9, -1e8, -1e6, -100, -.5, .5, 100, 1e6, 1e8, 1e9]:
        pairs.append((a, word(y)))
# Power-of-two output ties and adjacent exponents around under/overflow.
for e in [-149, -148, -126, -70, -1, 1, 63, 127]:
    a = word(2.0**e)
    for target in [-150, -149, -126, 127, 128]:
        y = np.float32(target / e)
        for b in [np.nextafter(y, np.float32(-np.inf)), y,
                  np.nextafter(y, np.float32(np.inf))]:
            pairs.append((a, word(b)))
# Odd integer powers: signs and final-rounding ties.
for x in [1.5, 3, 5, 7, 9, 11, 13, 15, 17, 33, 65, 129, 257]:
    for y in range(-25, 26):
        pairs += [(word(x), word(y)), (word(-x), word(y))]
# Exact rational roots can produce integer-power rounding ties too.
for root, power in [(11, 7), (17, 6), (257, 3)]:
    for root_degree in [2, 4]:
        for scale in [-20, -2, 0, 2, 10]:
            x = (root * 2.0**scale)**root_degree
            if float(np.float32(x)) == x:
                pairs.append((word(x), word(power / root_degree)))
# Deliberately target subnormal outputs and the overflow neighborhood.
for a in rng.integers(1, 0x7f800000, 2048, dtype=np.uint32):
    x = float(a.view(np.float32))
    for target in [rng.uniform(-151, -125), rng.uniform(126, 129)]:
        pairs.append((int(a), word(target / np.log2(x))))
a = np.array([x for x, _ in pairs], dtype=np.uint32).view(np.float32)
b = np.array([y for _, y in pairs], dtype=np.uint32).view(np.float32)
with np.errstate(all="ignore"):
    expected = np.power(a, b).view(np.uint32)
path = Path(__file__).with_name("power.bin")
with path.open("wb") as f:
    for (x, y), z in zip(pairs, expected):
        f.write(struct.pack("<III", x, y, int(z)))
print(f"NumPy {np.__version__}; seed 271828; {len(pairs)} power pairs; {path.stat().st_size} bytes")

# An independent high-precision diagnostic does not replace the pinned NumPy
# comparison or authorize a tolerance. Confirm rounded words at two precisions.
from decimal import Decimal, localcontext
from fractions import Fraction
from math import isqrt


def rounded_reference(xword, yword, numpy_word, precision):
    x, y = float(np.uint32(xword).view(np.float32)), float(np.uint32(yword).view(np.float32))
    # Special-value semantics reuse the NumPy words. Finite nontrivial powers
    # below are independently evaluated, including subnormal inputs/results.
    if not np.isfinite(x) or not np.isfinite(y) or x == 0 or y == 0 or x == 1:
        return numpy_word
    if x < 0 and y != np.trunc(y):
        return numpy_word
    with localcontext() as context:
        context.prec = precision
        dx, dy = Decimal(abs(x)), Decimal(y)
        logarithm = dx.ln() * dy
        sign = 0x80000000 if x < 0 and int(y) % 2 else 0
        if logarithm > 100:
            return sign | 0x7f800000
        if logarithm < -110:
            return sign
        value = dx ** dy
        # Decimal transcendental rounding can land on either side of an exact
        # rational tie. Resolve rational roots and integer powers as fractions.
        exact = None
        p, q = y.as_integer_ratio()
        if abs(p) <= 1024 and q <= 256:
            rn, rd = abs(x).as_integer_ratio()
            while q > 1:
                sn, sd = isqrt(rn), isqrt(rd)
                if sn * sn != rn or sd * sd != rd:
                    break
                rn, rd, q = sn, sd, q // 2
            if q == 1:
                exact = Fraction(rn, rd) ** p
        # binary32 overflow rounds at 2^128 - 2^103.
        if value >= Decimal(2)**128 - Decimal(2)**103:
            return sign | 0x7f800000
        with np.errstate(all="ignore"):
            center = word(float(value))
        center = min(center, 0x7f7fffff)
        candidates = range(max(0, center - 1), min(0x7f7fffff, center + 1) + 1)
        if exact is not None:
            best = min(candidates, key=lambda w: (abs(Fraction(float(np.uint32(w).view(np.float32))) - exact), w & 1))
        else:
            best = min(candidates, key=lambda w: (abs(Decimal(float(np.uint32(w).view(np.float32))) - value), w & 1))
        return sign | best

reference = []
for (x, y), z in zip(pairs, expected):
    a = rounded_reference(x, y, int(z), 100)
    b = rounded_reference(x, y, int(z), 160)
    assert a == b, (x, y, a, b)
    reference.append(a)
path = Path(__file__).with_name("power_rounded.bin")
path.write_bytes(b"".join(struct.pack("<I", w) for w in reference))
print(f"Decimal 100/160 digits agree on {len(reference)} rounded references")
