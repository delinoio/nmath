// Derived from NumPy 2.5.4 random/src/distributions/distributions.c.
// Copyright (c) 2005-2025, NumPy Developers. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019 Kevin Sheppard. All rights reserved.
// See NUMPY-RANDOM-LICENSE.txt and NUMPY-LICENSE.txt for the complete license.

#[path = "constants.rs"]
mod constants;
use constants::*;

use super::Pcg64;

impl Pcg64 {
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let r = self.next_u64();
            let idx = (r & 0xff) as usize;
            let r = r >> 8;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * WI_DOUBLE[idx];
            if r & 1 != 0 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.next_f64()).ln_1p();
                    let yy = -(-self.next_f64()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 1 != 0 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            }
            if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]).mul_add(self.next_f64(), FI_DOUBLE[idx])
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    pub fn standard_normal_f32(&mut self) -> f32 {
        loop {
            let r = self.next_u32();
            let idx = (r & 0xff) as usize;
            let r = r >> 8;
            let rabs = (r >> 1) & 0x007f_ffff;
            let mut x = rabs as f32 * WI_FLOAT[idx];
            if r & 1 != 0 {
                x = -x;
            }
            if rabs < KI_FLOAT[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R_F * (-self.next_f32()).ln_1p();
                    let yy = -(-self.next_f32()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 1 != 0 {
                            -(ZIGGURAT_NOR_R_F + xx)
                        } else {
                            ZIGGURAT_NOR_R_F + xx
                        };
                    }
                }
            }
            if (((FI_FLOAT[idx - 1] - FI_FLOAT[idx]).mul_add(self.next_f32(), FI_FLOAT[idx]))
                as f64)
                < (-0.5 * x as f64 * x as f64).exp()
            {
                return x;
            }
        }
    }

    pub fn standard_exponential(&mut self) -> f64 {
        loop {
            let r = self.next_u64() >> 3;
            let idx = (r & 0xff) as usize;
            let r = r >> 8;
            let x = r as f64 * WE_DOUBLE[idx];
            if r < KE_DOUBLE[idx] {
                return x;
            }
            if idx == 0 {
                return ZIGGURAT_EXP_R - (-self.next_f64()).ln_1p();
            }
            if (FE_DOUBLE[idx - 1] - FE_DOUBLE[idx]).mul_add(self.next_f64(), FE_DOUBLE[idx])
                < (-x).exp()
            {
                return x;
            }
        }
    }
}

/// NumPy's small-integer buffer is local to each distribution call. It is not
/// the persistent PCG64 uint32 cache; unused bits are discarded after the call.
#[derive(Default)]
pub(super) struct IntegerBuffer {
    bits: u32,
    remaining: u32,
}

impl IntegerBuffer {
    fn next(&mut self, rng: &mut Pcg64, width: u32) -> u64 {
        if width == 64 {
            return rng.next_u64();
        }
        if width == 32 {
            return rng.next_u32() as u64;
        }
        if self.remaining == 0 {
            self.bits = rng.next_u32();
            self.remaining = 32 / width;
        }
        let value = self.bits & ((1 << width) - 1);
        self.bits >>= width;
        self.remaining -= 1;
        value as u64
    }

    pub(super) fn bounded(&mut self, rng: &mut Pcg64, range: u64, width: u32) -> u64 {
        if range == 0 {
            return 0;
        }
        let width = if width == 64 && range <= u32::MAX as u64 {
            32
        } else {
            width
        };
        let max = u64::MAX >> (64 - width);
        if range == max {
            return self.next(rng, width);
        }
        let exclusive = range + 1;
        let threshold = (max - range) % exclusive;
        loop {
            let product = self.next(rng, width) as u128 * exclusive as u128;
            if (product as u64 & max) >= threshold {
                return (product >> width) as u64;
            }
        }
    }
}
