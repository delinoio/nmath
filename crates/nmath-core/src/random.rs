//! NumPy-compatible `SeedSequence` and `PCG64`.

mod distributions;
mod operation;
use std::sync::{Arc, Mutex};

pub use operation::RandomOperation;

use crate::sync::{FifoLock, Ticket};

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

/// NumPy `SeedSequence` for a non-negative integer seed without spawn keys.
#[derive(Clone, Debug)]
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

impl SeedSequence {
    pub fn new(seed: u128) -> Self {
        // `_coerce_to_uint32_array`: little-endian 32-bit words, at least one
        // word.
        let mut entropy = Vec::new();
        let mut rest = seed;
        loop {
            entropy.push(rest as u32);
            rest >>= 32;
            if rest == 0 {
                break;
            }
        }
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for src in 0..POOL_SIZE {
            for dst in 0..POOL_SIZE {
                if src != dst {
                    let h = hashmix(pool[src], &mut hash_const);
                    pool[dst] = mix(pool[dst], h);
                }
            }
        }
        for &word in entropy.iter().skip(POOL_SIZE) {
            for slot in pool.iter_mut() {
                let h = hashmix(word, &mut hash_const);
                *slot = mix(*slot, h);
            }
        }
        Self { pool }
    }

    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut v = self.pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^ (v >> XSHIFT)
            })
            .collect()
    }

    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        self.generate_state_u32(n_words * 2)
            .chunks(2)
            .map(|w| (w[0] as u64) | ((w[1] as u64) << 32))
            .collect()
    }
}

const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

/// NumPy `PCG64` (XSL-RR 128/64).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pcg64 {
    pub state: u128,
    pub inc: u128,
    pub has_uint32: bool,
    pub uinteger: u32,
}

impl Pcg64 {
    pub fn from_seed(seed: u128) -> Self {
        let words = SeedSequence::new(seed).generate_state_u64(4);
        let init_state = ((words[0] as u128) << 64) | words[1] as u128;
        let init_seq = ((words[2] as u128) << 64) | words[3] as u128;
        let mut rng = Pcg64 {
            state: 0,
            has_uint32: false,
            uinteger: 0,
            inc: (init_seq << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(init_state);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let value = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (value >> 32) as u32;
        value as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16777216.0)
    }

    /// `Generator.random()` for float64.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// A shared generator. Jobs that use it serialize in request order, read its
/// state when they start and commit the advanced state only on success.
#[derive(Clone, Debug)]
pub struct Generator {
    state: Arc<Mutex<Pcg64>>,
    order: FifoLock,
}

impl Generator {
    pub fn new(seed: u128) -> Self {
        Self {
            state: Arc::new(Mutex::new(Pcg64::from_seed(seed))),
            order: FifoLock::new(),
        }
    }

    pub fn ticket(&self) -> Ticket {
        self.order.ticket()
    }

    pub fn ptr_eq(&self, other: &Generator) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    pub fn current(&self) -> Pcg64 {
        *self.state.lock().unwrap()
    }

    pub fn commit(&self, state: Pcg64) {
        *self.state.lock().unwrap() = state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors recorded from NumPy 2.5.4.
    #[test]
    fn seed_sequence_matches_numpy() {
        assert_eq!(
            SeedSequence::new(12345).generate_state_u32(4),
            vec![2688385916, 3048105090, 4196366895, 3152189807]
        );
    }

    #[test]
    fn pcg64_matches_numpy() {
        let mut rng = Pcg64::from_seed(12345);
        assert_eq!(rng.state, 33261208707367790463622745601869196757);
        assert_eq!(rng.inc, 268209174141567072605526753992732310247);
        let raw: Vec<u64> = (0..3).map(|_| rng.next_u64()).collect();
        assert_eq!(
            raw,
            vec![
                4193609425186963869,
                5843160025838961886,
                14708796524633321433
            ]
        );
    }

    #[test]
    fn generator_random_matches_numpy() {
        let mut rng = Pcg64::from_seed(42);
        let values: Vec<f64> = (0..3).map(|_| rng.next_f64()).collect();
        assert_eq!(
            values,
            vec![0.7739560485559633, 0.4388784397520523, 0.8585979199113825]
        );
    }
}

#[cfg(test)]
#[path = "random/execution_tests.rs"]
mod execution_tests;
