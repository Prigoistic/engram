//! A tiny deterministic pseudo-random generator.
//!
//! HNSW assigns each node a random level drawn from an exponential
//! distribution. A fixed seed makes index construction reproducible, which is
//! what lets the recall tests assert a stable number rather than a flaky range.
//! This is `xorshift64*`; it is not cryptographic and is not meant to be.

/// A `xorshift64*` generator.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a generator from `seed`. A zero seed is remapped so the state is
    /// never zero (which `xorshift` cannot escape).
    pub fn new(seed: u64) -> Self {
        let state = seed ^ 0x9E37_79B9_7F4A_7C15;
        Self {
            state: if state == 0 {
                0x2545_F491_4F6C_DD1D
            } else {
                state
            },
        }
    }

    /// The next 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform `f64` in the open interval (0, 1). Never returns 0, so it is
    /// safe to pass to `ln()`.
    pub fn unit(&mut self) -> f64 {
        // Top 53 bits give a uniform integer in [0, 2^53); shift to (0, 1].
        let v = self.next_u64() >> 11;
        (v as f64 + 1.0) / (((1u64 << 53) as f64) + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_is_strictly_between_zero_and_one() {
        let mut rng = Rng::new(42);
        for _ in 0..100_000 {
            let u = rng.unit();
            assert!(u > 0.0 && u < 1.0, "{u}");
        }
    }

    #[test]
    fn is_deterministic_for_a_seed() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            assert_eq!(a.unit(), b.unit());
        }
    }

    #[test]
    fn mean_is_near_one_half() {
        let mut rng = Rng::new(123);
        let n = 200_000;
        let mean: f64 = (0..n).map(|_| rng.unit()).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean={mean}");
    }
}
