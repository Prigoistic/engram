//! SIMD distance kernels with per-target dispatch.
//!
//! [`dot`] and [`l2_sq`] are the engine's innermost hot loop. Each picks the
//! widest implementation available for the target:
//!
//! * **aarch64** — NEON, which is part of the AArch64 baseline and so is always
//!   present; no runtime detection is needed.
//! * **x86_64** — AVX2 + FMA when the running CPU reports them, else scalar.
//! * **everything else** — scalar.
//!
//! The scalar versions ([`dot_scalar`], [`l2_sq_scalar`]) double as the
//! reference the SIMD paths are checked against in the tests. SIMD reorders the
//! summation and uses fused multiply-add, so results match the scalar reference
//! to a tight tolerance rather than bit-for-bit.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

/// The dot product of two equal-length vectors.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: NEON is part of the AArch64 baseline, so always available here.
    unsafe { dot_neon(a, b) }
}

/// The dot product of two equal-length vectors.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: guarded by the runtime feature detection above.
        unsafe { dot_avx2(a, b) }
    } else {
        dot_scalar(a, b)
    }
}

/// The dot product of two equal-length vectors.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    dot_scalar(a, b)
}

/// The squared Euclidean distance between two equal-length vectors.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: NEON is part of the AArch64 baseline, so always available here.
    unsafe { l2_sq_neon(a, b) }
}

/// The squared Euclidean distance between two equal-length vectors.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: guarded by the runtime feature detection above.
        unsafe { l2_sq_avx2(a, b) }
    } else {
        l2_sq_scalar(a, b)
    }
}

/// The squared Euclidean distance between two equal-length vectors.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    l2_sq_scalar(a, b)
}

// ---------------------------------------------------------------------------
// Scalar reference / fallback
// ---------------------------------------------------------------------------

// On aarch64 these are reached only by the tests, since `dot`/`l2_sq` always
// take the NEON path; on x86_64 and other targets they are the fallback.
#[allow(dead_code)]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[allow(dead_code)]
pub fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ---------------------------------------------------------------------------
// NEON (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0;
        // Four 128-bit accumulators (16 lanes) per iteration to hide FMA latency.
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }

        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut sum = vaddvq_f32(acc);
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0;
        while i + 16 <= n {
            let d0 = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            let d2 = vsubq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            let d3 = vsubq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
            i += 16;
        }
        while i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc0 = vfmaq_f32(acc0, d, d);
            i += 4;
        }

        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut sum = vaddvq_f32(acc);
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            sum += d * d;
            i += 1;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// AVX2 + FMA (x86_64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: __m256) -> f32 {
    unsafe {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi); // 4 lanes
        let s = _mm_hadd_ps(s, s); // 2 lanes
        let s = _mm_hadd_ps(s, s); // 1 lane
        _mm_cvtss_f32(s)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut i = 0;
        // Two 256-bit accumulators (16 lanes) per iteration.
        while i + 16 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            i += 16;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }

        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut i = 0;
        while i + 16 <= n {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
            );
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            i += 16;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc0 = _mm256_fmadd_ps(d, d, acc0);
            i += 8;
        }

        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            sum += d * d;
            i += 1;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic LCG yielding `f32` in roughly [-1, 1). Avoids a
    /// `rand` dependency and keeps the equivalence test reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (self.0 >> 33) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    const LENGTHS: [usize; 17] = [
        1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100, 128, 129, 384, 1536,
    ];

    fn close(simd: f32, scalar: f32) {
        // SIMD reorders sums and uses FMA, so compare with a tolerance scaled
        // to the magnitude of the result.
        let tol = 1e-4 * (1.0 + scalar.abs());
        assert!(
            (simd - scalar).abs() <= tol,
            "simd={simd} scalar={scalar} diff={}",
            (simd - scalar).abs()
        );
    }

    #[test]
    fn dot_matches_scalar_across_lengths() {
        let mut rng = Lcg(0x1234_5678);
        // Lengths spanning every remainder path: full blocks, partial, scalar tail.
        for n in LENGTHS {
            let a = rng.vec(n);
            let b = rng.vec(n);
            close(dot(&a, &b), dot_scalar(&a, &b));
        }
    }

    #[test]
    fn l2_sq_matches_scalar_across_lengths() {
        let mut rng = Lcg(0xdead_beef);
        for n in LENGTHS {
            let a = rng.vec(n);
            let b = rng.vec(n);
            close(l2_sq(&a, &b), l2_sq_scalar(&a, &b));
        }
    }

    #[test]
    fn known_values() {
        close(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        close(l2_sq(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        // Lengths exercising the wide path explicitly.
        let a: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..20).map(|i| (i * 2) as f32).collect();
        close(dot(&a, &b), dot_scalar(&a, &b));
        close(l2_sq(&a, &b), l2_sq_scalar(&a, &b));
    }

    /// Throughput comparison, ignored by default. Run with:
    /// `cargo test --release -p engram kernels -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_throughput() {
        use std::time::Instant;
        let dim = 768;
        let n = 100_000;
        let mut rng = Lcg(0x00ab_cdef);
        let q = rng.vec(dim);
        let data: Vec<Vec<f32>> = (0..n).map(|_| rng.vec(dim)).collect();

        let t = Instant::now();
        let mut s1 = 0.0f32;
        for v in &data {
            s1 += dot_scalar(&q, v);
        }
        let scalar = t.elapsed();

        let t = Instant::now();
        let mut s2 = 0.0f32;
        for v in &data {
            s2 += dot(&q, v);
        }
        let simd = t.elapsed();

        let flops = (2.0 * dim as f64 * n as f64) / simd.as_secs_f64() / 1e9;
        println!(
            "dot dim={dim} n={n}: scalar={scalar:?} simd={simd:?} speedup={:.2}x simd_gflops={flops:.2} (checksum {s1:.1}/{s2:.1})",
            scalar.as_secs_f64() / simd.as_secs_f64()
        );
    }
}
