//! Distance metrics and the distance kernels behind them.
//!
//! Every metric is expressed as a *distance*, where a smaller value means the
//! two vectors are closer. This keeps the search and index code uniform: it
//! always wants the smallest distances, whatever the metric.
//!
//! The distance computation runs through the SIMD kernels in
//! [`super::kernels`]; this module only chooses how the kernel outputs are
//! combined into a metric and handles normalisation.

use super::kernels;

/// How the distance between two vectors is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Cosine distance, `1 - cos(a, b)`. Vectors are L2-normalised on insert,
    /// so at query time this reduces to a single dot product.
    Cosine,

    /// Euclidean (L2) distance.
    L2,

    /// Negative inner product, so that a larger dot product is a smaller
    /// distance. Useful for maximum-inner-product search.
    Dot,
}

impl Metric {
    /// Parses a metric name, case-insensitively.
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "cosine" => Some(Self::Cosine),
            "l2" | "euclidean" => Some(Self::L2),
            "dot" | "ip" => Some(Self::Dot),
            _ => None,
        }
    }

    /// A stable byte tag for on-disk encoding.
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Cosine => 0,
            Self::L2 => 1,
            Self::Dot => 2,
        }
    }

    /// The inverse of [`Metric::to_u8`].
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Cosine),
            1 => Some(Self::L2),
            2 => Some(Self::Dot),
            _ => None,
        }
    }

    /// The canonical lowercase name of the metric.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::L2 => "l2",
            Self::Dot => "dot",
        }
    }

    /// Whether vectors should be L2-normalised when stored under this metric.
    pub fn normalizes(self) -> bool {
        matches!(self, Self::Cosine)
    }

    /// The distance between `a` and `b` under this metric; smaller is closer.
    ///
    /// For [`Metric::Cosine`] both vectors are assumed already normalised.
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Self::Cosine => 1.0 - kernels::dot(a, b),
            Self::L2 => kernels::l2_sq(a, b).sqrt(),
            Self::Dot => -kernels::dot(a, b),
        }
    }
}

/// The L2 norm (length) of a vector.
pub fn norm(v: &[f32]) -> f32 {
    kernels::dot(v, v).sqrt()
}

/// L2-normalises a vector in place. A zero vector is left unchanged.
pub fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    #[test]
    fn norm_and_normalize() {
        approx(norm(&[3.0, 4.0]), 5.0);
        let mut v = [3.0, 4.0];
        normalize(&mut v);
        approx(norm(&v), 1.0);
        approx(v[0], 0.6);
        approx(v[1], 0.8);
    }

    #[test]
    fn zero_vector_normalizes_to_itself() {
        let mut v = [0.0, 0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_distance_is_euclidean() {
        approx(Metric::L2.distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
    }

    #[test]
    fn cosine_distance_of_identical_normalized_is_zero() {
        let mut a = [1.0, 2.0, 2.0];
        normalize(&mut a);
        approx(Metric::Cosine.distance(&a, &a), 0.0);
    }

    #[test]
    fn cosine_distance_of_orthogonal_is_one() {
        approx(Metric::Cosine.distance(&[1.0, 0.0], &[0.0, 1.0]), 1.0);
    }

    #[test]
    fn dot_distance_orders_by_negative_inner_product() {
        // Larger dot product => smaller (more negative) distance.
        let q = [1.0, 1.0];
        let near = Metric::Dot.distance(&q, &[2.0, 2.0]); // dot 4
        let far = Metric::Dot.distance(&q, &[0.0, 1.0]); // dot 1
        assert!(near < far);
    }

    #[test]
    fn metric_parse_roundtrip() {
        for m in [Metric::Cosine, Metric::L2, Metric::Dot] {
            assert_eq!(Metric::parse(m.as_str()), Some(m));
        }
        assert_eq!(Metric::parse("COSINE"), Some(Metric::Cosine));
        assert_eq!(Metric::parse("euclidean"), Some(Metric::L2));
        assert_eq!(Metric::parse("ip"), Some(Metric::Dot));
        assert_eq!(Metric::parse("nope"), None);
    }
}
