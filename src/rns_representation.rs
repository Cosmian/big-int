use crate::{BigInt, BigIntError};
use std::ops::{Deref, MulAssign};

/// RNS representation of a unsigned integer modulo a list of primes.
///
/// For all `n`, given the primes `[p_1,..., p_k]`, one can compute the RNS
/// reprensetation of `n` as follows:
///
/// for all `i` in `[[1, k]]`, `n_i = n (mod p_i)`
///
/// For a given RNS representation `(n_1,..., n_k)` for the given primes
/// `[p_1,..., p_k]`, one can compute `n` as follows:
///
/// `n = Sum(n_i * P_i * Q_i)`
///
/// where `n_i` is the RNS composant given the prime `p_i`, `P_i` the
/// product of all other primes and `Q_i` the inverse of this product
/// modulo `p_i`.
#[derive(Debug, Clone)]
pub struct RNSRepresentation {
    pub(crate) data: Vec<u64>,
    pub(crate) modulus: Vec<u64>,
}

impl RNSRepresentation {
    /// Create a new RNS representation from raw data. The slices given as
    /// data and modulus should have the same length.
    ///
    /// - `data`    : data to represent in its RNS form
    /// - `modulus` : list of prime moduli used to compute the RNS form
    pub fn new(data: Vec<u64>, modulus: &[u64]) -> Result<RNSRepresentation, BigIntError> {
        if data.len() == modulus.len() {
            Ok(Self {
                data,
                modulus: modulus.to_vec(),
            })
        } else {
            Err(BigIntError::InvalidSize {
                expected: modulus.len(),
                found: data.len(),
            })
        }
    }

    fn add(&self, other: &Self) -> Self {
        let mut res = self.clone();
        res.add_assign(other);
        res
    }

    fn add_assign(&mut self, other: &Self) {
        assert_eq!(
            self.modulus.len(),
            other.modulus.len(),
            "Cannot add objects with different modulus!"
        );

        for (((a, &b), &p1), &p2) in self
            .data
            .iter_mut()
            .zip(other.data.iter())
            .zip(&self.modulus)
            .zip(&other.modulus)
        {
            assert_eq!(p1, p2, "Cannot add objects with different modulus!");
            *a = (*a + b) % p1;
        }
    }

    fn sub(&self, other: &Self) -> Self {
        let mut res = self.clone();
        res.sub_assign(other);
        res
    }

    fn sub_assign(&mut self, other: &Self) {
        assert_eq!(
            self.modulus.len(),
            other.modulus.len(),
            "Cannot add objects with different coefficient modulus!"
        );

        for (((a, &b), &p1), &p2) in self
            .data
            .iter_mut()
            .zip(other.data.iter())
            .zip(&self.modulus)
            .zip(&other.modulus)
        {
            assert_eq!(p1, p2, "Cannot add objects with different modulus!");
            if *a < b {
                *a = ((p1 + *a) - b) % p1;
            } else {
                *a = (*a - b) % p1;
            }
        }
    }

    fn mul_scalar<'a, T>(&self, scalar: &'a T) -> Self
    where
        u64: MulAssign<&'a T>,
    {
        let mut res = self.clone();
        res.mul_assign_scalar(scalar);
        res
    }

    fn mul_assign_scalar<'a, T>(&mut self, scalar: &'a T)
    where
        u64: MulAssign<&'a T>,
    {
        for (e, p) in self.data.iter_mut().zip(&self.modulus) {
            *e *= scalar;
            *e %= p;
        }
    }

    fn mul_big_int(&self, scalar: &BigInt) -> Self {
        let mut res = self.clone();
        res.mul_assign_big_int(scalar);
        res
    }

    fn mul_assign_big_int(&mut self, scalar: &BigInt) {
        let scalar = scalar.to_rns(&self.modulus);
        for ((e, s), p) in self
            .data
            .iter_mut()
            .zip(scalar.data.iter())
            .zip(&self.modulus)
        {
            *e *= s;
            *e %= p;
        }
    }
}

impl Deref for RNSRepresentation {
    type Target = Vec<u64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

crate::impl_ops_trait!(
    RNSRepresentation,
    RNSRepresentation,
    Add { add },
    AddAssign { add_assign },
    add,
    add_assign
);

crate::impl_ops_trait!(
    RNSRepresentation,
    RNSRepresentation,
    Sub { sub },
    SubAssign { sub_assign },
    sub,
    sub_assign
);

crate::impl_ops_trait!(
    RNSRepresentation,
    u64,
    Mul { mul },
    MulAssign { mul_assign },
    mul_scalar,
    mul_assign_scalar
);

crate::impl_ops_trait!(
    RNSRepresentation,
    BigInt,
    Mul { mul },
    MulAssign { mul_assign },
    mul_big_int,
    mul_assign_big_int
);
