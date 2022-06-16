use crate::{BigIntError, RNSRepresentation};
use gmp::{
    mpz::{Mpz, ProbabPrimeResult},
    rand::RandState,
};
use num::traits::{One, Zero};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Mul, Neg},
};

/// Implementation based on GMP of a big integer.
#[derive(Debug, Clone, PartialEq, PartialOrd, Hash)]
pub struct BigInt(pub(crate) Mpz);

impl BigInt {
    pub fn new() -> Self {
        Self(Mpz::new())
    }

    /*
     * Random generation
     */

    /// Initialise a random state later used to generate random `mpz`.
    ///
    /// - `seed`: integer used as a seed for the random generator
    pub fn rand_init(seed: u64) -> RandState {
        let mut state = RandState::new();
        state.seed_ui(seed);
        state
    }

    /// Get a random integer in `[[0 ; 2^n_bits - 1]]`.
    /// TODO: benchmark both random generation methods
    ///
    /// - `n_bits`  : upper bound on the number of bits of the generated number
    /// - `state`   : random state
    pub fn rand_range_2exp(modulus: u64, state: &mut RandState) -> Self {
        Self(state.urandom_2exp(modulus))
    }

    /// Get a random integer in `Zq`.
    ///
    /// - `q`       : upper bound on the generated number
    /// - `state`   : random state
    pub fn rand_range(q: &Self, state: &mut RandState) -> Self {
        Self(state.urandom(&q.0))
    }

    /// Get a random integer in `Zq \ {0}`.
    ///
    /// - `q`       : upper bound on the generated number
    /// - `state`   : random state
    pub fn rand_range_ex_0(q: &Self, state: &mut RandState) -> Self {
        let mut res;
        loop {
            res = Self::rand_range(q, state);
            if !res.is_zero() {
                return res;
            }
        }
    }

    /// Get an invertible random in `Z_q`.
    ///
    /// - `q`       : upper bound on the generated number
    /// - `state`   : random state
    pub fn invertible_rand(q: &Self, state: &mut RandState) -> Self {
        let mut res;
        loop {
            res = Self::rand_range(q, state);
            if res.is_invertible(q) {
                return res;
            }
        }
    }

    /// Get an invertible random in a given range of `Z_m`.
    ///
    /// - `modulus` : result is invertible modulo the modulus.
    /// - `range    : upper bound on the generated number
    /// - `state`   : random state
    pub fn invertible_rand_in_range(modulus: &Self, range: &Self, state: &mut RandState) -> Self {
        let mut res;
        loop {
            res = Self::rand_range(range, state);
            if res.is_invertible(modulus) {
                return res;
            }
        }
    }

    /// Generate n-bits random elements and use the Miller-Rabin primality test
    /// until a prime is found.
    ///
    /// - `n_bits`  : number of bits to generate
    /// - `n_reps`  : number of repetitions for the Miller-Rabin algorithm
    /// - `state`   : random state
    pub fn generate_prime(n_bits: u32, n_reps: i32, state: &mut RandState) -> Self {
        // mask to ensure both the first and last bits are set
        let b = BigInt::from(2).pow(n_bits - 1) + 1;
        let mut res;
        loop {
            res = BigInt::rand_range_2exp(n_bits as u64, state) | &b;
            if ProbabPrimeResult::NotPrime != res.0.probab_prime(n_reps) {
                return res;
            }
        }
    }

    /*
     * Arithmetic
     */

    /// Greatest Common Divisor, always positive.
    pub fn gcd(&self, q: &Self) -> Self {
        Self(self.0.gcd(&q.0))
    }

    /// Least Common Multiple
    pub fn lcm(&self, q: &Self) -> Self {
        Self(self.0.lcm(&q.0))
    }

    fn add(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }

    fn add_assign(&mut self, other: &Self) {
        self.0 += &other.0;
    }

    fn add_ui(&self, other: &u64) -> Self {
        Self(&self.0 + *other)
    }

    fn add_assign_ui(&mut self, other: &u64) {
        self.0 += *other;
    }

    fn sub(&self, other: &Self) -> Self {
        Self(&self.0 - &other.0)
    }

    fn sub_assign(&mut self, other: &Self) {
        self.0 -= &other.0;
    }

    fn sub_ui(&self, other: &u64) -> Self {
        Self(&self.0 - *other)
    }

    fn sub_assign_ui(&mut self, other: &u64) {
        self.0 -= *other;
    }

    fn mul(&self, other: &Self) -> Self {
        Self(&self.0 * &other.0)
    }

    fn mul_assign(&mut self, other: &Self) {
        self.0 *= &other.0;
    }

    fn mul_ui(&self, other: &u64) -> Self {
        Self(&self.0 * *other)
    }

    fn mul_assign_ui(&mut self, other: &u64) {
        self.0 *= *other;
    }

    fn bit_or(&self, other: &Self) -> Self {
        Self(&self.0 | &other.0)
    }

    fn bit_or_assign(&mut self, other: &Self) {
        self.0 |= &other.0;
    }

    fn bit_or_ui(&self, other: &u64) -> Self {
        Self(&self.0 | Mpz::from(*other))
    }

    fn bit_or_assign_ui(&mut self, other: &u64) {
        self.0 |= Mpz::from(*other);
    }

    fn bit_and(&self, other: &Self) -> Self {
        Self(&self.0 & &other.0)
    }

    fn bit_and_assign(&mut self, other: &Self) {
        self.0 &= &other.0;
    }

    fn bit_and_ui(&self, other: &u64) -> Self {
        Self(&self.0 & Mpz::from(*other))
    }

    fn bit_and_assign_ui(&mut self, other: &u64) {
        self.0 &= Mpz::from(*other);
    }

    /// Check if the given `BigInt` divides `self`.
    pub fn is_divisible_by(&self, other: &Self) -> bool {
        other.0.divides(&self.0)
    }

    /// Divide `self` by the given `BigInt` exactly if this is possible. Return
    pub fn div_exact(&self, other: &Self) -> Result<Self, BigIntError> {
        if other.is_zero() {
            Err(BigIntError::DivisionByZero)
        } else if !self.is_divisible_by(other) {
            Err(BigIntError::DivisionError {
                dividand: self.clone(),
                divisor: other.clone(),
            })
        } else {
            Ok(Self(self.0.div_floor(&other.0)))
        }
    }

    /// Divide `self` by the given `BigInt`, round to the closest `BigInt`.
    pub fn div_round_closest(&self, other: &Self) -> Result<Self, BigIntError> {
        let floor = self.div_floor(other)?;
        let ceil = &floor + Self::from(1);
        if (self - &floor * other) < (&ceil * other - self) {
            Ok(floor)
        } else {
            Ok(ceil)
        }
    }

    /// Divide `self` by the given `BigInt`, round toward infinity.
    pub fn div_ceil(&self, other: &Self) -> Result<Self, BigIntError> {
        if other.is_zero() {
            Err(BigIntError::DivisionByZero)
        } else {
            Ok(self.div_floor(other)? + Self::from(1))
        }
    }

    /// Divide `self` by the given `BigInt`, round toward minus infinity.
    pub fn div_floor(&self, other: &Self) -> Result<Self, BigIntError> {
        if other.is_zero() {
            Err(BigIntError::DivisionByZero)
        } else {
            Ok(Self(self.0.div_floor(&other.0)))
        }
    }

    /// Elevate `self` to the power of the given exponent.
    pub fn pow(&self, exp: u32) -> Self {
        Self(self.0.pow(exp))
    }

    /// Elevate `self` to the power of the given exponent in `Z_m`.
    /// The modulus used must be odd (requirement from gmp library).
    /// We use `powm_sec` to get the result in a constant time.
    ///
    /// - `exp`     : exponent to use
    /// - `modulus` : modulus to use to reduce the exponentiation result
    pub fn powm(&self, exp: &Self, modulus: &Self) -> Result<Self, BigIntError> {
        if modulus.is_divisible_by(&BigInt::from(2)) {
            Err(BigIntError::EvenModulus)
        } else if modulus.is_zero() {
            Err(BigIntError::DivisionByZero)
        } else if exp < &BigInt::zero() {
            Err(BigIntError::NegativeExponent)
        } else {
            Ok(Self(self.0.powm_sec(&exp.0, &modulus.0)))
        }
    }

    /// Check if `self` is invertible modulo the given `BigInt`.
    pub fn is_invertible(&self, modulus: &Self) -> bool {
        self.gcd(modulus) == 1
    }

    /// Invert `self` modulo the given `BigInt` if this is possible. Return
    /// `None` if it is impossible.
    pub fn invmod(&self, modulus: &Self) -> Result<Self, BigIntError> {
        if modulus.is_zero() {
            Err(BigIntError::DivisionByZero)
        } else {
            self.0
                .invert(&modulus.0)
                .map(Self)
                .ok_or_else(|| BigIntError::NonInvertible {
                    operand: self.clone(),
                    modulus: modulus.clone(),
                })
        }
    }

    /// Reduce `self` modulo the given `BigInt`.
    pub fn reduce(&self, modulus: &Self) -> Result<Self, BigIntError> {
        if modulus.is_zero() {
            Err(BigIntError::DivisionByZero)
        } else {
            Ok(Self(self.0.modulus(&modulus.0)))
        }
    }

    /// Reduce `self` modulo the given `u64`.
    pub fn reduce_ui(&self, modulus: u64) -> Result<u64, BigIntError> {
        u64::try_from(&self.reduce(&BigInt::from(modulus))?)
    }

    /// Convert the given BigInt `m` into its RNS reprensentation using the
    /// following formula:
    ///
    /// m_i = m (mod p_i)
    ///
    /// where `m_i` is the RNS composant given the prime `p_i`
    ///
    /// # Panic
    ///
    /// This function will panic if it is called on a negative number.
    ///
    /// - `modulus` : list of primes `p_i`
    pub fn to_rns(&self, modulus: &[u64]) -> Result<RNSRepresentation, BigIntError> {
        assert!(
            self >= &BigInt::zero(),
            "negative BigInt to RNSRepresentation conversion is not yet supported!"
        );
        Ok(RNSRepresentation {
            data: modulus
                .iter()
                .map(|p| -> Result<u64, BigIntError> { self.reduce_ui(*p) })
                .collect::<Result<Vec<u64>, BigIntError>>()?,
            modulus: modulus.to_vec(),
        })
    }

    /// Hash to an invertible BigInt modulo the given modulus.
    /// - `input`     : input to the hash function;
    /// - `modulus`   : we require the result of the hash function to be invertible modulo this `BigInt`.
    pub fn hash_to_invertible<T>(input: &T, modulus: &Self) -> Result<Self, BigIntError>
    where
        T: Hash,
    {
        let mut hasher = DefaultHasher::new();
        let mut result = BigInt::zero();
        while !result.is_invertible(modulus) {
            input.hash(&mut hasher);
            result = BigInt::from(hasher.finish()).reduce(modulus)?;
        }
        Ok(result)
    }

    /// Return the size in byte of a BigInt.
    pub fn size(&self) -> usize {
        self.0.size_in_base(8)
    }

    /// Convert the given `BigInt` into the corresponding little endian byte string.
    pub fn to_le_bytes(&self) -> Result<Vec<u8>, BigIntError> {
        // one byte encode 256 different numbers
        const MODULUS: u64 = 256;
        let modulus = BigInt::from(MODULUS);
        // we cannot know easily which size we will need
        // => allocate 128 bytes to avoid reallocations
        const PRE_ALLOCATION_SIZE: usize = 128;
        let mut res = Vec::with_capacity(PRE_ALLOCATION_SIZE);
        let mut n = self.clone();
        while !n.is_zero() {
            let rem = n.reduce_ui(MODULUS)?;
            n -= rem;
            n = n.div_exact(&modulus)?;
            res.push(rem as u8);
        }
        Ok(res)
    }
}

impl From<&RNSRepresentation> for BigInt {
    /// Convert the given RNS reprensentation `n` into a BigInt. This is
    /// done using the following formula:
    ///
    /// n = Sum(n_i * P_i * Q_i)
    ///
    /// where `n_i` is the RNS composant given the prime `p_i`, `P_i` the
    /// product of all other primes and `Q_i` the inverse of this product
    /// modulo `p_i`.
    fn from(n: &RNSRepresentation) -> Self {
        n.data
            .iter()
            .enumerate()
            .map(|(i, &m_i)| {
                let prod: BigInt = n
                    .modulus
                    .iter()
                    .enumerate()
                    .map(|(j, &p_j)| {
                        if i == j {
                            BigInt::one()
                        } else {
                            BigInt::from(p_j)
                        }
                    })
                    .product();
                let inv = prod
                    .invmod(&BigInt::from(n.modulus[i]))
                    .expect("coeff_modulus composants should be coprime!");
                BigInt::from(m_i) * inv * prod
            })
            .sum()
    }
}

impl From<usize> for BigInt {
    fn from(n: usize) -> Self {
        Self(Mpz::from(n as u64))
    }
}

impl From<u64> for BigInt {
    fn from(n: u64) -> Self {
        Self(Mpz::from(n))
    }
}

impl From<i64> for BigInt {
    fn from(s: i64) -> Self {
        Self(Mpz::from(s))
    }
}

impl From<i32> for BigInt {
    fn from(s: i32) -> Self {
        Self(Mpz::from(s))
    }
}

impl<'a> TryFrom<&'a BigInt> for u64 {
    type Error = BigIntError;
    fn try_from(n: &'a BigInt) -> Result<Self, Self::Error> {
        Option::<u64>::from(&n.0).ok_or(Self::Error::ConversionError(String::new()))
    }
}

impl From<Mpz> for BigInt {
    fn from(z: Mpz) -> Self {
        Self(z)
    }
}

impl Neg for &BigInt {
    type Output = BigInt;

    fn neg(self) -> Self::Output {
        BigInt(-&self.0)
    }
}

impl Neg for BigInt {
    type Output = BigInt;

    fn neg(self) -> Self::Output {
        BigInt(-self.0)
    }
}

impl Default for BigInt {
    fn default() -> Self {
        Self(Mpz::zero())
    }
}

impl PartialEq<u64> for BigInt {
    fn eq(&self, other: &u64) -> bool {
        self.0 == Mpz::from(*other)
    }
}

impl<'a> PartialEq<u64> for &'a BigInt {
    fn eq(&self, other: &u64) -> bool {
        self.0 == Mpz::from(*other)
    }
}

impl Sum for BigInt {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |x, acc| x + acc)
    }
}

impl Product for BigInt {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, n| n * acc)
    }
}

impl Zero for BigInt {
    fn zero() -> Self {
        BigInt::from(0)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for BigInt {
    fn one() -> Self {
        BigInt::from(1)
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl<'a> Mul<&'a BigInt> for u64 {
    type Output = BigInt;

    fn mul(self, rhs: &'a BigInt) -> Self::Output {
        rhs.add_ui(&self)
    }
}

impl<'a, T> From<&'a T> for BigInt
where
    T: Copy,
    BigInt: From<T>,
{
    fn from(n: &'a T) -> Self {
        BigInt::from(*n)
    }
}

impl TryFrom<&BigInt> for Vec<u8> {
    type Error = BigIntError;

    fn try_from(value: &BigInt) -> Result<Self, Self::Error> {
        value.to_le_bytes()
    }
}

impl From<&[u8]> for BigInt {
    fn from(v: &[u8]) -> Self {
        v.iter()
            .rev()
            .fold(BigInt::zero(), |acc, &e| acc * 256 + u64::from(e))
    }
}

crate::impl_ops_trait!(
    BigInt,
    BigInt,
    Add { add },
    AddAssign { add_assign },
    add,
    add_assign
);

crate::impl_ops_trait!(
    BigInt,
    u64,
    Add { add },
    AddAssign { add_assign },
    add_ui,
    add_assign_ui
);

crate::impl_ops_trait!(
    BigInt,
    BigInt,
    Sub { sub },
    SubAssign { sub_assign },
    sub,
    sub_assign
);

crate::impl_ops_trait!(
    BigInt,
    u64,
    Sub { sub },
    SubAssign { sub_assign },
    sub_ui,
    sub_assign_ui
);

crate::impl_ops_trait!(
    BigInt,
    BigInt,
    Mul { mul },
    MulAssign { mul_assign },
    mul,
    mul_assign
);

crate::impl_ops_trait!(
    BigInt,
    u64,
    Mul { mul },
    MulAssign { mul_assign },
    mul_ui,
    mul_assign_ui
);

crate::impl_ops_trait!(
    BigInt,
    BigInt,
    BitOr { bitor },
    BitOrAssign { bitor_assign },
    bit_or,
    bit_or_assign
);

crate::impl_ops_trait!(
    BigInt,
    u64,
    BitOr { bitor },
    BitOrAssign { bitor_assign },
    bit_or_ui,
    bit_or_assign_ui
);

crate::impl_ops_trait!(
    BigInt,
    BigInt,
    BitAnd { bitand },
    BitAndAssign { bitand_assign },
    bit_and,
    bit_and_assign
);

crate::impl_ops_trait!(
    BigInt,
    u64,
    BitAnd { bitand },
    BitAndAssign { bitand_assign },
    bit_and_ui,
    bit_and_assign_ui
);
