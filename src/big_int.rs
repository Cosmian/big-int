use crate::Errors;
use eyre::Result;
use gmp::{
    mpz::{Mpz, ProbabPrimeResult},
    rand::RandState,
};
use num::traits::{One, Zero};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Deref, Mul, MulAssign, Neg},
};

#[derive(Debug, Clone)]
pub struct RNSRepresentation {
    data: Vec<u64>,
    modulus: Vec<u64>,
}

impl RNSRepresentation {
    pub fn new(data: Vec<u64>, modulus: &[u64]) -> Result<RNSRepresentation> {
        eyre::ensure!(data.len() == modulus.len(), "Length do not match!");
        Ok(Self {
            data,
            modulus: modulus.to_vec(),
        })
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

    /// Compute the addition of two `BigInt`.
    fn add(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }

    /// Compute the addition of two `BigInt`.
    fn add_assign(&mut self, other: &Self) {
        self.0 += &other.0;
    }

    /// Compute the addition of a `BigInt` with a `u64`.
    fn add_ui(&self, other: &u64) -> Self {
        Self(&self.0 + *other)
    }

    /// Compute the addition of two `BigInt`.
    fn add_assign_ui(&mut self, other: &u64) {
        self.0 += *other;
    }

    /// Compute the substraction of two `BigInt`.
    fn sub(&self, other: &Self) -> Self {
        Self(&self.0 - &other.0)
    }

    /// Compute the substraction of two `BigInt`.
    fn sub_assign(&mut self, other: &Self) {
        self.0 -= &other.0;
    }

    /// Compute the substraction of two `BigInt`.
    fn sub_ui(&self, other: &u64) -> Self {
        Self(&self.0 - *other)
    }

    /// Compute the substraction of two `BigInt`.
    fn sub_assign_ui(&mut self, other: &u64) {
        self.0 -= *other;
    }

    /// Compute the multiplication of two `BigInt`.
    fn mul(&self, other: &Self) -> Self {
        Self(&self.0 * &other.0)
    }

    /// Compute the multiplication of two `BigInt`.
    fn mul_assign(&mut self, other: &Self) {
        self.0 *= &other.0;
    }

    /// Compute the multiplication of two `BigInt`.
    fn mul_ui(&self, other: &u64) -> Self {
        Self(&self.0 * *other)
    }

    /// Compute the multiplication of two `BigInt`.
    fn mul_assign_ui(&mut self, other: &u64) {
        self.0 *= *other;
    }

    /// Compute the remainder of a `BigInt` modulo a `&BigInt`,
    /// keeping it positive.
    fn rem(&self, modulo: &Self) -> Self {
        Self(self.0.modulus(&modulo.0))
    }

    /// Compute the remainder of a `BigInt` modulo a `&BigInt`.
    fn rem_assign(&mut self, modulo: &Self) {
        self.0 = self.0.modulus(&modulo.0)
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
    /// `None` if it is not possible.
    pub fn div_exact(&self, other: &Self) -> Result<Option<Self>> {
        eyre::ensure!(!other.is_zero(), "Cannot exactly divide by zero!");
        if self.is_divisible_by(other) {
            Ok(Some(Self(self.0.div_floor(&other.0))))
        } else {
            Ok(None)
        }
    }

    /// Divide `self` by the given `BigInt`, round to the closest `BigInt`.
    pub fn div_round_closest(&self, other: &Self) -> Result<Self> {
        let floor = self.div_floor(other)?;
        let ceil = &floor + Self::from(1);
        if (self - &floor * other) < (&ceil * other - self) {
            Ok(floor)
        } else {
            Ok(ceil)
        }
    }

    /// Divide `self` by the given `BigInt`, round toward infinity.
    pub fn div_ceil(&self, other: &Self) -> Result<Self> {
        eyre::ensure!(!other.is_zero(), "Cannot ceil divide by zero!");
        Ok(self.div_floor(other)? + Self::from(1))
    }

    /// Divide `self` by the given `BigInt`, round toward minus infinity.
    pub fn div_floor(&self, other: &Self) -> Result<Self> {
        eyre::ensure!(!other.is_zero(), "Cannot floor divide by zero!");
        Ok(Self(self.0.div_floor(&other.0)))
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
    pub fn powm(&self, exp: &Self, modulus: &Self) -> Result<Self> {
        eyre::ensure!(
            !modulus.is_divisible_by(&BigInt::from(2)),
            "The modulus used cannot be an even number !"
        );
        Ok(Self(self.0.powm_sec(&exp.0, &modulus.0)))
    }

    /// Check if `self` is invertible modulo the given `BigInt`.
    pub fn is_invertible(&self, modulus: &Self) -> bool {
        self.gcd(modulus) == 1
    }

    /// Invert `self` modulo the given `BigInt` if this is possible. Return
    /// `None` if it is impossible.
    pub fn invmod(&self, modulus: &Self) -> Option<Self> {
        self.0.invert(&modulus.0).map(Self)
    }

    /// Reduce `self` modulo the given `BigInt`.
    pub fn reduce(&self, modulus: &Self) -> Self {
        Self(self.0.modulus(&modulus.0))
    }

    /// Reduce `self` modulo the given `u64`.
    pub fn reduce_ui(&self, modulus: u64) -> u64 {
        u64::try_from(&self.reduce(&BigInt::from(modulus)))
            .expect("Reducing a `BigInt` by a `u64` should return a 64-bit sized `BigInt`")
    }

    /// Convert the given BigInt `m` into its RNS reprensentation using the
    /// following formula:
    ///
    /// m_i = m (mod p_i)
    ///
    /// where `m_i` is the RNS composant given the prime `p_i`
    ///
    /// - `modulus` : list of primes `p_i`
    pub fn to_rns(&self, modulus: &[u64]) -> RNSRepresentation {
        // TODO: ensure self is a positive number
        assert!(
            self >= &BigInt::zero(),
            "negative BigInt to RNSRepresentation convversion is not yet supported!"
        );
        RNSRepresentation {
            data: modulus.iter().map(|&p| self.reduce_ui(p)).collect(),
            modulus: modulus.to_vec(),
        }
    }

    /// Hash to an invertible BigInt modulo the given modulus.
    /// - `input`     : input to the hash function;
    /// - `modulus`   : we require the result of the hash function to be invertible modulo this `BigInt`.
    pub fn hash_to_invertible<T>(input: &T, modulus: &BigInt) -> BigInt
    where
        T: Hash,
    {
        // initializing at 0 as it is not invertible; the BigInt default value is 0.
        let mut result = BigInt::default();
        let mut hasher = DefaultHasher::new();
        while !result.is_invertible(modulus) {
            input.hash(&mut hasher);
            result = BigInt::from(hasher.finish()) % modulus;
        }
        result
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
    type Error = Errors;
    fn try_from(n: &'a BigInt) -> Result<Self, Self::Error> {
        Option::<u64>::from(&n.0).ok_or(Self::Error::ConversionError)
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
        BigInt(-self.0.clone())
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
    Rem { rem },
    RemAssign { rem_assign },
    rem,
    rem_assign
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
