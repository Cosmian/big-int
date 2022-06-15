use big_int::BigInt;

#[test]
fn test_pow() {
    let a = BigInt::from(2u64);
    assert_eq!(a.pow(10), 1024u64, "Wrong exponentiation result for pow!");
}

#[test]
fn test_powm() {
    let a = BigInt::from(2u64);
    let b = BigInt::from(8u64);
    let modulus = BigInt::from(999u64);
    assert_eq!(
        a.powm(&b, &modulus).unwrap(),
        256u64,
        "Wrong exponentiation result for pow!"
    );
}

#[test]
fn test_division_by_0() {
    let (a, b) = (BigInt::from(2), BigInt::from(0));
    assert_eq!(
        big_int::BigIntError::DivisionByZero,
        a.div_exact(&b).unwrap_err(),
        "Wrong error returned"
    );
}

#[test]
fn test_div_exact() {
    let a = BigInt::from(124u64);
    let b = BigInt::from(2u64);
    assert!(a.is_divisible_by(&b), "Cannot divide {a:?} by {b:?}!");
    // Can unwrap safely since `a.is_divisible_by(&b) == true`
    assert_eq!(
        a.div_exact(&b).unwrap(),
        62u64,
        "Wrong exact division result!"
    );
}

#[test]
fn test_add() {
    let a = BigInt::from(3u64);
    let b = BigInt::from(123u64);
    assert_eq!(b + a, 126u64, "Wrong addition result!");
}

#[test]
fn test_and() {
    let a = BigInt::from(3u64);
    let b = 1u64;
    assert_eq!(a & b, 1u64, "Wrong and bitwise operation result!");
}

#[test]
fn test_hash_to_invertible() {
    let mut state = BigInt::rand_init(0u64);
    let modulus = BigInt::from(30);
    let n = BigInt::rand_range(&modulus, &mut state);
    let hashee = BigInt::hash_to_invertible(&n, &modulus);
    assert!(
        (hashee.is_invertible(&modulus) && (hashee < modulus)),
        "The function for hashing to an invertible BigInt is wrong."
    );
}

#[test]
fn test_rns() {
    let modulus = [3, 5];
    let q: BigInt = modulus.iter().map(|&e| BigInt::from(e)).product();
    let n = BigInt::from(40);
    let rns = n.to_rns(&modulus);
    let res = BigInt::from(&rns) % &q;
    assert_eq!(n % &q, res, "Invalid RNS conversion!");
}
