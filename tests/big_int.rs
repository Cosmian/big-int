#![cfg(test)]

use eyre::Result;
use big_int::BigInt;

#[test]
fn test_pow() -> Result<()> {
    let a = BigInt::from(2u64);
    eyre::ensure!(a.pow(10) == 1024u64, "Wrong exponentiation result for pow!");
    Ok(())
}

#[test]
fn test_powm() -> Result<()> {
    let a = BigInt::from(2u64);
    let b = BigInt::from(8u64);
    let modulus = BigInt::from(999u64);
    eyre::ensure!(
        a.powm(&b, &modulus)? == 256u64,
        "Wrong exponentiation result for pow!"
    );
    Ok(())
}

#[test]
fn test_div_exact() -> Result<()> {
    let a = BigInt::from(124u64);
    let b = BigInt::from(2u64);
    eyre::ensure!(a.is_divisible_by(&b), "Cannot divide {:?} by {:?}!", a, b);
    // Can unwrap safely since `a.is_divisible_by(&b) == true`
    eyre::ensure!(
        a.div_exact(&b)?.unwrap() == 62u64,
        "Wrong exact division result!"
    );
    Ok(())
}

#[test]
fn test_add() -> Result<()> {
    let a = BigInt::from(3u64);
    let b = BigInt::from(123u64);
    eyre::ensure!(b + a == 126u64, "Wrong addition result!");
    Ok(())
}

#[test]
fn test_and() -> Result<()> {
    let a = BigInt::from(3u64);
    let b = 1u64;
    eyre::ensure!(a & b == 1u64, "Wrong and bitwise operation result!");
    Ok(())
}

#[test]
fn test_hash_to_invertible() -> Result<()> {
    let mut state = BigInt::rand_init(0u64);
    let modulus = BigInt::from(30);
    let n = BigInt::rand_range(&modulus, &mut state);
    let hashee = BigInt::hash_to_invertible(&n, &modulus);
    eyre::ensure!(
        (hashee.is_invertible(&modulus) && (hashee < modulus)),
        "The function for hashing to an invertible BigInt is wrong."
    );
    Ok(())
}

#[test]
fn test_rns() -> Result<()> {
    let modulus = [3, 5];
    let q: BigInt = modulus.iter().map(|&e| BigInt::from(e)).product();
    let n = BigInt::from(40);
    let rns = n.to_rns(&modulus);
    let res = BigInt::from(&rns) % &q;
    eyre::ensure!(n % &q == res, "Invalid RNS conversion!");
    Ok(())
}
