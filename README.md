# BigInt

## Description

This crate is an implementation of a big integer type. It is based on GMP and
thus offers both the speed and precision this library allows. Its aim is to
provide an easy to use `BigInt` type that can be used in any crypto crate.

Since efficiency is often critical in cryptographic computations, this crate
also provides an RNS representation of integers ([see
below](#RNS)).

## GMP

The [GNU Multi Precision](https://gmplib.org/) arithmetic library is a portable library written in C.
It provides representations and arithmetic operations over integers, rational
number and floating points [3]. This crate focuses on the `mpz` integers, even
though it could easily be extended.

In order to compile this crate, one has to install GMP. This can be done by
installing the dev package provided by the package manager on any recent Linux
system, or by installing it from source.

## RNS

Given coprimes integers `[p_1, ..., p_k]` and `P = Prod(p_i)`, one can convert an integer
`n (mod P)` as the composants of its RNS representation `[n_1, ..., n_k]` as
follows:

	for all i in [[1;k]], n_i = n (mod p_i)

then

	n = Sum(n_i * P_i * Q_i)

with `P_i = P / p_i` and `Q = P_i^{-1} (mod p_i)` [1], [2].

Provided that all primes `p_i` can be encoded into a machine word, this
method provides an efficient representation of big integers and
efficient arithmetic operations (big integer operations become `k` machine word
integer operations).


## Bibliography

[1] https://en.wikipedia.org/wiki/Chinese_remainder_theorem

[2] https://en.wikipedia.org/wiki/Residue_numeral_system

[3] https://gmplib.org/gmp-man-6.2.1.pdf
