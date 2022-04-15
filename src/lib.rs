use thiserror::Error;

mod big_int;
mod macros;
mod rns_representation;

pub use crate::big_int::BigInt;
pub use crate::rns_representation::RNSRepresentation;

#[derive(Error, Debug, PartialEq)]
pub enum BigIntError {
    #[error("Error while converting!")]
    ConversionError,
    #[error("Invalid size, expected {expected:?}, found {found:?}")]
    InvalidSize { expected: usize, found: usize },
    #[error("Cannot divide by 0!")]
    DivisionByZero,
    #[error("Cannot divide {dividand:?} by {divisor:?}")]
    DivisionError { dividand: BigInt, divisor: BigInt },
    #[error("Modulus cannot be an even number!")]
    EvenModulus,
    #[error("Inverse of {operand:?} does not exist given modulus {modulus:?}")]
    NonexistantInverse { operand: BigInt, modulus: BigInt },
}
