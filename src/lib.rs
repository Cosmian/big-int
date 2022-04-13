mod big_int;
mod macros;

#[derive(Debug)]
pub enum Errors {
    ConversionError,
}

pub type BigInt = big_int::BigInt;
pub type RNSRepresentation = big_int::RNSRepresentation;
