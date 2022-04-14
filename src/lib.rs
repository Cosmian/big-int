mod big_int;
mod macros;

#[derive(Debug)]
pub enum Errors {
    ConversionError,
}

pub use big_int::BigInt;
pub use big_int::RNSRepresentation;
