pub mod rref;

use std::fmt;
use std::error;

#[derive(Debug, Clone, PartialEq)]
pub struct InputError {
    pub message: String
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Error: {}", self.message)
    }
}

impl error::Error for InputError {
    fn description(&self) -> &str {
        &self.message
    }
}