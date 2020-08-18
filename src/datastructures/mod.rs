mod matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct InputError {
    pub message: String
}

pub use self::matrix::Matrix;