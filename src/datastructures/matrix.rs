use num::Num;
use std::ops::{Mul, Index, IndexMut};

///
/// Matrix stricture. This represents a 2D matrix of elements of type `T` with some constrains.
/// `T` must implement the following traits:
///   - `Copy`. Operations on a matrix like linear combination of rows etc need this. Moving will mess up the row.
///   - `Num`. Matrix operations have to happen on elements that support numeric operations.
///   - `PartialEq`. The elements have to be equated with Zero for the type.
///   - `Mul`. Since, `Num` does not include the multiplication operation, this has to be separately defined.
///
/// This structure defines methods for basic row operations:
///   - Linear combination
///   - Exchange
///   - Scalar multiplication
///   - Find next pivot element after a row. This is necessary in algorithms like Gauss Jordan, if the current pivot is 0.
///
/// Matrix also has index operations defined. You can do `matrix[i][j]` on both mutable and immutable objects/references.
///
#[derive(Clone, Debug)]
pub struct Matrix<T: Copy + Num + PartialEq + Mul<Output=T>> {
    pub rows: usize,
    pub cols: usize,
    store: Vec<Vec<T>>
}

impl <T: Copy + Num + PartialEq + Mul<Output=T>> Matrix<T> {
    ///
    /// Constructor.
    /// Arguments:
    ///   - rows: Number of rows.
    ///   - cols: Number of columns in matrix.
    /// Returns:
    ///   Matrix (rows x cols)
    ///
    pub fn new(rows: usize, cols: usize)-> Matrix<T> {
        let mut store: Vec<Vec<T>> = Vec::with_capacity(rows);
        for _ in 0..rows {
            store.push(vec![T::zero(); cols]);
        }
        Matrix { rows, cols, store }
    }

    ///
    /// Constructor. This moves the input vector's ownership to itself.
    /// Arguments:
    ///   - Slice of vectors.
    /// Returns:
    ///   - Matrix
    ///
    pub fn from_vectors(matrix: Vec<Vec<T>>)-> Matrix<T> {
        let rows = matrix.len();
        assert!(rows > 0, "Cannot create matrix out of empty vector.");
        let cols = matrix[0].len();
        assert!(cols > 0, "Number of columns cannot be 0.");
        for i in 0..rows {
            assert_eq!(cols, matrix[i].len(), "Unequal columns in rows.");
        }
        Matrix { rows, cols, store: matrix }
    }

    ///
    /// Constructor. This returns an `nxn` identity matrix.
    /// Arguments:
    ///   - Size if identity matrix
    /// Returns:
    ///   - Identity matrix of size `n`.
    ///
    pub fn identity(n: usize)-> Matrix<T> {
        let mut retval = Matrix::new(n, n);
        for i in 0..n {
            retval[i][i] = T::one();
        }
        retval
    }

    ///
    /// Matrix method to exchange 2 rows.
    /// Arguments:
    ///   - r1 and r2: Rows to swap.
    /// Returns:
    ///   Nothing
    ///
    pub fn exchange_rows(&mut self, r1: usize, r2: usize) {
        self.store.swap(r1, r2)
    }

    ///
    /// Matrix method to replace a row with a linear combination of itself and another row.
    /// R1 -> AR1 + BR2
    /// Arguments:
    ///   - a: `A` in the formula above.
    ///   - r1: The row to replace.
    ///   - b: `B` in the formula above.
    ///   - r2: The second row involved in the linear combination
    /// Returns:
    ///   Nothing
    ///
    pub fn linear_comb_replace(
        &mut self,
        a: T, r1: usize,
        b: T, r2: usize) {
        for i in 0..self.cols {
            self.store[r1][i] = a*self.store[r1][i] + b*self.store[r2][i];
        }
    }

    ///
    /// Matrix method to find the next pivot element. This function takes a row/col and looks at the
    /// rows below the specified row to check which one has a non zero element in that col.
    /// This is useful in algorithms like Gauss Jordan elimination where the pivot element might
    /// turn out ti be Zero during a reduction operation.
    ///
    /// If all the subsequent rows have 0 in the pivot column, this function returns a None, otherwise
    /// it returns an Option(row)
    ///
    /// Arguments:
    ///   - row: The pivot row.
    ///   - pivot: The pivot column.
    /// Returns:
    ///   Optional subsequent row. None if not found.
    ///
    pub fn find_next_pivot(&self, row: usize, pivot: usize) -> Option<usize> {
        for i in row+1..self.rows {
            if !self.store[i][pivot].is_zero() {
                return Some(i)
            }
        }
        return None
    }

    ///
    /// Matrix method to multiply a row with a scalar.
    /// Arguments:
    ///   - row: The row to scale.
    ///   - a: The scaling factor.
    /// Returns:
    ///   Nothing
    ///
    pub fn scalar(&mut self, row: usize, a: T) {
        for i in 0..self.cols {
            self.store[row][i] = self.store[row][i] * a;
        }
    }

    ///
    /// Matrix method to add with another matrix and return a new matrix. This function returns a
    /// `Result<Matrix, InputError>`. If the dimensions do not match, an Error is returned.
    /// Arguments:
    ///   - other: RHS matrix
    /// Returns:
    ///   New matrix that is a sum of this and RHS. (Err if input error)
    ///
    pub fn add(&self, other: &Matrix<T>)-> Result<Matrix<T>, super::InputError> {
        if self.rows != other.rows || self.cols != other.cols {
            Err(
                super::InputError {
                    message: format!(
                        "RHS matrix dims ({} x {}) not the same as this ({} x {})",
                        other.rows,
                        other.cols,
                        self.rows,
                        self.cols
                    )
                }
            )
        } else {
            let mut retval = Matrix::<T>::new(self.rows,self.cols);
            for r in  0..self.rows {
                for c in 0..self.cols {
                    retval[r][c] = self.store[r][c] + other.store[r][c];
                }
            }
            Ok(retval)
        }
    }

    ///
    /// Matrix method to multiply with another matrix and return a new matrix. This function returns a
    /// `Result<Matrix, InputError>`. If the dimensions do not match, an Error is returned.
    /// Arguments:
    ///   - other: RHS matrix
    /// Returns:
    ///   New matrix that is a product of this and RHS. (Err if input error)
    ///
    pub fn multiply(&self, other: &Matrix<T>)-> Result<Matrix<T>, super::InputError> {
        if self.cols != other.rows {
            Err(
                super::InputError {
                    message: format!("# cols of the matrix ({}) != # rows of RHS matrix ({})", self.cols, other.rows)
                }
            )
        } else {
            let mut retval = Matrix::new(self.rows, other.cols);
            for i in 0..self.rows {
                for j in 0..other.cols {
                    let mut acc: T = T::zero();
                    for k in 0..self.cols {
                        acc = acc + self[i][k]*other[k][j];
                    }
                    retval[i][j] = acc;
                }
            }
            Ok(retval)
        }
    }
}

impl <T: Copy + Num + PartialEq + Mul<Output=T>> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.store[index]
    }
}

impl <T: Copy + Num + PartialEq + Mul<Output=T>> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut [T] {
        &mut self.store[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::datastructures::Matrix;
    use num::Num;
    use std::ops::Mul;

    // We are omitting implementing the equality trait as it is an expensive operation, and is
    // not that important outside of testing scenarios.
    impl <T: Copy + Num + PartialEq + Mul<Output=T>> PartialEq for Matrix<T> {
        fn eq(&self, other: &Self) -> bool {
            self.rows == other.rows &&
                self.cols == other.cols &&
                {
                    let mut flag = true;
                    for r in 0..self.rows {
                        for c in 0..self.cols {
                            if self.store[r][c] != other.store[r][c] {
                                flag = false;
                                break;
                            }
                        }
                        if !flag {
                            break;
                        }
                    }
                    flag
                }
        }
    }

    #[test]
    fn test_constructor() {
        let my_matrix: super::Matrix<f64> = super::Matrix::new(3, 4);
        assert_eq!(3, my_matrix.rows);
        assert_eq!(4, my_matrix.cols);

        let my_matrix_2: super::Matrix<f64> = super::Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64, 10f64],
                vec![4f64, 5f64, 6f64, 20f64],
                vec![7f64, 8f64, 9f64, 30f64]
            ]
        );
        assert_eq!(3, my_matrix_2.rows);
        assert_eq!(4, my_matrix_2.cols);

        let id_matrix: Matrix<f64> = Matrix::identity(3);
        assert_eq!(
            Matrix::<f64>::from_vectors(
                vec![
                    vec![1f64, 0f64, 0f64],
                    vec![0f64, 1f64, 0f64],
                    vec![0f64, 0f64, 1f64]
                ]
            ),
            id_matrix
        )
    }

    #[test]
    fn test_access() {
        let mut my_matrix: super::Matrix<f64> = super::Matrix::new(3,4);
        my_matrix[0][0] = 100f64;
        my_matrix[1][3] = 200f64;
        assert_eq!(100f64, my_matrix[0][0]);

        let my_matrix_2: super::Matrix<f64> = super::Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64, 10f64],
                vec![4f64, 5f64, 6f64, 20f64],
                vec![7f64, 8f64, 9f64, 30f64]
            ]
        );
        assert_eq!( 1f64, my_matrix_2[0][0]);
        assert_eq!( 2f64, my_matrix_2[0][1]);
        assert_eq!( 3f64, my_matrix_2[0][2]);
        assert_eq!(10f64, my_matrix_2[0][3]);
        assert_eq!( 4f64, my_matrix_2[1][0]);
        assert_eq!( 5f64, my_matrix_2[1][1]);
        assert_eq!( 6f64, my_matrix_2[1][2]);
        assert_eq!(20f64, my_matrix_2[1][3]);
        assert_eq!( 7f64, my_matrix_2[2][0]);
        assert_eq!( 8f64, my_matrix_2[2][1]);
        assert_eq!( 9f64, my_matrix_2[2][2]);
        assert_eq!(30f64, my_matrix_2[2][3]);
    }

    #[test]
    fn test_exchange_rows() {
        let mut my_matrix_2: super::Matrix<f64> = super::Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64, 10f64],
                vec![4f64, 5f64, 6f64, 20f64],
                vec![7f64, 8f64, 9f64, 30f64]
            ]
        );
        assert_eq!(1f64, my_matrix_2[0][0]);
        assert_eq!(4f64, my_matrix_2[1][0]);

        my_matrix_2.exchange_rows(0, 1);

        assert_eq!(4f64, my_matrix_2[0][0]);
        assert_eq!(1f64, my_matrix_2[1][0]);
    }

    #[test]
    fn test_linear_comb_replace() {
        let mut my_matrix_2: super::Matrix<f64> = super::Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64, 10f64],
                vec![4f64, 5f64, 6f64, 20f64],
                vec![7f64, 8f64, 9f64, 30f64]
            ]
        );
        my_matrix_2.linear_comb_replace(1f64, 0, 2f64, 1);
        assert_eq!( 9f64, my_matrix_2[0][0]);
        assert_eq!(12f64, my_matrix_2[0][1]);
        assert_eq!(15f64, my_matrix_2[0][2]);
        assert_eq!(50f64, my_matrix_2[0][3]);

        assert_eq!( 4f64, my_matrix_2[1][0]);
        assert_eq!( 5f64, my_matrix_2[1][1]);
        assert_eq!( 6f64, my_matrix_2[1][2]);
        assert_eq!(20f64, my_matrix_2[1][3]);
    }

    #[test]
    fn test_next_pivot() {
        let mut my_matrix_2: super::Matrix<f64> = super::Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64, 10f64],
                vec![0f64, 5f64, 6f64, 20f64],
                vec![7f64, 8f64, 9f64, 30f64]
            ]
        );
        assert_eq!(Some(2), my_matrix_2.find_next_pivot(0, 0));
        my_matrix_2 = super::Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64, 10f64],
                vec![0f64, 5f64, 6f64, 20f64],
                vec![0f64, 8f64, 9f64, 30f64]
            ]
        );
        assert_eq!(None, my_matrix_2.find_next_pivot(0, 0));
    }

    #[test]
    fn test_scale() {
        let mut my_matrix_2: super::Matrix<f64> = super::Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64, 10f64],
                vec![0f64, 5f64, 6f64, 20f64],
                vec![7f64, 8f64, 9f64, 30f64]
            ]
        );
        my_matrix_2.scalar(0, 2f64);
        assert_eq!(2f64, my_matrix_2[0][0]);
        assert_eq!(4f64, my_matrix_2[0][1]);
        assert_eq!(6f64, my_matrix_2[0][2]);
        assert_eq!(20f64, my_matrix_2[0][3]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_inputs_uneven() {
        super::Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64, 10f64],
                vec![0f64, 5f64, 6f64],
                vec![7f64, 8f64, 9f64, 30f64]
            ]
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_inputs_no_rows() {
        super::Matrix::from_vectors(
            Vec::<Vec<f64>>::new()
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_inputs_no_cols() {
        super::Matrix::from_vectors(
            vec![
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                Vec::<f64>::new()
            ]
        );
    }

    #[test]
    fn test_add_happy_case() {
        let m1: Matrix<f64> = Matrix::from_vectors(
            vec![
                vec![1f64, 2f64],
                vec![3f64, 4f64]
            ]
        );

        let m2: Matrix<f64> = Matrix::from_vectors(
            vec![
                vec![5f64, 6f64],
                vec![7f64, 8f64]
            ]
        );

        assert!(
            m1.add(&m2).unwrap() ==
            Matrix::<f64>::from_vectors(
                vec![
                    vec![ 6f64,  8f64],
                    vec![10f64, 12f64]
                ]
            )
        )
    }

    #[test]
    fn test_add_error() {
        let m1: Matrix<f64> = Matrix::from_vectors(
            vec![
                vec![1f64, 2f64],
                vec![3f64, 4f64]
            ]
        );

        let m2: Matrix<f64> = Matrix::from_vectors(
            vec![
                vec![5f64, 6f64, 100f64],
                vec![7f64, 8f64, 200f64]
            ]
        );

        assert_eq!(
            Err(
                super::super::InputError {
                    message: "RHS matrix dims (2 x 3) not the same as this (2 x 2)".to_string()
                }
            ),
            m1.add(&m2)
        )
    }

    #[test]
    fn test_multiply_happy_case() {
        let m1: Matrix<f64> = Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 3f64],
                vec![4f64, 5f64, 6f64]
            ]
        );

        let m2: Matrix<f64> = Matrix::from_vectors(
            vec![
                vec![1f64],
                vec![2f64],
                vec![3f64]
            ]
        );

        assert!(
            m1.multiply(&m2).unwrap() ==
                Matrix::<f64>::from_vectors(
                    vec![
                        vec![14f64],
                        vec![32f64]
                    ]
                )
        );

        assert_eq!(
            m1,
            m1.multiply(
                &Matrix::identity(m1.cols)
            ).unwrap()
        )
    }

    #[test]
    fn test_multiply_error() {
        let m1: Matrix<f64> = Matrix::from_vectors(
            vec![
                vec![1f64, 2f64, 10f64],
                vec![3f64, 4f64, 11f64]
            ]
        );

        let m2: Matrix<f64> = Matrix::from_vectors(
            vec![
                vec![5f64, 6f64, 100f64],
                vec![7f64, 8f64, 200f64]
            ]
        );

        assert_eq!(
            Err(
                super::super::InputError {
                    message: "# cols of the matrix (3) != # rows of RHS matrix (2)".to_string()
                }
            ),
            m1.multiply(&m2)
        )
    }
}