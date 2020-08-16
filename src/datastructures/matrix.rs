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
#[derive(Clone)]
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
}