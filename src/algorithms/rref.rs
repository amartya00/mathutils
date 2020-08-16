use num::Num;
use std::ops::Neg;
use super::super::datastructures::Matrix;
use super::InputError;

///
/// This function converts a matr
pub fn gaussian_elimination<T: Num + Copy + Neg<Output=T>>(matrix: &mut Matrix<T>)-> Result<(), InputError> {
    // Step 0: Validate if matrix is in augmented form
    if matrix.cols > matrix.rows+1 {
        return Err(
            InputError {
                message: format!("The matrix has {} rows and {} cols. This cannot an augmented matrix with a unique solution.", matrix.rows, matrix.cols)
            }
        )
    }

    // Step 1: Turn to row echelon form
    for i in 0..(matrix.rows) {
        // If pivot is 0, we need to do row operations to turn it non 0
        if matrix[i][i].is_zero() {
            match matrix.find_next_pivot( i, i) {
                None=>{ return Err(InputError { message: format!("The input matrix has no solutions") });}
                Some(r) => { matrix.exchange_rows(i, r); }
            }
        }

        // Normalize pivot
        matrix.scalar(i, T::one()/matrix[i][i]);

        // Operate on subsequent rows.
        // TODO: Parallelize this.
        for other_rows in 0..matrix.rows {
            if matrix[other_rows][i].is_zero() || other_rows == i {
                continue;
            }
            let div: T = -matrix[other_rows][i];
            matrix.linear_comb_replace(T::one(), other_rows, div, i);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::InputError;
    use super::super::super::datastructures::Matrix;

    fn round_matrix_to_2_digits(matrix: &mut Matrix<f64>) {
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                matrix[i][j] = (matrix[i][j]*100f64).round()/100f64;
            }
        }
    }

    #[test]
    fn test_gaussian_happy_case() {
        let mut matrix: Matrix<f64> =
            Matrix::from_vectors(vec![
                vec![11f64, 22f64, 17f64, 100f64],
                vec![ 0f64,  0f64, 22f64, 200f64],
                vec![19f64, 82f64, 67f64, 300f64],
            ]
        );
        assert_eq!(Ok(()), super::gaussian_elimination(&mut matrix));

        round_matrix_to_2_digits(&mut matrix);

        assert_eq!((4.81f64, -4.88f64, 9.09f64), (matrix[0][3], matrix[1][3], matrix[2][3]));
        matrix = Matrix::from_vectors(vec![
            vec![11f64, 22f64,  17f64, 100f64, 100f64],
            vec![11f64, 22f64,  99f64, 123f64, 145f64],
            vec![ 0f64,  0f64,  36f64,  45f64, 123f64],
            vec![21f64, 21f64, 634f64, 987f64, 1213f64]
        ]);
        assert_eq!(Ok(()), super::gaussian_elimination(&mut matrix));
        round_matrix_to_2_digits(&mut matrix);
        assert_eq!((-128.21f64, 55.42f64, -0.28f64, 2.96f64), (matrix[0][4], matrix[1][4], matrix[2][4], matrix[3][4]));
    }

    #[test]
    fn test_gaussian_no_solution() {
        let mut matrix: Matrix<f64> = Matrix::from_vectors(vec![
            vec![11f64, 22f64, 17f64, 100f64, 100f64],
            vec![11f64, 22f64, 99f64, 123f64, 145f64],
            vec![ 1f64,  2f64, 36f64,  45f64, 123f64],
            vec![ 2f64,  4f64, 63f64,  98f64, 1413f64]
        ]);
        assert_eq!(Err(InputError { message: format!("The input matrix has no solutions") }), super::gaussian_elimination(&mut matrix));
    }

    #[test]
    fn test_err_on_non_augmented_form() {
        let mut matrix: Matrix<f64> = Matrix::from_vectors(vec![
            vec![11f64, 22f64, 17f64, 100f64, 100f64],
            vec![11f64, 22f64, 99f64, 123f64, 145f64],
            vec![ 1f64,  2f64, 36f64,  45f64, 123f64]
        ]);
        assert_eq!(
            Err(InputError{message: "The matrix has 3 rows and 5 cols. This cannot an augmented matrix with a unique solution.".to_string()}),
            super::gaussian_elimination(&mut matrix)
        );
    }
}