use matrix::*;
fn main() {
    let a = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
    let b = Matrix::from(vec![vec![4, 3, 6], vec![1, -2, 3]]);

    let c = a * b;
    println!("{c}");

    let a = Matrix::from(vec![
        vec![2, -5, 0],
        vec![-1, 3, -4],
        vec![6, -8, -7],
        vec![-3, 0, 9],
    ]);
    let b = Matrix::from(vec![vec![4, -6], vec![7, 1], vec![3, 2]]);

    let c = a.thread_mul_matrix(b);
    println!("{c}");

    let a = Matrix::from(vec![vec![5, 1], vec![3, -2]]);
    let b = Matrix::from(vec![vec![2, 0], vec![4, 3]]);
    let c = a.clone() * b.clone();
    println!("{c}");
    let d = b * a;
    println!("{d}");

    let mut a = Matrix::from(vec![
        vec![3, 0, -6, 0, 0],
        vec![0, 3, 0, -6, 0],
        vec![-1, 0, 2, 0, 0],
        vec![0, -1, 0, 2, 0],
    ]);
    let _ = a.rref();
    println!("{a}");

    let a = Matrix::from(vec![vec![3, -6], vec![-1, 2]]);
    let b = Matrix::from(vec![vec![2, 2], vec![1, 1]]);
    println!("{}", a * b);
}
