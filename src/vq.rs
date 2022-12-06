use crate::mt::Matrix;
use std::ops::{Add, Mul, Sub};
#[derive(Debug, PartialEq, Clone)]
pub struct VectorQuantity {
    target: Vec<f64>,
}
#[derive(Debug, PartialEq)]
pub struct Span {
    gather: Vec<VectorQuantity>,
}

impl Span {
    pub fn new(gather: Vec<VectorQuantity>) -> Self {
        Span { gather }
    }

    pub fn to_matrix_with(self, a_vq: VectorQuantity) -> Matrix {
        assert_eq!(self.gather[0].row_count(), a_vq.row_count());
        let row_count = a_vq.row_count();
        let mut target = Vec::new();
        for row in 0..row_count {
            let mut temp_target = Vec::new();
            for item in self.gather.iter() {
                temp_target.push(item.target[row]);
            }
            temp_target.push(a_vq.target[row]);
            target.push(temp_target);
        }
        Matrix::from(target)
    }

    pub fn to_matrix(self) -> Matrix {
        let row_count = self.gather[0].row_count();
        let mut target = Vec::new();
        for row in 0..row_count {
            let mut temp_target = Vec::new();
            for item in self.gather.iter() {
                temp_target.push(item.target[row]);
            }
            target.push(temp_target);
        }
        Matrix::from(target)
    }
}

impl VectorQuantity {
    pub fn new<A>(target: Vec<A>) -> Self
    where
        A: Into<f64>,
    {
        let target = target.into_iter().map(|x| x.into()).collect();
        VectorQuantity { target }
    }

    pub fn row_count(&self) -> usize {
        self.target.len()
    }

    pub fn target(&self) -> &Vec<f64> {
        &self.target
    }
}

impl Add for VectorQuantity {
    type Output = VectorQuantity;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.target.len(), rhs.target.len());
        let mut target = Vec::new();
        let len = self.target.len();
        for i in 0..len {
            target.push(self.target[i] + rhs.target[i]);
        }
        VectorQuantity { target }
    }
}

impl Sub for VectorQuantity {
    type Output = VectorQuantity;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.target.len(), rhs.target.len());
        let mut target = Vec::new();
        let len = self.target.len();
        for i in 0..len {
            target.push(self.target[i] - rhs.target[i]);
        }
        VectorQuantity { target }
    }
}

impl Mul<f64> for VectorQuantity {
    type Output = VectorQuantity;
    fn mul(self, rhs: f64) -> Self::Output {
        VectorQuantity {
            target: self.target.iter().map(|x| x * rhs).collect(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add() {
        let a = VectorQuantity::new(vec![1f64, 2f64, 3f64]);
        let b = VectorQuantity::new(vec![1f64, 2f64, 3f64]);
        let c = a + b;
        let d = VectorQuantity::new(vec![2f64, 4f64, 6f64]);

        assert_eq!(c, d);
    }

    #[test]
    fn test_sub() {
        let a = VectorQuantity::new(vec![1f64, 2f64, 3f64]);
        let b = VectorQuantity::new(vec![1f64, 2f64, 3f64]);
        let c = a - b;
        let d = VectorQuantity::new(vec![0f64, 0f64, 0f64]);

        assert_eq!(c, d);
    }

    #[test]
    fn test_mul() {
        let a = VectorQuantity::new(vec![1f64, 2f64, 3f64]);
        let b = a * 2f64;
        let c = VectorQuantity::new(vec![2f64, 4f64, 6f64]);
        assert_eq!(b, c);
    }

    #[test]
    fn test_to_matrix_with() {
        let a = VectorQuantity::new(vec![1f64, 2f64, 3f64]);
        let b = VectorQuantity::new(vec![2f64, 4f64, 6f64]);
        let c = VectorQuantity::new(vec![3f64, 6f64, 9f64]);
        let gather = vec![a, b, c];
        let the_span = Span::new(gather);
        let d = VectorQuantity::new(vec![0f64, 0f64, 0f64]);
        let the_matrix = the_span.to_matrix_with(d);
        let the_matrix_1 = Matrix::from(vec![
            vec![1f64, 2f64, 3f64, 0f64],
            vec![2f64, 4f64, 6f64, 0f64],
            vec![3f64, 6f64, 9f64, 0f64],
        ]);
        assert_eq!(the_matrix, the_matrix_1);
    }

    #[test]
    fn test_to_matrix() {
        let a = VectorQuantity::new(vec![1f64, 2f64, 3f64]);
        let b = VectorQuantity::new(vec![2f64, 4f64, 6f64]);
        let c = VectorQuantity::new(vec![3f64, 6f64, 9f64]);
        let gather = vec![a, b, c];
        let the_span = Span::new(gather);

        let the_matrix = the_span.to_matrix();
        let the_matrix_1 = Matrix::from(vec![
            vec![1f64, 2f64, 3f64],
            vec![2f64, 4f64, 6f64],
            vec![3f64, 6f64, 9f64],
        ]);
        assert_eq!(the_matrix, the_matrix_1);
    }
}
