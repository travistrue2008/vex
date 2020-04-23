use crate::common;
use crate::vector2::Vector2;

use std::cmp;
use std::fmt;
use std::fmt::{Display, Formatter};

use std::ops::{
    Neg,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
};

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct Matrix2 {
    pub m: [f32; 4],
}

impl Matrix2 {
    /// Creates a matrix set to its identity
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::new();
    /// assert_eq!(actual.m, [
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    /// ]);
    /// ```
    #[inline]
    pub fn new() -> Matrix2 {
        Matrix2 {
            m: [
                1.0, 0.0,
                0.0, 1.0,
            ],
        }
    }

    /// Creates a matrix from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// let expected = [1.0, 2.0, 3.0, 4.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn make(m11: f32, m21: f32, m12: f32, m22: f32) -> Matrix2 {
        Matrix2 {
            m: [m11, m21, m12, m22],
        }
    }

    /// Gets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual.m11(), 1.0);
    /// ```
    #[inline]
    pub fn m11(&self) -> f32 {
        self.m[0]
    }

    /// Gets the value for the m21 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual.m21(), 2.0);
    /// ```
    #[inline]
    pub fn m21(&self) -> f32 {
        self.m[1]
    }

    /// Gets the value for the m12 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual.m12(), 3.0);
    /// ```
    #[inline]
    pub fn m12(&self) -> f32 {
        self.m[2]
    }

    /// Gets the value for the m22 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual.m22(), 4.0);
    /// ```
    #[inline]
    pub fn m22(&self) -> f32 {
        self.m[3]
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(0.0, 0.0, 0.0, 0.0);
    /// actual.set_m11(1.0);
    /// let expected = [1.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m11(&mut self, v: f32) {
        self.m[0] = v;
    }

    /// Sets the value for the m21 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(0.0, 0.0, 0.0, 0.0);
    /// actual.set_m21(1.0);
    /// let expected = [0.0, 1.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m21(&mut self, v: f32) {
        self.m[1] = v;
    }

    /// Sets the value for the m12 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(0.0, 0.0, 0.0, 0.0);
    /// actual.set_m12(1.0);
    /// let expected = [0.0, 0.0, 1.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m12(&mut self, v: f32) {
        self.m[2] = v;
    }

    /// Sets the value for the m22 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(0.0, 0.0, 0.0, 0.0);
    /// actual.set_m22(1.0);
    /// let expected = [0.0, 0.0, 0.0, 1.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m22(&mut self, v: f32) {
        self.m[3] = v;
    }

    /// Sets the internal contents of the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::new();
    /// actual.set(1.0, 2.0, 3.0, 4.0);
    /// let expected = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn set(&mut self, m11: f32, m21: f32, m12: f32, m22: f32) {
        self.set_m11(m11);
        self.set_m21(m21);
        self.set_m12(m12);
        self.set_m22(m22);
    }

    /// Transposes the matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// actual.transpose();
    /// let expected = Matrix2::make(1.0, 3.0, 2.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn transpose(&mut self) {
        let temp = self.m[1];
        self.m[1] = self.m[2];
        self.m[2] = temp;
    }

    /// Find the matrix's determinant
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0).determinant();
    /// assert_eq!(actual, -2.0);
    /// ```
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.m11() * self.m22() - self.m12() * self.m21()
    }

    /// Inverses the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// actual.inverse();
    /// let expected = Matrix2::make(-2.0, 1.0, 1.5, -0.5);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn inverse(&mut self) -> bool {
        let det = self.determinant();
        if det == 0.0 {
            return false;
        }

        let inv_det = 1.0 / det;
        let m11 = self.m22() * inv_det;
        let m21 = -self.m21() * inv_det;
        let m12 = -self.m12() * inv_det;
        let m22 = self.m11() * inv_det;

        self.set_m11(m11);
        self.set_m21(m21);
        self.set_m12(m12);
        self.set_m22(m22);
        true
    }

    /// Determine whether or not all elements of the matrix are valid
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// assert!(actual.is_valid());
    /// ```
    #[inline]
    pub fn is_valid(&self) -> bool {
        for i in 0..4 {
            if !common::is_valid(self.m[i]) {
                return false;
            }
        }

        true
    }
}

impl Neg for Matrix2 {
    type Output = Matrix2;

    /// Negates the matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = -Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// let expected = Matrix2::make(-1.0, -2.0, -3.0, -4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn neg(self) -> Matrix2 {
        let mut m = [0.0; 4];

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                m[i] = -*elem;
            }
        }

        Matrix2 { m }
    }
}

impl Add<f32> for Matrix2 {
    type Output = Matrix2;

    /// Find the resulting matrix by adding a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0) + 1.0;
    /// let expected = Matrix2::make(2.0, 3.0, 4.0, 5.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: f32) -> Matrix2 {
        let mut mat = Matrix2::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem + _rhs;
            }
        }

        mat
    }
}

impl Add<Matrix2> for Matrix2 {
    type Output = Matrix2;

    /// Add two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let a = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// let b = Matrix2::make(5.0, 6.0, 7.0, 8.0);
    /// let actual = a + b;
    /// let expected = Matrix2::make(6.0, 8.0, 10.0, 12.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: Matrix2) -> Matrix2 {
        let mut mat = Matrix2::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem + _rhs.m[i];
            }
        }

        mat
    }
}

impl AddAssign<f32> for Matrix2 {
    /// Increment a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// actual += 10.0;
    /// let expected = Matrix2::make(11.0, 12.0, 13.0, 14.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: f32) {
        unsafe {
            for elem in self.m.iter_mut() {
                *elem += _rhs;
            }
        }
    }
}

impl AddAssign<Matrix2> for Matrix2 {
    /// Increment a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// actual += Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// let expected = Matrix2::make(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: Matrix2) {
        unsafe {
            for (i, elem) in self.m.iter_mut().enumerate() {
                *elem += _rhs.m[i];
            }
        }
    }
}

impl Sub<f32> for Matrix2 {
    type Output = Matrix2;

    /// Find the resulting matrix by subtracting a scalar from a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0) - 10.0;
    /// let expected = Matrix2::make(-9.0, -8.0, -7.0, -6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: f32) -> Matrix2 {
        let mut mat = Matrix2::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem - _rhs;
            }
        }

        mat
    }
}

impl Sub<Matrix2> for Matrix2 {
    type Output = Matrix2;

    /// Subtract two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let a = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// let b = Matrix2::make(5.0, 4.0, 3.0, 2.0);
    /// let actual = a - b;
    /// let expected = Matrix2::make(-4.0, -2.0, 0.0, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: Matrix2) -> Matrix2 {
        let mut mat = Matrix2::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem - _rhs.m[i];
            }
        }

        mat
    }
}

impl SubAssign<f32> for Matrix2 {
    /// Decrement a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// actual -= 1.0;
    /// let expected = Matrix2::make(0.0, 1.0, 2.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: f32) {
        unsafe {
            for elem in self.m.iter_mut() {
                *elem -= _rhs;
            }
        }
    }
}

impl SubAssign<Matrix2> for Matrix2 {
    /// Decrement a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(2.0, 2.0, 3.0, 5.0);
    /// actual -= Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, Matrix2::new());
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: Matrix2) {
        unsafe {
            for (i, elem) in self.m.iter_mut().enumerate() {
                *elem -= _rhs.m[i];
            }
        }
    }
}

impl Mul<f32> for Matrix2 {
    type Output = Matrix2;

    /// Find the resulting matrix by multiplying a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0) * 2.0;
    /// let expected = Matrix2::make(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: f32) -> Matrix2 {
        let mut mat = Matrix2::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem * _rhs;
            }
        }

        mat
    }
}

impl Mul<Matrix2> for Matrix2 {
    type Output = Matrix2;

    /// Multiply two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let a = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// let b = Matrix2::make(5.0, 6.0, 7.0, 8.0);
    /// let actual = a * b;
    /// let expected = Matrix2::make(23.0, 34.0, 31.0, 46.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: Matrix2) -> Matrix2 {
        let m11 = self.m11() * _rhs.m11() + self.m12() * _rhs.m21();
        let m21 = self.m21() * _rhs.m11() + self.m22() * _rhs.m21();
        let m12 = self.m11() * _rhs.m12() + self.m12() * _rhs.m22();
        let m22 = self.m21() * _rhs.m12() + self.m22() * _rhs.m22();
        Matrix2::make(m11, m21, m12, m22)
    }
}

impl MulAssign<f32> for Matrix2 {
    /// Multiply a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// actual *= 2.0;
    /// let expected = Matrix2::make(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: f32) {
        unsafe {
            for elem in self.m.iter_mut() {
                *elem *= _rhs;
            }
        }
    }
}

impl MulAssign<Matrix2> for Matrix2 {
    /// Multiply a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// actual *= Matrix2::make(5.0, 6.0, 7.0, 8.0);
    /// let expected = Matrix2::make(23.0, 34.0, 31.0, 46.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: Matrix2) {
        let res = *self * _rhs;
        self.m = res.m;
    }
}

impl Div<f32> for Matrix2 {
    type Output = Matrix2;

    /// Find the resulting matrix by dividing a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let actual = Matrix2::make(1.0, 2.0, 3.0, 4.0) / 2.0;
    /// let expected = Matrix2::make(0.5, 1.0, 1.5, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div(self, _rhs: f32) -> Matrix2 {
        let mut mat = Matrix2::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem / _rhs;
            }
        }

        mat
    }
}

impl DivAssign<f32> for Matrix2 {
    /// Divide a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    /// 
    /// let mut actual = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// actual /= 2.0;
    /// let expected = Matrix2::make(0.5, 1.0, 1.5, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div_assign(&mut self, _rhs: f32) {
        unsafe {
            for elem in self.m.iter_mut() {
                *elem /= _rhs;
            }
        }
    }
}

impl cmp::PartialEq for Matrix2 {
    /// Determines if two matrices' elements are equivalent
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix2;
    ///
    /// assert!(Matrix2::new() == Matrix2::new());
    /// ```
    #[inline]
    fn eq(&self, _rhs: &Matrix2) -> bool {
        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                if *elem != _rhs.m[i] {
                    return false;
                }
            }
        }

        true
    }
}

impl Display for Matrix2 {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "[\n  {}, {}\n  {}, {}\n]",
            self.m11(),
            self.m12(),
            self.m21(),
            self.m22()
        )
    }
}

impl common::Matrix<Vector2> for Matrix2 {
    /// Find the resulting vector given a vector and matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix;
    /// use vex::Matrix2;
    /// use vex::Vector2;
    ///
    /// let m = Matrix2::make(1.0, 2.0, 3.0, 4.0);
    /// let v = Vector2::make(1.0, 2.0);
    /// let actual = m.transform_point(&v);
    /// let expected = Vector2::make(7.0, 10.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn transform_point(&self, point: &Vector2) -> Vector2 {
        Vector2::make(
            self.m11() * point.x + self.m12() * point.y,
            self.m21() * point.x + self.m22() * point.y,
        )
    }
}
