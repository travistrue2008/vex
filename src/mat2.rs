use super::math;
use super::vec2::Vec2;
use std::cmp;
use std::fmt;
use std::ops;

#[derive(Copy, Clone)]
pub struct Mat2 {
    m: [f32; 4],
}

impl Mat2 {
    /// Creates a new matrix set to an identity
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let actual = Mat2::new();
    /// let expected = [1.0, 0.0, 0.0, 1.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn new() -> Mat2 {
        Mat2 {
            m: [1.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a new matrix from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// let expected = [1.0, 2.0, 3.0, 4.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn construct(m11: f32, m21: f32, m12: f32, m22: f32) -> Mat2 {
        Mat2 {
            m: [m11, m21, m12, m22],
        }
    }

    /// Gets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual.m11(), 1.0);
    /// ```
    pub fn m11(&self) -> f32 {
        self.m[0]
    }

    /// Gets the value for the m21 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual.m21(), 2.0);
    /// ```
    pub fn m21(&self) -> f32 {
        self.m[1]
    }

    /// Gets the value for the m12 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual.m12(), 3.0);
    /// ```
    pub fn m12(&self) -> f32 {
        self.m[2]
    }

    /// Gets the value for the m22 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual.m22(), 4.0);
    /// ```
    pub fn m22(&self) -> f32 {
        self.m[3]
    }

    /// Gets the internal contents of the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// let expected = [1.0, 2.0, 3.0, 4.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn m(&self) -> [f32; 4] {
        self.m
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let mut actual = Mat2::construct(0.0, 0.0, 0.0, 0.0);
    /// actual.set_m11(1.0);
    /// let expected = [1.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m11(&mut self, v: f32) {
        self.m[0] = v;
    }

    /// Sets the value for the m21 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let mut actual = Mat2::construct(0.0, 0.0, 0.0, 0.0);
    /// actual.set_m21(1.0);
    /// let expected = [0.0, 1.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m21(&mut self, v: f32) {
        self.m[1] = v;
    }

    /// Sets the value for the m12 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let mut actual = Mat2::construct(0.0, 0.0, 0.0, 0.0);
    /// actual.set_m12(1.0);
    /// let expected = [0.0, 0.0, 1.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m12(&mut self, v: f32) {
        self.m[2] = v;
    }

    /// Sets the value for the m22 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let mut actual = Mat2::construct(0.0, 0.0, 0.0, 0.0);
    /// actual.set_m22(1.0);
    /// let expected = [0.0, 0.0, 0.0, 1.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m22(&mut self, v: f32) {
        self.m[3] = v;
    }

    /// Sets the internal contents of the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let mut actual = Mat2::new();
    /// actual.set(1.0, 2.0, 3.0, 4.0);
    /// let expected = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn set(&mut self, m11: f32, m21: f32, m12: f32, m22: f32) {
        self.set_m11(m11);
        self.set_m21(m21);
        self.set_m12(m12);
        self.set_m22(m22);
    }

    /// Sets the matrix to its identity
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// actual.identity();
    /// assert_eq!(actual, Mat2::new());
    /// ```
    pub fn identity(&mut self) {
        self.set_m11(1.0);
        self.set_m21(0.0);
        self.set_m12(0.0);
        self.set_m22(1.0);
    }

    /// Transposes the matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// actual.transpose();
    /// let expected = Mat2::construct(1.0, 3.0, 2.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn transpose(&mut self) {
        let temp = self.m[1];
        self.m[1] = self.m[2];
        self.m[2] = temp;
    }

    /// Find the matrix's determinant
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0).determinant();
    /// assert_eq!(actual, -2.0);
    /// ```
    pub fn determinant(&self) -> f32 {
        self.m11() * self.m22() - self.m12() * self.m21()
    }

    /// Inverses the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat2;
    /// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// actual.inverse();
    /// let expected = Mat2::construct(-2.0, 1.0, 1.5, -0.5);
    /// assert_eq!(actual, expected);
    /// ```
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
    /// use vex::Mat2;
    /// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
    /// assert!(actual.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        math::is_valid(self.m11())
            && math::is_valid(self.m21())
            && math::is_valid(self.m12())
            && math::is_valid(self.m22())
    }

    fn print(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

/// Negates the matrix's elements
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let actual = -Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// let expected = Mat2::construct(-1.0, -2.0, -3.0, -4.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Neg for Mat2 {
    type Output = Mat2;

    fn neg(self) -> Mat2 {
        let mut m = [0.0; 4];
        for (i, elem) in self.m.iter().enumerate() {
            m[i] = -*elem;
        }

        Mat2 { m }
    }
}

/// Find the resulting matrix by adding a scalar to a matrix's elements
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0) + 1.0;
/// let expected = Mat2::construct(2.0, 3.0, 4.0, 5.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Add<f32> for Mat2 {
    type Output = Mat2;

    fn add(self, _rhs: f32) -> Mat2 {
        let mut mat = Mat2::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem + _rhs;
        }

        mat
    }
}

/// Add two matrices
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let a = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// let b = Mat2::construct(5.0, 6.0, 7.0, 8.0);
/// let actual = a + b;
/// let expected = Mat2::construct(6.0, 8.0, 10.0, 12.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Add<Mat2> for Mat2 {
    type Output = Mat2;

    fn add(self, _rhs: Mat2) -> Mat2 {
        let mut mat = Mat2::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem + _rhs.m[i];
        }

        mat
    }
}

/// Increment a matrix by a scalar
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// actual += 10.0;
/// let expected = Mat2::construct(11.0, 12.0, 13.0, 14.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::AddAssign<f32> for Mat2 {
    fn add_assign(&mut self, _rhs: f32) {
        for elem in self.m.iter_mut() {
            *elem += _rhs;
        }
    }
}

/// Increment a matrix by another matrix
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// actual += Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// let expected = Mat2::construct(2.0, 4.0, 6.0, 8.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::AddAssign<Mat2> for Mat2 {
    fn add_assign(&mut self, _rhs: Mat2) {
        for (i, elem) in self.m.iter_mut().enumerate() {
            *elem += _rhs.m[i];
        }
    }
}

/// Find the resulting matrix by subtracting a scalar from a matrix's elements
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0) - 10.0;
/// let expected = Mat2::construct(-9.0, -8.0, -7.0, -6.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Sub<f32> for Mat2 {
    type Output = Mat2;

    fn sub(self, _rhs: f32) -> Mat2 {
        let mut mat = Mat2::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem - _rhs;
        }

        mat
    }
}

/// Subtract two matrices
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let a = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// let b = Mat2::construct(5.0, 4.0, 3.0, 2.0);
/// let actual = a - b;
/// let expected = Mat2::construct(-4.0, -2.0, 0.0, 2.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Sub<Mat2> for Mat2 {
    type Output = Mat2;

    fn sub(self, _rhs: Mat2) -> Mat2 {
        let mut mat = Mat2::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem - _rhs.m[i];
        }

        mat
    }
}

/// Decrement a matrix by a scalar
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// actual -= 1.0;
/// let expected = Mat2::construct(0.0, 1.0, 2.0, 3.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::SubAssign<f32> for Mat2 {
    fn sub_assign(&mut self, _rhs: f32) {
        for elem in self.m.iter_mut() {
            *elem -= _rhs;
        }
    }
}

/// Decrement a matrix by another matrix
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let mut actual = Mat2::construct(2.0, 2.0, 3.0, 5.0);
/// actual -= Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// assert_eq!(actual, Mat2::new());
/// ```
impl ops::SubAssign<Mat2> for Mat2 {
    fn sub_assign(&mut self, _rhs: Mat2) {
        for (i, elem) in self.m.iter_mut().enumerate() {
            *elem -= _rhs.m[i];
        }
    }
}

/// Find the resulting matrix by multiplying a scalar to a matrix's elements
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0) * 2.0;
/// let expected = Mat2::construct(2.0, 4.0, 6.0, 8.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Mul<f32> for Mat2 {
    type Output = Mat2;

    fn mul(self, _rhs: f32) -> Mat2 {
        let mut mat = Mat2::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem * _rhs;
        }

        mat
    }
}

/// Find the resulting vector given a vector and matrix
///
/// # Examples
/// ```
/// use vex::Mat2;
/// use vex::Vec2;
/// let m = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// let v = Vec2::construct(1.0, 2.0);
/// let actual = m * v;
/// let expected = Vec2::construct(7.0, 10.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Mul<Vec2> for Mat2 {
    type Output = Vec2;

    fn mul(self, _rhs: Vec2) -> Vec2 {
        Vec2::construct(
            self.m11() * _rhs.x + self.m12() * _rhs.y,
            self.m21() * _rhs.x + self.m22() * _rhs.y,
        )
    }
}

/// Multiply two matrices
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let a = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// let b = Mat2::construct(5.0, 6.0, 7.0, 8.0);
/// let actual = a * b;
/// let expected = Mat2::construct(23.0, 34.0, 31.0, 46.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Mul<Mat2> for Mat2 {
    type Output = Mat2;

    fn mul(self, _rhs: Mat2) -> Mat2 {
        let m11 = self.m11() * _rhs.m11() + self.m12() * _rhs.m21();
        let m21 = self.m21() * _rhs.m11() + self.m22() * _rhs.m21();
        let m12 = self.m11() * _rhs.m12() + self.m12() * _rhs.m22();
        let m22 = self.m21() * _rhs.m12() + self.m22() * _rhs.m22();
        Mat2::construct(m11, m21, m12, m22)
    }
}

/// Multiply a matrix by a scalar
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// actual *= 2.0;
/// let expected = Mat2::construct(2.0, 4.0, 6.0, 8.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::MulAssign<f32> for Mat2 {
    fn mul_assign(&mut self, _rhs: f32) {
        for elem in self.m.iter_mut() {
            *elem *= _rhs;
        }
    }
}

/// Multiply a matrix by another matrix
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// actual *= Mat2::construct(5.0, 6.0, 7.0, 8.0);
/// let expected = Mat2::construct(23.0, 34.0, 31.0, 46.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::MulAssign<Mat2> for Mat2 {
    fn mul_assign(&mut self, _rhs: Mat2) {
        let res = *self * _rhs;
        self.m = res.m;
    }
}

/// Find the resulting matrix by dividing a scalar to a matrix's elements
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let actual = Mat2::construct(1.0, 2.0, 3.0, 4.0) / 2.0;
/// let expected = Mat2::construct(0.5, 1.0, 1.5, 2.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Div<f32> for Mat2 {
    type Output = Mat2;

    fn div(self, _rhs: f32) -> Mat2 {
        let mut mat = Mat2::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem / _rhs;
        }

        mat
    }
}

/// Divide a matrix by a scalar
///
/// # Examples
/// ```
/// use vex::Mat2;
/// let mut actual = Mat2::construct(1.0, 2.0, 3.0, 4.0);
/// actual /= 2.0;
/// let expected = Mat2::construct(0.5, 1.0, 1.5, 2.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::DivAssign<f32> for Mat2 {
    fn div_assign(&mut self, _rhs: f32) {
        for elem in self.m.iter_mut() {
            *elem /= _rhs;
        }
    }
}

/// Determines if two matrices' elements are equivalent
///
/// # Examples
/// ```
/// use vex::Mat2;
/// assert!(Mat2::new() == Mat2::new());
/// ```
impl cmp::PartialEq for Mat2 {
    fn eq(&self, _rhs: &Mat2) -> bool {
        self.m11() == _rhs.m11()
            && self.m21() == _rhs.m21()
            && self.m12() == _rhs.m12()
            && self.m22() == _rhs.m22()
    }
}

impl fmt::Debug for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}
