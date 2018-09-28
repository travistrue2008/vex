use super::mat2::Mat2;
use super::math;
use super::vec2::Vec2;
use super::vec3::Vec3;
use std::cmp;
use std::fmt;
use std::ops;

#[derive(Copy, Clone)]
pub struct Mat3 {
    m: [f32; 9],
}

impl Mat3 {
    /// Creates a matrix set to an identity
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::new();
    /// let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn new() -> Mat3 {
        Mat3 {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a matrix from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn construct(
        m11: f32,
        m21: f32,
        m31: f32,
        m12: f32,
        m22: f32,
        m32: f32,
        m13: f32,
        m23: f32,
        m33: f32,
    ) -> Mat3 {
        Mat3 {
            m: [m11, m21, m31, m12, m22, m32, m13, m23, m33],
        }
    }

    /// Creates a matrix from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// use vex::Vec3;
    ///
    /// let position = Vec3::construct(0.0, 1.0, 1.0);
    /// let target = Vec3::new();
    /// let actual = Mat3::look_at(position, target, Vec3::up());
    /// let expected = [
    ///    1.0, -0.0,         0.0,        // column 1
    ///    0.0,  0.70710677, -0.70710677, // column 2
    ///   -0.0,  0.70710677,  0.70710677, // column 3
    /// ];
    ///
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn look_at(position: Vec3, target: Vec3, up: Vec3) -> Mat3 {
        let mut forward = target - position;
        forward.normalize();

        let mut right = Vec3::cross(&forward, &up);
        right.normalize();
        let up = Vec3::cross(&right, &forward);

        Mat3::construct(
            right.x, right.y, right.z, up.x, up.y, up.z, -forward.x, -forward.y, -forward.z,
        )
    }

    /// Gets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m11(), 1.0);
    /// ```
    pub fn m11(&self) -> f32 {
        self.m[0]
    }

    /// Gets the value for the m21 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m21(), 2.0);
    /// ```
    pub fn m21(&self) -> f32 {
        self.m[1]
    }

    /// Gets the value for the m31 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m31(), 3.0);
    /// ```
    pub fn m31(&self) -> f32 {
        self.m[2]
    }

    /// Gets the value for the m12 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m12(), 4.0);
    /// ```
    pub fn m12(&self) -> f32 {
        self.m[3]
    }

    /// Gets the value for the m22 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m22(), 5.0);
    /// ```
    pub fn m22(&self) -> f32 {
        self.m[4]
    }

    /// Gets the value for the m32 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m32(), 6.0);
    /// ```
    pub fn m32(&self) -> f32 {
        self.m[5]
    }

    /// Gets the value for the m13 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m13(), 7.0);
    /// ```
    pub fn m13(&self) -> f32 {
        self.m[6]
    }

    /// Gets the value for the m23 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m23(), 8.0);
    /// ```
    pub fn m23(&self) -> f32 {
        self.m[7]
    }

    /// Gets the value for the m33 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual.m33(), 9.0);
    /// ```
    pub fn m33(&self) -> f32 {
        self.m[8]
    }

    /// Gets the internal contents of the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn m(&self) -> [f32; 9] {
        self.m
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m11(1.0);
    /// let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m11(&mut self, v: f32) {
        self.m[0] = v;
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m21(1.0);
    /// let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m21(&mut self, v: f32) {
        self.m[1] = v;
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m31(1.0);
    /// let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m31(&mut self, v: f32) {
        self.m[2] = v;
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m12(1.0);
    /// let expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m12(&mut self, v: f32) {
        self.m[3] = v;
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m22(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m22(&mut self, v: f32) {
        self.m[4] = v;
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m32(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m32(&mut self, v: f32) {
        self.m[5] = v;
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m13(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m13(&mut self, v: f32) {
        self.m[6] = v;
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m23(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m23(&mut self, v: f32) {
        self.m[7] = v;
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m33(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    /// assert_eq!(actual.m(), expected);
    /// ```
    pub fn set_m33(&mut self, v: f32) {
        self.m[8] = v;
    }

    /// Sets the internal contents of the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::new();
    /// actual.set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let expected = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn set(
        &mut self,
        m11: f32,
        m21: f32,
        m31: f32,
        m12: f32,
        m22: f32,
        m32: f32,
        m13: f32,
        m23: f32,
        m33: f32,
    ) {
        self.set_m11(m11);
        self.set_m21(m21);
        self.set_m31(m31);
        self.set_m12(m12);
        self.set_m22(m22);
        self.set_m32(m32);
        self.set_m13(m13);
        self.set_m23(m23);
        self.set_m33(m33);
    }

    /// Sets the matrix to its identity
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// actual.identity();
    /// assert_eq!(actual, Mat3::new());
    /// ```
    pub fn identity(&mut self) {
        self.set_m11(1.0);
        self.set_m21(0.0);
        self.set_m31(0.0);
        self.set_m12(0.0);
        self.set_m22(1.0);
        self.set_m32(0.0);
        self.set_m13(0.0);
        self.set_m23(0.0);
        self.set_m33(1.0);
    }

    /// Transposes the matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// actual.transpose();
    /// let expected = Mat3::construct(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn transpose(&mut self) {
        let mut m = self.m;

        let temp = m[1];
        m[1] = m[3];
        m[3] = temp;
        let temp = m[5];
        m[5] = m[7];
        m[7] = temp;
        let temp = m[2];
        m[2] = m[6];
        m[6] = temp;
        self.m = m;
    }

    /// Find the matrix's determinant
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0).determinant();
    /// assert_eq!(actual, 0.0);
    /// ```
    pub fn determinant(&self) -> f32 {
        self.m11() * (self.m22() * self.m33() - self.m23() * self.m32())
            - (self.m12() * (self.m21() * self.m33() - self.m23() * self.m31()))
            + (self.m13() * (self.m21() * self.m32() - self.m22() * self.m31()))
    }

    /// Inverses the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 0.0, 5.0, 2.0, 1.0, 6.0, 3.0, 4.0, 0.0);
    /// actual.inverse();
    /// let expected = Mat3::construct(-24.0, 20.0, -5.0, 18.0, -15.0, 4.0, 5.0, -4.0, 1.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn inverse(&mut self) -> bool {
        let det = self.determinant();
        if det == 0.0 {
            return false;
        }

        let inv_det = 1.0 / det;
        let m11 =
            Mat2::construct(self.m22(), self.m23(), self.m32(), self.m33()).determinant() * inv_det;
        let m21 =
            Mat2::construct(self.m23(), self.m21(), self.m33(), self.m31()).determinant() * inv_det;
        let m31 =
            Mat2::construct(self.m21(), self.m22(), self.m31(), self.m32()).determinant() * inv_det;
        let m12 =
            Mat2::construct(self.m13(), self.m12(), self.m33(), self.m32()).determinant() * inv_det;
        let m22 =
            Mat2::construct(self.m11(), self.m13(), self.m31(), self.m33()).determinant() * inv_det;
        let m32 =
            Mat2::construct(self.m12(), self.m11(), self.m32(), self.m31()).determinant() * inv_det;
        let m13 =
            Mat2::construct(self.m12(), self.m13(), self.m22(), self.m23()).determinant() * inv_det;
        let m23 =
            Mat2::construct(self.m13(), self.m11(), self.m23(), self.m21()).determinant() * inv_det;
        let m33 =
            Mat2::construct(self.m11(), self.m12(), self.m21(), self.m22()).determinant() * inv_det;

        self.set_m11(m11);
        self.set_m21(m21);
        self.set_m31(m31);
        self.set_m12(m12);
        self.set_m22(m22);
        self.set_m32(m32);
        self.set_m13(m13);
        self.set_m23(m23);
        self.set_m33(m33);
        true
    }

    /// Determine whether or not all elements of the matrix are valid
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// assert!(actual.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        math::is_valid(self.m11())
            && math::is_valid(self.m21())
            && math::is_valid(self.m31())
            && math::is_valid(self.m12())
            && math::is_valid(self.m22())
            && math::is_valid(self.m32())
            && math::is_valid(self.m13())
            && math::is_valid(self.m23())
            && math::is_valid(self.m33())
    }

    fn print(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[\n  {}, {}, {}\n  {}, {}, {}\n  {}, {}, {}\n]",
            self.m11(),
            self.m12(),
            self.m13(),
            self.m21(),
            self.m22(),
            self.m23(),
            self.m31(),
            self.m32(),
            self.m33(),
        )
    }
}

impl ops::Neg for Mat3 {
    type Output = Mat3;

    /// Negates the matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = -Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let expected = Mat3::construct(-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn neg(self) -> Mat3 {
        let mut m = [0.0; 9];
        for (i, elem) in self.m.iter().enumerate() {
            m[i] = -*elem;
        }

        Mat3 { m }
    }
}

impl ops::Add<f32> for Mat3 {
    type Output = Mat3;

    /// Find the resulting matrix by adding a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) + 1.0;
    /// let expected = Mat3::construct(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add(self, _rhs: f32) -> Mat3 {
        let mut mat = Mat3::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem + _rhs;
        }

        mat
    }
}

impl ops::Add<Mat3> for Mat3 {
    type Output = Mat3;

    /// Add two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let a = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let b = Mat3::construct(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let actual = a + b;
    /// let expected = Mat3::construct(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add(self, _rhs: Mat3) -> Mat3 {
        let mut mat = Mat3::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem + _rhs.m[i];
        }

        mat
    }
}

impl ops::AddAssign<f32> for Mat3 {
    /// Increment a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// actual += 10.0;
    /// let expected = Mat3::construct(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add_assign(&mut self, _rhs: f32) {
        for elem in self.m.iter_mut() {
            *elem += _rhs;
        }
    }
}

impl ops::AddAssign<Mat3> for Mat3 {
    /// Increment a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// actual += Mat3::construct(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let expected = Mat3::construct(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add_assign(&mut self, _rhs: Mat3) {
        for (i, elem) in self.m.iter_mut().enumerate() {
            *elem += _rhs.m[i];
        }
    }
}

impl ops::Sub<f32> for Mat3 {
    type Output = Mat3;

    /// Find the resulting matrix by subtracting a scalar from a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) - 10.0;
    /// let expected = Mat3::construct(-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub(self, _rhs: f32) -> Mat3 {
        let mut mat = Mat3::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem - _rhs;
        }

        mat
    }
}

impl ops::Sub<Mat3> for Mat3 {
    type Output = Mat3;

    /// Subtract two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let a = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let b = Mat3::construct(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let actual = a - b;
    /// let expected = Mat3::construct(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub(self, _rhs: Mat3) -> Mat3 {
        let mut mat = Mat3::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem - _rhs.m[i];
        }

        mat
    }
}

impl ops::SubAssign<f32> for Mat3 {
    /// Decrement a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// actual -= 1.0;
    /// let expected = Mat3::construct(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub_assign(&mut self, _rhs: f32) {
        for elem in self.m.iter_mut() {
            *elem -= _rhs;
        }
    }
}

impl ops::SubAssign<Mat3> for Mat3 {
    /// Decrement a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
    /// actual -= Mat3::construct(1.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 9.0, 9.0);
    /// assert_eq!(actual, Mat3::new());
    /// ```
    fn sub_assign(&mut self, _rhs: Mat3) {
        for (i, elem) in self.m.iter_mut().enumerate() {
            *elem -= _rhs.m[i];
        }
    }
}

impl ops::Mul<f32> for Mat3 {
    type Output = Mat3;

    /// Find the resulting matrix by multiplying a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 2.0;
    /// let expected = Mat3::construct(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul(self, _rhs: f32) -> Mat3 {
        let mut mat = Mat3::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem * _rhs;
        }

        mat
    }
}

impl ops::Mul<Mat3> for Mat3 {
    type Output = Mat3;

    /// Multiply two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let a = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let b = Mat3::construct(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let actual = a * b;
    /// let expected = Mat3::construct(90.0, 114.0, 138.0, 54.0, 69.0, 84.0, 18.0, 24.0, 30.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul(self, _rhs: Mat3) -> Mat3 {
        let m11 = self.m11() * _rhs.m11() + self.m12() * _rhs.m21() + self.m13() * _rhs.m31();
        let m21 = self.m21() * _rhs.m11() + self.m22() * _rhs.m21() + self.m23() * _rhs.m31();
        let m31 = self.m31() * _rhs.m11() + self.m32() * _rhs.m21() + self.m33() * _rhs.m31();
        let m12 = self.m11() * _rhs.m12() + self.m12() * _rhs.m22() + self.m13() * _rhs.m32();
        let m22 = self.m21() * _rhs.m12() + self.m22() * _rhs.m22() + self.m23() * _rhs.m32();
        let m32 = self.m31() * _rhs.m12() + self.m32() * _rhs.m22() + self.m33() * _rhs.m32();
        let m13 = self.m11() * _rhs.m13() + self.m12() * _rhs.m23() + self.m13() * _rhs.m33();
        let m23 = self.m21() * _rhs.m13() + self.m22() * _rhs.m23() + self.m23() * _rhs.m33();
        let m33 = self.m31() * _rhs.m13() + self.m32() * _rhs.m23() + self.m33() * _rhs.m33();
        Mat3::construct(m11, m21, m31, m12, m22, m32, m13, m23, m33)
    }
}

impl ops::MulAssign<f32> for Mat3 {
    /// Multiply a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// actual *= 2.0;
    /// let expected = Mat3::construct(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul_assign(&mut self, _rhs: f32) {
        for elem in self.m.iter_mut() {
            *elem *= _rhs;
        }
    }
}

impl ops::MulAssign<Mat3> for Mat3 {
    /// Multiply a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// actual *= Mat3::construct(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let expected = Mat3::construct(90.0, 114.0, 138.0, 54.0, 69.0, 84.0, 18.0, 24.0, 30.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul_assign(&mut self, _rhs: Mat3) {
        let res = *self * _rhs;
        self.m = res.m;
    }
}

impl ops::Div<f32> for Mat3 {
    type Output = Mat3;

    /// Find the resulting matrix by dividing a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) / 2.0;
    /// let expected = Mat3::construct(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5);
    /// assert_eq!(actual, expected);
    /// ```
    fn div(self, _rhs: f32) -> Mat3 {
        let mut mat = Mat3::new();
        for (i, elem) in self.m.iter().enumerate() {
            mat.m[i] = *elem / _rhs;
        }

        mat
    }
}

impl ops::DivAssign<f32> for Mat3 {
    /// Divide a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// let mut actual = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// actual /= 2.0;
    /// let expected = Mat3::construct(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5);
    /// assert_eq!(actual, expected);
    /// ```
    fn div_assign(&mut self, _rhs: f32) {
        for elem in self.m.iter_mut() {
            *elem /= _rhs;
        }
    }
}

impl cmp::PartialEq for Mat3 {
    /// Determines if two matrices' elements are equivalent
    ///
    /// # Examples
    /// ```
    /// use vex::Mat3;
    /// assert!(Mat3::new() == Mat3::new());
    /// ```
    fn eq(&self, _rhs: &Mat3) -> bool {
        self.m11() == _rhs.m11()
            && self.m21() == _rhs.m21()
            && self.m31() == _rhs.m31()
            && self.m12() == _rhs.m12()
            && self.m22() == _rhs.m22()
            && self.m32() == _rhs.m32()
            && self.m13() == _rhs.m13()
            && self.m23() == _rhs.m23()
            && self.m33() == _rhs.m33()
    }
}

impl fmt::Debug for Mat3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl fmt::Display for Mat3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl math::TransformPoint<Vec2> for Mat3 {
    /// Find the resulting vector given a vector and matrix
    ///
    /// # Examples
    /// ```
    /// use vex::math::TransformPoint;
    /// use vex::Mat3;
    /// use vex::Vec2;
    /// let m = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let v = Vec2::construct(1.0, 2.0);
    /// let actual = m.transform_point(&v);
    /// let expected = Vec2::construct(16.0, 20.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn transform_point(&self, point: &Vec2) -> Vec2 {
        Vec2::construct(
            self.m11() * point.x + self.m12() * point.y + self.m13(),
            self.m21() * point.x + self.m22() * point.y + self.m23(),
        )
    }
}

impl math::TransformPoint<Vec3> for Mat3 {
    /// Find the resulting vector given a vector and matrix
    ///
    /// # Examples
    /// ```
    /// use vex::math::TransformPoint;
    /// use vex::Mat3;
    /// use vex::Vec3;
    /// let m = Mat3::construct(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    /// let v = Vec3::construct(1.0, 2.0, 3.0);
    /// let actual = m.transform_point(&v);
    /// let expected = Vec3::construct(30.0, 36.0, 42.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn transform_point(&self, point: &Vec3) -> Vec3 {
        Vec3::construct(
            self.m11() * point.x + self.m12() * point.y + self.m13() * point.z,
            self.m21() * point.x + self.m22() * point.y + self.m23() * point.z,
            self.m31() * point.x + self.m32() * point.y + self.m33() * point.z,
        )
    }
}
