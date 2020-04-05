use crate::common;
use crate::matrix3::Matrix3;
use crate::vector3::Vector3;
use crate::vector4::Vector4;

use std::cmp;
use std::fmt;
use std::fmt::{Display, Formatter};

use std::ops::{
    Index,
    IndexMut,
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

pub const IDENTITY: Matrix4 = Matrix4 {
    m: [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ],
};

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct Matrix4 {
    pub m: [f32; 16],
}

impl Matrix4 {
    /// Creates a matrix set to its identity
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// use vex::matrix4::IDENTITY;
    /// let actual = Matrix4::new();
    /// assert_eq!(actual, IDENTITY);
    /// ```
    #[inline]
    pub fn new() -> Matrix4 {
        IDENTITY
    }

    /// Creates a matrix from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn make(
        m11: f32,
        m21: f32,
        m31: f32,
        m41: f32,
        m12: f32,
        m22: f32,
        m32: f32,
        m42: f32,
        m13: f32,
        m23: f32,
        m33: f32,
        m43: f32,
        m14: f32,
        m24: f32,
        m34: f32,
        m44: f32,
    ) -> Matrix4 {
        Matrix4 {
            m: [
                m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44,
            ],
        }
    }

    /// Creates a orthogonal projection matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    ///
    /// let actual = Matrix4::ortho(-960.0, 960.0, 540.0, -540.0, -100.0, 100.0);
    /// let expected = [
    ///      0.0010416667,  0.0,           0.0,  0.0, // column 1
    ///      0.0,           0.0018518518,  0.0,  0.0, // column 2
    ///      0.0,           0.0,          -0.01, 0.0, // column 3
    ///     -0.0,          -0.0,          -0.0,  1.0, // column 4
    /// ];
    ///
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn ortho(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> Matrix4 {
        let width = right - left;
        let height = top - bottom;
        let depth = far - near;
        let tx = -(right + left) / width;
        let ty = -(top + bottom) / height;
        let tz = -(far + near) / depth;
        let mut mat = Matrix4::new();

        mat.set_m11(2.0 / width);
        mat.set_m22(2.0 / height);
        mat.set_m33(-2.0 / depth);
        mat.set_m14(tx);
        mat.set_m24(ty);
        mat.set_m34(tz);
        mat
    }

    /// Creates a orthogonal projection matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    ///
    /// let width = 1920;
    /// let height = 1080;
    /// let aspect_ratio = width as f32 / height as f32;
    /// let actual = Matrix4::perspective(75.0, aspect_ratio, 1.0, 1000.0);
    /// let expected = [
    ///      0.73306423,  0.0,        0.0,       0.0,      // column 1
    ///      0.0,         1.3032253,  0.0,       0.0,      // column 2
    ///      0.0,         0.0,       -1.002002, -2.002002, // column 3
    ///      0.0,         0.0,        0.0,       0.0       // column 4
    /// ];
    ///
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn perspective(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Matrix4 {
        let radians: f32 = (fov / 2.0).to_radians();
        let sine = radians.sin();
        let cotangent = radians.cos() / sine;
        let depth = far - near;

        // setup the projection matrix
        let mut mat = Matrix4::new();
        mat.set_m11(cotangent / aspect_ratio);
        mat.set_m22(cotangent);
        mat.set_m33(-(far + near) / depth);
        mat.set_m43(-1.0);
        mat.set_m43(-2.0 * near * far / depth);
        mat.set_m44(0.0);
        mat
    }

    /// Creates a look-at matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// use vex::Vector3;
    /// use vex::vector3::UP;
    ///
    /// let position = Vector3::make(0.0, 1.0, 1.0);
    /// let target = Vector3::new();
    /// let actual = Matrix4::look_at(position, target, UP);
    /// let expected = [
    ///   1.0, 0.0,         0.0,        0.0, // column 1
    ///   0.0, 0.70710677, -0.70710677, 0.0, // column 2
    ///   0.0, 0.70710677,  0.70710677, 0.0, // column 3
    ///   0.0, 1.0,         1.0,        1.0, // column 4
    /// ];
    ///
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn look_at(position: Vector3, target: Vector3, up: Vector3) -> Matrix4 {
        let mut forward = target - position;
        forward.norm();

        let mut right = Vector3::cross(&forward, &up);
        right.norm();
        let up = Vector3::cross(&right, &forward);

        Matrix4::make(
            right.x, right.y, right.z, 0.0, up.x, up.y, up.z, 0.0, -forward.x, -forward.y,
            -forward.z, 0.0, position.x, position.y, position.z, 1.0,
        )
    }

    /// Creates a translation matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// use vex::Vector3;
    ///
    /// let position = Vector3::make(0.0, 1.0, 1.0);
    /// let target = Vector3::new();
    /// let actual = Matrix4::translate(1.0, 2.0, 3.0);
    /// let expected = [
    ///   1.0, 0.0, 0.0, 0.0, // column 1
    ///   0.0, 1.0, 0.0, 0.0, // column 2
    ///   0.0, 0.0, 1.0, 0.0, // column 3
    ///   1.0, 2.0, 3.0, 1.0, // column 4
    /// ];
    ///
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn translate(x: f32, y: f32, z: f32) -> Matrix4 {
        let mut mat = Matrix4::new();
        mat.set_m14(x);
        mat.set_m24(y);
        mat.set_m34(z);
        mat
    }

    /// Creates an x-rotation matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// use vex::Vector3;
    ///
    /// let actual = Matrix4::rotate_x(1.5707);
    /// let expected = [
    ///     1.0,  0.0,           0.0,           0.0,
    ///     0.0,  0.00009627739, 1.0,           0.0,
    ///     0.0, -1.0,           0.00009627739, 0.0,
    ///     0.0,  0.0,           0.0,           1.0,
    /// ];
    ///
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn rotate_x(angle: f32) -> Matrix4 {
        let mut mat = Matrix4::new();
        mat.set_m22(angle.cos());
        mat.set_m32(angle.sin());
        mat.set_m23(-angle.sin());
        mat.set_m33(angle.cos());
        mat
    }

    /// Creates an y-rotation matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// use vex::Vector3;
    ///
    /// let actual = Matrix4::rotate_y(1.5707);
    /// let expected = [
    ///     0.00009627739,  0.0, 1.0,           0.0, // column 1
    ///     0.0,            1.0, 0.0,           0.0, // column 2
    ///     0.0,            0.0, 0.00009627739, 0.0, // column 3
    ///     0.0,            0.0, 0.0,           1.0, // column 4
    /// ];
    ///
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn rotate_y(angle: f32) -> Matrix4 {
        let mut mat = Matrix4::new();
        mat.set_m11(angle.cos());
        mat.set_m31(-angle.sin());
        mat.set_m31(angle.sin());
        mat.set_m33(angle.cos());
        mat
    }

    /// Creates an z-rotation matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// use vex::Vector3;
    ///
    /// let actual = Matrix4::rotate_z(1.5707);
    /// let expected = [
    ///     0.00009627739, -1.0,           0.0, 0.0, // column 1
    ///     1.0,            0.00009627739, 0.0, 0.0, // column 2
    ///     0.0,            0.0,           1.0, 0.0, // column 3
    ///     0.0,            0.0,           0.0, 1.0, // column 4
    /// ];
    ///
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn rotate_z(angle: f32) -> Matrix4 {
        let mut mat = Matrix4::new();
        mat.set_m11(angle.cos());
        mat.set_m21(-angle.sin());
        mat.set_m12(angle.sin());
        mat.set_m22(angle.cos());
        mat
    }

    /// Creates a scale matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::scale(1.0, 2.0, 3.0);
    /// let expected = [
    ///     1.0, 0.0, 0.0, 0.0, // column 1
    ///     0.0, 2.0, 0.0, 0.0, // column 2
    ///     0.0, 0.0, 3.0, 0.0, // column 3
    ///     0.0, 0.0, 0.0, 1.0, // column 4
    /// ];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn scale(x: f32, y: f32, z: f32) -> Matrix4 {
        let mut mat = Matrix4::new();
        mat.set_m11(x);
        mat.set_m22(y);
        mat.set_m33(z);
        mat
    }

    /// Gets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
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
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m21(), 2.0);
    /// ```
    #[inline]
    pub fn m21(&self) -> f32 {
        self.m[1]
    }

    /// Gets the value for the m31 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m31(), 3.0);
    /// ```
    #[inline]
    pub fn m31(&self) -> f32 {
        self.m[2]
    }

    /// Gets the value for the m41 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m41(), 4.0);
    /// ```
    #[inline]
    pub fn m41(&self) -> f32 {
        self.m[3]
    }

    /// Gets the value for the m12 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m12(), 5.0);
    /// ```
    #[inline]
    pub fn m12(&self) -> f32 {
        self.m[4]
    }

    /// Gets the value for the m22 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m22(), 6.0);
    /// ```
    #[inline]
    pub fn m22(&self) -> f32 {
        self.m[5]
    }

    /// Gets the value for the m32 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m32(), 7.0);
    /// ```
    #[inline]
    pub fn m32(&self) -> f32 {
        self.m[6]
    }

    /// Gets the value for the m42 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m42(), 8.0);
    /// ```
    #[inline]
    pub fn m42(&self) -> f32 {
        self.m[7]
    }

    /// Gets the value for the m13 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m13(), 9.0);
    /// ```
    #[inline]
    pub fn m13(&self) -> f32 {
        self.m[8]
    }

    /// Gets the value for the m23 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m23(), 10.0);
    /// ```
    #[inline]
    pub fn m23(&self) -> f32 {
        self.m[9]
    }

    /// Gets the value for the m33 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m33(), 11.0);
    /// ```
    #[inline]
    pub fn m33(&self) -> f32 {
        self.m[10]
    }

    /// Gets the value for the m43 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m43(), 12.0);
    /// ```
    #[inline]
    pub fn m43(&self) -> f32 {
        self.m[11]
    }

    /// Gets the value for the m14 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m14(), 13.0);
    /// ```
    #[inline]
    pub fn m14(&self) -> f32 {
        self.m[12]
    }

    /// Gets the value for the m24 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m24(), 14.0);
    /// ```
    #[inline]
    pub fn m24(&self) -> f32 {
        self.m[13]
    }

    /// Gets the value for the m34 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m34(), 15.0);
    /// ```
    #[inline]
    pub fn m34(&self) -> f32 {
        self.m[14]
    }

    /// Gets the value for the m44 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual.m44(), 16.0);
    /// ```
    #[inline]
    pub fn m44(&self) -> f32 {
        self.m[15]
    }

    /// Sets the value for the m11 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m11(1.0);
    /// let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
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
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m21(1.0);
    /// let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m21(&mut self, v: f32) {
        self.m[1] = v;
    }

    /// Sets the value for the m31 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m31(1.0);
    /// let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m31(&mut self, v: f32) {
        self.m[2] = v;
    }

    /// Sets the value for the m41 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m41(1.0);
    /// let expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m41(&mut self, v: f32) {
        self.m[3] = v;
    }

    /// Sets the value for the m12 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m12(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m12(&mut self, v: f32) {
        self.m[4] = v;
    }

    /// Sets the value for the m22 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m22(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m22(&mut self, v: f32) {
        self.m[5] = v;
    }

    /// Sets the value for the m32 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m32(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m32(&mut self, v: f32) {
        self.m[6] = v;
    }

    /// Sets the value for the m42 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m42(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m42(&mut self, v: f32) {
        self.m[7] = v;
    }

    /// Sets the value for the m13 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m13(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m13(&mut self, v: f32) {
        self.m[8] = v;
    }

    /// Sets the value for the m23 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m23(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m23(&mut self, v: f32) {
        self.m[9] = v;
    }

    /// Sets the value for the m33 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m33(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m33(&mut self, v: f32) {
        self.m[10] = v;
    }

    /// Sets the value for the m43 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m43(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m43(&mut self, v: f32) {
        self.m[11] = v;
    }

    /// Sets the value for the m14 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m14(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m14(&mut self, v: f32) {
        self.m[12] = v;
    }

    /// Sets the value for the m24 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m24(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m24(&mut self, v: f32) {
        self.m[13] = v;
    }

    /// Sets the value for the m34 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m34(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m34(&mut self, v: f32) {
        self.m[14] = v;
    }

    /// Sets the value for the m44 element
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// actual.set_m44(1.0);
    /// let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    /// assert_eq!(actual.m, expected);
    /// ```
    #[inline]
    pub fn set_m44(&mut self, v: f32) {
        self.m[15] = v;
    }

    /// Sets the internal contents of the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::new();
    /// actual.set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// let expected = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn set(
        &mut self,
        m11: f32,
        m21: f32,
        m31: f32,
        m41: f32,
        m12: f32,
        m22: f32,
        m32: f32,
        m42: f32,
        m13: f32,
        m23: f32,
        m33: f32,
        m43: f32,
        m14: f32,
        m24: f32,
        m34: f32,
        m44: f32,
    ) {
        self.set_m11(m11);
        self.set_m21(m21);
        self.set_m31(m31);
        self.set_m41(m41);
        self.set_m12(m12);
        self.set_m22(m22);
        self.set_m32(m32);
        self.set_m42(m42);
        self.set_m13(m13);
        self.set_m23(m23);
        self.set_m33(m33);
        self.set_m43(m43);
        self.set_m14(m14);
        self.set_m24(m24);
        self.set_m34(m34);
        self.set_m44(m44);
    }

    /// Transposes the matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// actual.transpose();
    /// let expected = Matrix4::make(1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn transpose(&mut self) {
        let mut m = self.m;

        let temp = m[1];
        m[1] = m[4];
        m[4] = temp;
        let temp = m[2];
        m[2] = m[8];
        m[8] = temp;
        let temp = m[6];
        m[6] = m[9];
        m[9] = temp;
        let temp = m[7];
        m[7] = m[13];
        m[13] = temp;
        let temp = m[11];
        m[11] = m[14];
        m[14] = temp;
        let temp = m[3];
        m[3] = m[12];
        m[12] = temp;
        self.m = m;
    }

    /// Find the matrix's determinant
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0).determinant();
    /// assert_eq!(actual, 0.0);
    /// ```
    #[inline]
    pub fn determinant(&self) -> f32 {
        let a = Matrix3::make(
            self.m22(),
            self.m23(),
            self.m24(),
            self.m32(),
            self.m33(),
            self.m34(),
            self.m42(),
            self.m43(),
            self.m44(),
        )
        .determinant()
            * self.m11();

        let b = Matrix3::make(
            self.m21(),
            self.m23(),
            self.m24(),
            self.m31(),
            self.m33(),
            self.m34(),
            self.m41(),
            self.m43(),
            self.m44(),
        )
        .determinant()
            * self.m12();

        let c = Matrix3::make(
            self.m21(),
            self.m22(),
            self.m24(),
            self.m31(),
            self.m32(),
            self.m34(),
            self.m41(),
            self.m42(),
            self.m44(),
        )
        .determinant()
            * self.m13();

        let d = Matrix3::make(
            self.m21(),
            self.m22(),
            self.m23(),
            self.m31(),
            self.m32(),
            self.m33(),
            self.m41(),
            self.m42(),
            self.m43(),
        )
        .determinant()
            * self.m14();

        a - b + c - d
    }

    /// Inverses the matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 4.0);
    /// actual.inverse();
    /// let expected = Matrix4::make(-2.0, 1.0, -8.0, 3.0, -0.5, 0.5, -1.0, 0.5, 1.0, 0.0, 2.0, -1.0, 0.5, -0.5, 2.0, -0.5);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn inverse(&mut self) -> bool {
        let det = self.determinant();
        if det == 0.0 {
            return false;
        }

        let inv_det = 1.0 / det;

        // process the first column
        let pre_m11 = self.m22() * self.m33() * self.m44()
            - self.m22() * self.m43() * self.m34()
            - self.m23() * self.m32() * self.m44()
            + self.m23() * self.m42() * self.m34()
            + self.m24() * self.m32() * self.m43()
            - self.m24() * self.m42() * self.m33();
        let pre_m21 = -self.m21() * self.m33() * self.m44()
            + self.m21() * self.m43() * self.m34()
            + self.m23() * self.m31() * self.m44()
            - self.m23() * self.m41() * self.m34()
            - self.m24() * self.m31() * self.m43()
            + self.m24() * self.m41() * self.m33();
        let pre_m31 = self.m21() * self.m32() * self.m44()
            - self.m21() * self.m42() * self.m34()
            - self.m22() * self.m31() * self.m44()
            + self.m22() * self.m41() * self.m34()
            + self.m24() * self.m31() * self.m42()
            - self.m24() * self.m41() * self.m32();
        let pre_m41 = -self.m21() * self.m32() * self.m43()
            + self.m21() * self.m42() * self.m33()
            + self.m22() * self.m31() * self.m43()
            - self.m22() * self.m41() * self.m33()
            - self.m23() * self.m31() * self.m42()
            + self.m23() * self.m41() * self.m32();

        // process the second column
        let pre_m12 = -self.m12() * self.m33() * self.m44()
            + self.m12() * self.m43() * self.m34()
            + self.m13() * self.m32() * self.m44()
            - self.m13() * self.m42() * self.m34()
            - self.m14() * self.m32() * self.m43()
            + self.m14() * self.m42() * self.m33();
        let pre_m22 = self.m11() * self.m33() * self.m44()
            - self.m11() * self.m43() * self.m34()
            - self.m13() * self.m31() * self.m44()
            + self.m13() * self.m41() * self.m34()
            + self.m14() * self.m31() * self.m43()
            - self.m14() * self.m41() * self.m33();
        let pre_m32 = -self.m11() * self.m32() * self.m44()
            + self.m11() * self.m42() * self.m34()
            + self.m12() * self.m31() * self.m44()
            - self.m12() * self.m41() * self.m34()
            - self.m14() * self.m31() * self.m42()
            + self.m14() * self.m41() * self.m32();
        let pre_m42 = self.m11() * self.m32() * self.m43()
            - self.m11() * self.m42() * self.m33()
            - self.m12() * self.m31() * self.m43()
            + self.m12() * self.m41() * self.m33()
            + self.m13() * self.m31() * self.m42()
            - self.m13() * self.m41() * self.m32();

        // process the third column
        let pre_m13 = self.m12() * self.m23() * self.m44()
            - self.m12() * self.m43() * self.m24()
            - self.m13() * self.m22() * self.m44()
            + self.m13() * self.m42() * self.m24()
            + self.m14() * self.m22() * self.m43()
            - self.m14() * self.m42() * self.m23();
        let pre_m23 = -self.m11() * self.m23() * self.m44()
            + self.m11() * self.m43() * self.m24()
            + self.m13() * self.m21() * self.m44()
            - self.m13() * self.m41() * self.m24()
            - self.m14() * self.m21() * self.m43()
            + self.m14() * self.m41() * self.m23();
        let pre_m33 = self.m11() * self.m22() * self.m44()
            - self.m11() * self.m42() * self.m24()
            - self.m12() * self.m21() * self.m44()
            + self.m12() * self.m41() * self.m24()
            + self.m14() * self.m21() * self.m42()
            - self.m14() * self.m41() * self.m22();
        let pre_m43 = -self.m11() * self.m22() * self.m43()
            + self.m11() * self.m42() * self.m23()
            + self.m12() * self.m21() * self.m43()
            - self.m12() * self.m41() * self.m23()
            - self.m13() * self.m21() * self.m42()
            + self.m13() * self.m41() * self.m22();

        // process the fourth column
        let pre_m14 = -self.m12() * self.m23() * self.m34()
            + self.m12() * self.m33() * self.m24()
            + self.m13() * self.m22() * self.m34()
            - self.m13() * self.m32() * self.m24()
            - self.m14() * self.m22() * self.m33()
            + self.m14() * self.m32() * self.m23();
        let pre_m24 = self.m11() * self.m23() * self.m34()
            - self.m11() * self.m33() * self.m24()
            - self.m13() * self.m21() * self.m34()
            + self.m13() * self.m31() * self.m24()
            + self.m14() * self.m21() * self.m33()
            - self.m14() * self.m31() * self.m23();
        let pre_m34 = -self.m11() * self.m22() * self.m34()
            + self.m11() * self.m32() * self.m24()
            + self.m12() * self.m21() * self.m34()
            - self.m12() * self.m31() * self.m24()
            - self.m14() * self.m21() * self.m32()
            + self.m14() * self.m31() * self.m22();
        let pre_m44 = self.m11() * self.m22() * self.m33()
            - self.m11() * self.m32() * self.m23()
            - self.m12() * self.m21() * self.m33()
            + self.m12() * self.m31() * self.m23()
            + self.m13() * self.m21() * self.m32()
            - self.m13() * self.m31() * self.m22();

        // set the values
        self.set_m11(pre_m11 * inv_det);
        self.set_m21(pre_m21 * inv_det);
        self.set_m31(pre_m31 * inv_det);
        self.set_m41(pre_m41 * inv_det);
        self.set_m12(pre_m12 * inv_det);
        self.set_m22(pre_m22 * inv_det);
        self.set_m32(pre_m32 * inv_det);
        self.set_m42(pre_m42 * inv_det);
        self.set_m13(pre_m13 * inv_det);
        self.set_m23(pre_m23 * inv_det);
        self.set_m33(pre_m33 * inv_det);
        self.set_m43(pre_m43 * inv_det);
        self.set_m14(pre_m14 * inv_det);
        self.set_m24(pre_m24 * inv_det);
        self.set_m34(pre_m34 * inv_det);
        self.set_m44(pre_m44 * inv_det);
        true
    }

    /// Determine whether or not all elements of the matrix are valid
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// assert!(actual.is_valid());
    /// ```
    #[inline]
    pub fn is_valid(&self) -> bool {
        for i in 0..16 {
            if !common::is_valid(self.m[i]) {
                return false;
            }
        }

        true
    }
}

impl Neg for Matrix4 {
    type Output = Matrix4;

    /// Negates the matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = -Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// let expected = Matrix4::make(-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn neg(self) -> Matrix4 {
        let mut m = [0.0; 16];

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                m[i] = -*elem;
            }
        }

        Matrix4 { m }
    }
}

impl Add<f32> for Matrix4 {
    type Output = Matrix4;

    /// Find the resulting matrix by adding a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0) + 1.0;
    /// let expected = Matrix4::make(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: f32) -> Matrix4 {
        let mut mat = Matrix4::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem + _rhs;
            }
        }

        mat
    }
}

impl Add<Matrix4> for Matrix4 {
    type Output = Matrix4;

    /// Add two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let a = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// let b = Matrix4::make(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let actual = a + b;
    /// let expected = Matrix4::make(17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: Matrix4) -> Matrix4 {
        let mut mat = Matrix4::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem + _rhs.m[i];
            }
        }

        mat
    }
}

impl AddAssign<f32> for Matrix4 {
    /// Increment a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// actual += 10.0;
    /// let expected = Matrix4::make(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0);
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

impl AddAssign<Matrix4> for Matrix4 {
    /// Increment a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// actual += Matrix4::make(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let expected = Matrix4::make(17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: Matrix4) {
        unsafe {
            for (i, elem) in self.m.iter_mut().enumerate() {
                *elem += _rhs.m[i];
            }
        }
    }
}

impl Sub<f32> for Matrix4 {
    type Output = Matrix4;

    /// Find the resulting matrix by subtracting a scalar from a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0) - 17.0;
    /// let expected = Matrix4::make(-16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: f32) -> Matrix4 {
        let mut mat = Matrix4::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem - _rhs;
            }
        }

        mat
    }
}

impl Sub<Matrix4> for Matrix4 {
    type Output = Matrix4;

    /// Subtract two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let a = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// let b = Matrix4::make(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let actual = a - b;
    /// let expected = Matrix4::make(-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: Matrix4) -> Matrix4 {
        let mut mat = Matrix4::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem - _rhs.m[i];
            }
        }

        mat
    }
}

impl SubAssign<f32> for Matrix4 {
    /// Decrement a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// actual -= 1.0;
    /// let expected = Matrix4::make(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0);
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

impl SubAssign<Matrix4> for Matrix4 {
    /// Decrement a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// actual -= Matrix4::make(0.0, 2.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 9.0, 10.0, 10.0, 12.0, 13.0, 14.0, 15.0, 15.0);
    /// assert_eq!(actual, Matrix4::new());
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: Matrix4) {
        unsafe {
            for (i, elem) in self.m.iter_mut().enumerate() {
                *elem -= _rhs.m[i];
            }
        }
    }
}

impl Mul<f32> for Matrix4 {
    type Output = Matrix4;

    /// Find the resulting matrix by multiplying a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0) * 2.0;
    /// let expected = Matrix4::make(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: f32) -> Matrix4 {
        let mut mat = Matrix4::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem * _rhs;
            }
        }

        mat
    }
}

impl Mul<Matrix4> for Matrix4 {
    type Output = Matrix4;

    /// Multiply two matrices
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let a = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// let b = Matrix4::make(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let actual = a * b;
    /// let expected = Matrix4::make(
    ///   386.0, 444.0, 502.0, 560.0,
    ///   274.0, 316.0, 358.0, 400.0,
    ///   162.0, 188.0, 214.0, 240.0,
    ///    50.0,  60.0,  70.0,  80.0,
    /// );
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: Matrix4) -> Matrix4 {
        let m11 = self.m11() * _rhs.m11()
            + self.m12() * _rhs.m21()
            + self.m13() * _rhs.m31()
            + self.m14() * _rhs.m41();
        let m21 = self.m21() * _rhs.m11()
            + self.m22() * _rhs.m21()
            + self.m23() * _rhs.m31()
            + self.m24() * _rhs.m41();
        let m31 = self.m31() * _rhs.m11()
            + self.m32() * _rhs.m21()
            + self.m33() * _rhs.m31()
            + self.m34() * _rhs.m41();
        let m41 = self.m41() * _rhs.m11()
            + self.m42() * _rhs.m21()
            + self.m43() * _rhs.m31()
            + self.m44() * _rhs.m41();

        let m12 = self.m11() * _rhs.m12()
            + self.m12() * _rhs.m22()
            + self.m13() * _rhs.m32()
            + self.m14() * _rhs.m42();
        let m22 = self.m21() * _rhs.m12()
            + self.m22() * _rhs.m22()
            + self.m23() * _rhs.m32()
            + self.m24() * _rhs.m42();
        let m32 = self.m31() * _rhs.m12()
            + self.m32() * _rhs.m22()
            + self.m33() * _rhs.m32()
            + self.m34() * _rhs.m42();
        let m42 = self.m41() * _rhs.m12()
            + self.m42() * _rhs.m22()
            + self.m43() * _rhs.m32()
            + self.m44() * _rhs.m42();

        let m13 = self.m11() * _rhs.m13()
            + self.m12() * _rhs.m23()
            + self.m13() * _rhs.m33()
            + self.m14() * _rhs.m43();
        let m23 = self.m21() * _rhs.m13()
            + self.m22() * _rhs.m23()
            + self.m23() * _rhs.m33()
            + self.m24() * _rhs.m43();
        let m33 = self.m31() * _rhs.m13()
            + self.m32() * _rhs.m23()
            + self.m33() * _rhs.m33()
            + self.m34() * _rhs.m43();
        let m43 = self.m41() * _rhs.m13()
            + self.m42() * _rhs.m23()
            + self.m43() * _rhs.m33()
            + self.m44() * _rhs.m43();

        let m14 = self.m11() * _rhs.m14()
            + self.m12() * _rhs.m24()
            + self.m13() * _rhs.m34()
            + self.m14() * _rhs.m44();
        let m24 = self.m21() * _rhs.m14()
            + self.m22() * _rhs.m24()
            + self.m23() * _rhs.m34()
            + self.m24() * _rhs.m44();
        let m34 = self.m31() * _rhs.m14()
            + self.m32() * _rhs.m24()
            + self.m33() * _rhs.m34()
            + self.m34() * _rhs.m44();
        let m44 = self.m41() * _rhs.m14()
            + self.m42() * _rhs.m24()
            + self.m43() * _rhs.m34()
            + self.m44() * _rhs.m44();

        Matrix4::make(
            m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44,
        )
    }
}

impl MulAssign<f32> for Matrix4 {
    /// Multiply a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// actual *= 2.0;
    /// let expected = Matrix4::make(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0);
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

impl MulAssign<Matrix4> for Matrix4 {
    /// Multiply a matrix by another matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// actual *= Matrix4::make(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    /// let expected = Matrix4::make(
    ///   386.0, 444.0, 502.0, 560.0,
    ///   274.0, 316.0, 358.0, 400.0,
    ///   162.0, 188.0, 214.0, 240.0,
    ///    50.0,  60.0,  70.0,  80.0,
    /// );
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: Matrix4) {
        let res = *self * _rhs;
        self.m = res.m;
    }
}

impl Div<f32> for Matrix4 {
    type Output = Matrix4;

    /// Find the resulting matrix by dividing a scalar to a matrix's elements
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0) / 2.0;
    /// let expected = Matrix4::make(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div(self, _rhs: f32) -> Matrix4 {
        let mut mat = Matrix4::new();

        unsafe {
            for (i, elem) in self.m.iter().enumerate() {
                mat.m[i] = *elem / _rhs;
            }
        }

        mat
    }
}

impl DivAssign<f32> for Matrix4 {
    /// Divide a matrix by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// let mut actual = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// actual /= 2.0;
    /// let expected = Matrix4::make(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0);
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

impl cmp::PartialEq for Matrix4 {
    /// Determines if two matrices' elements are equivalent
    ///
    /// # Examples
    /// ```
    /// use vex::Matrix4;
    /// assert!(Matrix4::new() == Matrix4::new());
    /// ```
    #[inline]
    fn eq(&self, _rhs: &Matrix4) -> bool {
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

impl Display for Matrix4 {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "[\n  {}, {}, {}, {}\n  {}, {}, {}, {}\n  {}, {}, {}, {}\n  {}, {}, {}, {}\n]",
            self.m11(),
            self.m12(),
            self.m13(),
            self.m14(),
            self.m21(),
            self.m22(),
            self.m23(),
            self.m24(),
            self.m31(),
            self.m32(),
            self.m33(),
            self.m34(),
            self.m41(),
            self.m42(),
            self.m43(),
            self.m44(),
        )
    }
}

impl common::TransformPoint<Vector3> for Matrix4 {
    /// Find the resulting vector given a vector and matrix
    ///
    /// # Examples
    /// ```
    /// use vex::common::TransformPoint;
    /// use vex::Matrix4;
    /// use vex::Vector3;
    /// let m = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// let v = Vector3::make(1.0, 2.0, 3.0);
    /// let actual = m.transform_point(&v);
    /// let expected = Vector3::make(51.0, 58.0, 65.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn transform_point(&self, point: &Vector3) -> Vector3 {
        Vector3::make(
            self.m11() * point.x + self.m12() * point.y + self.m13() * point.z + self.m14(),
            self.m21() * point.x + self.m22() * point.y + self.m23() * point.z + self.m24(),
            self.m31() * point.x + self.m32() * point.y + self.m33() * point.z + self.m34(),
        )
    }
}

impl common::TransformPoint<Vector4> for Matrix4 {
    /// Find the resulting vector given a vector and matrix
    ///
    /// # Examples
    /// ```
    /// use vex::common::TransformPoint;
    /// use vex::Matrix4;
    /// use vex::Vector4;
    /// let m = Matrix4::make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    /// let v = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// let actual = m.transform_point(&v);
    /// let expected = Vector4::make(90.0, 100.0, 110.0, 120.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn transform_point(&self, point: &Vector4) -> Vector4 {
        Vector4::make(
            self.m11() * point.x
                + self.m12() * point.y
                + self.m13() * point.z
                + self.m14() * point.w,
            self.m21() * point.x
                + self.m22() * point.y
                + self.m23() * point.z
                + self.m24() * point.w,
            self.m31() * point.x
                + self.m32() * point.y
                + self.m33() * point.z
                + self.m34() * point.w,
            self.m41() * point.x
                + self.m42() * point.y
                + self.m43() * point.z
                + self.m44() * point.w,
        )
    }
}
