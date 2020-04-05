use crate::common;
use crate::vector3::Vector3;

use std::cmp;
use std::convert::From;
use std::f32::EPSILON;
use std::fmt;
use std::ops;

pub const ZERO: Vector4 = Vector4 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
    w: 0.0,
};

pub const ONE: Vector4 = Vector4 {
    x: 1.0,
    y: 1.0,
    z: 1.0,
    w: 1.0,
};

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct Vector4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vector4 {
    /// Creates a vector <0.0, 0.0, 0.0, 0.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::new();
    /// let expected = Vector4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn new() -> Vector4 {
        ZERO
    }

    /// Creates a vector from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// let expected = Vector4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn make(x: f32, y: f32, z: f32, w: f32) -> Vector4 {
        Vector4 { x, y, z, w }
    }

    /// Find the dot product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let a = Vector4::make(1.0, 0.0, 0.0, 0.0);
    /// let b = Vector4::make(0.0, 0.0, 1.0, 0.0);
    /// let actual = Vector4::dot(&a, &b);
    /// let expected = 0.0;
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn dot(a: &Vector4, b: &Vector4) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    }

    /// Find the minimum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let a = Vector4::make(1.0, 4.0, 5.0, 7.0);
    /// let b = Vector4::make(2.0, 3.0, 6.0, 8.0);
    /// let actual = Vector4::min(&a, &b);
    /// let expected = Vector4::make(1.0, 3.0, 5.0, 7.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn min(a: &Vector4, b: &Vector4) -> Vector4 {
        Vector4::make(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z), a.w.min(b.w))
    }

    /// Find the maximum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let a = Vector4::make(1.0, 4.0, 5.0, 7.0);
    /// let b = Vector4::make(2.0, 3.0, 6.0, 8.0);
    /// let actual = Vector4::max(&a, &b);
    /// let expected = Vector4::make(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn max(a: &Vector4, b: &Vector4) -> Vector4 {
        Vector4::make(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z), a.w.max(b.w))
    }

    /// Find the clamped (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let a = Vector4::make(1.0, 3.0, 5.0, 7.0);
    /// let b = Vector4::make(2.0, 4.0, 6.0, 8.0);
    /// let mut actual = Vector4::make(0.0, 5.0, 10.0, 20.0);
    /// actual.clamp(&a, &b);
    /// let expected = Vector4::make(1.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn clamp(&mut self, a: &Vector4, b: &Vector4) {
        let low = Self::min(a, b);
        let high = Self::max(a, b);
        let result = Self::max(&low, &Self::min(self, &high));
        self.set(result.x, result.y, result.z, result.w);
    }

    /// Set the components of a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::new();
    /// actual.set(1.0, 2.0, 3.0, 4.0);
    /// let expected = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32, w: f32) {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
    }

    /// Get the magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::make(1.0, 2.0, 3.0, 4.0).mag();
    /// let expected = 5.47722557505;
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn mag(&self) -> f32 {
        self.mag_sq().sqrt()
    }

    /// Get the squared magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::make(1.0, 2.0, 3.0, 4.0).mag_sq();
    /// let expected = 30.0;
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn mag_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Normalize the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// actual.norm();
    /// let expected = Vector4::make(0.18257418, 0.36514837, 0.5477225, 0.73029673);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn norm(&mut self) -> f32 {
        let length = self.mag();
        if length > EPSILON {
            self.x /= length;
            self.y /= length;
            self.z /= length;
            self.w /= length;
            length
        } else {
            0.0
        }
    }

    /// Set the components of a vector to their absolute values
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(-1.0, -2.0, -3.0, -4.0);
    /// actual.abs();
    /// let expected = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn abs(&mut self) {
        self.x = self.x.abs();
        self.y = self.y.abs();
        self.z = self.z.abs();
        self.w = self.w.abs();
    }

    /// Determine whether or not all components of the vector are valid
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// assert!(actual.is_valid());
    /// ```
    #[inline]
    pub fn is_valid(&self) -> bool {
        for i in 0..4 {
            if !common::is_valid(self[i]) {
                return false;
            }
        }

        true
    }
}

impl From<Vector3> for Vector4 {
    /// Creates a Vector4 from the components of a Vector3
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// use vex::Vector4;
    /// let input = Vector3::make(1.0, 2.0, 3.0);
    /// let actual = Vector4::from(input);
    /// let expected = Vector4 { x: 1.0, y: 2.0, z: 3.0, w: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn from(item: Vector3) -> Vector4 {
        Vector4 {
            x: item.x,
            y: item.y,
            z: item.z,
            w: 0.0,
        }
    }
}

impl ops::Index<u32> for Vector4 {
    type Output = f32;

    /// Looks up a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut v = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v[1], 2.0);
    /// assert_eq!(v[2], 3.0);
    /// assert_eq!(v[3], 4.0);
    /// ```
    #[inline]
    fn index(&self, index: u32) -> &f32 {
        unsafe {
            match index {
                0 => &self.x,
                1 => &self.y,
                2 => &self.z,
                3 => &self.w,
                _ => panic!("Invalid index for Vector4: {}", index),
            }
        }
    }
}

impl ops::IndexMut<u32> for Vector4 {
    /// Mutate a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut v = Vector4::new();
    /// v[0] = 4.0;
    /// v[1] = 5.0;
    /// v[2] = 6.0;
    /// v[3] = 7.0;
    /// assert_eq!(v[0], 4.0);
    /// assert_eq!(v[1], 5.0);
    /// assert_eq!(v[2], 6.0);
    /// assert_eq!(v[3], 7.0);
    /// ```
    #[inline]
    fn index_mut<'a>(&'a mut self, index: u32) -> &'a mut f32 {
        unsafe {
            match index {
                0 => &mut self.x,
                1 => &mut self.y,
                2 => &mut self.z,
                3 => &mut self.w,
                _ => panic!("Invalid index for Vector4: {}", index),
            }
        }
    }
}

impl ops::Neg for Vector4 {
    type Output = Vector4;

    /// Negates all components in a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = -Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// let expected = Vector4::make(-1.0, -2.0, -3.0, -4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn neg(self) -> Vector4 {
        Vector4::make(-self.x, -self.y, -self.z, -self.w)
    }
}

impl ops::Add<f32> for Vector4 {
    type Output = Vector4;

    /// Find the resulting vector by adding a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::make(1.0, 2.0, 3.0, 4.0) + 1.0;
    /// let expected = Vector4::make(2.0, 3.0, 4.0, 5.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: f32) -> Vector4 {
        Vector4::make(self.x + _rhs, self.y + _rhs, self.z + _rhs, self.w + _rhs)
    }
}

impl ops::Add<Vector4> for Vector4 {
    type Output = Vector4;

    /// Add two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let a = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// let b = Vector4::make(5.0, 6.0, 7.0, 8.0);
    /// let actual = a + b;
    /// let expected = Vector4::make(6.0, 8.0, 10.0, 12.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: Vector4) -> Vector4 {
        Vector4::make(
            self.x + _rhs.x,
            self.y + _rhs.y,
            self.z + _rhs.z,
            self.w + _rhs.w,
        )
    }
}

impl ops::AddAssign<f32> for Vector4 {
    /// Increment a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// actual += 10.0;
    /// let expected = Vector4::make(11.0, 12.0, 13.0, 14.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: f32) {
        self.x += _rhs;
        self.y += _rhs;
        self.z += _rhs;
        self.w += _rhs;
    }
}

impl ops::AddAssign<Vector4> for Vector4 {
    /// Increment a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// actual += Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// let expected = Vector4::make(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: Vector4) {
        self.x += _rhs.x;
        self.y += _rhs.y;
        self.z += _rhs.z;
        self.w += _rhs.w;
    }
}

impl ops::Sub<f32> for Vector4 {
    type Output = Vector4;

    /// Find the resulting vector by subtracting a scalar from a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::make(1.0, 2.0, 3.0, 4.0) - 10.0;
    /// let expected = Vector4::make(-9.0, -8.0, -7.0, -6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: f32) -> Vector4 {
        Vector4::make(self.x - _rhs, self.y - _rhs, self.z - _rhs, self.w - _rhs)
    }
}

impl ops::Sub<Vector4> for Vector4 {
    type Output = Vector4;

    /// Subtract two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let a = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// let b = Vector4::make(5.0, 4.0, 3.0, 2.0);
    /// let actual = a - b;
    /// let expected = Vector4::make(-4.0, -2.0, 0.0, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: Vector4) -> Vector4 {
        Vector4::make(
            self.x - _rhs.x,
            self.y - _rhs.y,
            self.z - _rhs.z,
            self.w - _rhs.w,
        )
    }
}

impl ops::SubAssign<f32> for Vector4 {
    /// Decrement a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// actual -= 1.0;
    /// let expected = Vector4::make(0.0, 1.0, 2.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: f32) {
        self.x -= _rhs;
        self.y -= _rhs;
        self.z -= _rhs;
        self.w -= _rhs;
    }
}

impl ops::SubAssign<Vector4> for Vector4 {
    /// Decrement a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// actual -= Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, Vector4::new());
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: Vector4) {
        self.x -= _rhs.x;
        self.y -= _rhs.y;
        self.z -= _rhs.z;
        self.w -= _rhs.w;
    }
}

impl ops::Mul<f32> for Vector4 {
    type Output = Vector4;

    /// Find the resulting vector by multiplying a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::make(1.0, 2.0, 3.0, 4.0) * 2.0;
    /// let expected = Vector4::make(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: f32) -> Vector4 {
        Vector4::make(self.x * _rhs, self.y * _rhs, self.z * _rhs, self.w * _rhs)
    }
}

impl ops::Mul<Vector4> for Vector4 {
    type Output = Vector4;

    /// Multiply two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let a = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// let b = Vector4::make(5.0, 6.0, 7.0, 8.0);
    /// let actual = a * b;
    /// let expected = Vector4::make(5.0, 12.0, 21.0, 32.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: Vector4) -> Vector4 {
        Vector4::make(
            self.x * _rhs.x,
            self.y * _rhs.y,
            self.z * _rhs.z,
            self.w * _rhs.w,
        )
    }
}

impl ops::MulAssign<f32> for Vector4 {
    /// Multiply a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// actual *= 2.0;
    /// let expected = Vector4::make(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: f32) {
        self.x *= _rhs;
        self.y *= _rhs;
        self.z *= _rhs;
        self.w *= _rhs;
    }
}

impl ops::MulAssign<Vector4> for Vector4 {
    /// Multiply a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// actual *= Vector4::make(2.0, 3.0, 6.0, 8.0);
    /// let expected = Vector4::make(2.0, 6.0, 18.0, 32.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: Vector4) {
        self.x *= _rhs.x;
        self.y *= _rhs.y;
        self.z *= _rhs.z;
        self.w *= _rhs.w;
    }
}

impl ops::Div<f32> for Vector4 {
    type Output = Vector4;

    /// Find the resulting vector by dividing a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let actual = Vector4::make(1.0, 2.0, 3.0, 4.0) / 2.0;
    /// let expected = Vector4::make(0.5, 1.0, 1.5, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div(self, _rhs: f32) -> Vector4 {
        Vector4::make(self.x / _rhs, self.y / _rhs, self.z / _rhs, self.w / _rhs)
    }
}

impl ops::Div<Vector4> for Vector4 {
    type Output = Vector4;

    /// Divide two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let a = Vector4::make(2.0, 4.0, 6.0, 8.0);
    /// let b = Vector4::make(1.0, 4.0, 12.0, 32.0);
    /// let actual = a / b;
    /// let expected = Vector4::make(2.0, 1.0, 0.5, 0.25);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div(self, _rhs: Vector4) -> Vector4 {
        Vector4::make(
            self.x / _rhs.x,
            self.y / _rhs.y,
            self.z / _rhs.z,
            self.w / _rhs.w,
        )
    }
}

impl ops::DivAssign<f32> for Vector4 {
    /// Divide a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// actual /= 2.0;
    /// let expected = Vector4::make(0.5, 1.0, 1.5, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div_assign(&mut self, _rhs: f32) {
        self.x /= _rhs;
        self.y /= _rhs;
        self.z /= _rhs;
        self.w /= _rhs;
    }
}

impl ops::DivAssign<Vector4> for Vector4 {
    /// Divide a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// let mut actual = Vector4::make(2.0, 4.0, 6.0, 8.0);
    /// actual /= Vector4::make(1.0, 4.0, 12.0, 32.0);
    /// let expected = Vector4::make(2.0, 1.0, 0.5, 0.25);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div_assign(&mut self, _rhs: Vector4) {
        self.x /= _rhs.x;
        self.y /= _rhs.y;
        self.z /= _rhs.z;
        self.w /= _rhs.w;
    }
}

impl cmp::PartialEq for Vector4 {
    /// Determines if two vectors' components are equivalent
    ///
    /// # Examples
    /// ```
    /// use vex::Vector4;
    /// assert!(Vector4::new() == Vector4::new());
    /// ```
    #[inline]
    fn eq(&self, _rhs: &Vector4) -> bool {
        for i in 0..4 {
            if self[i] != _rhs[i] {
                return false;
            }
        }

        true
    }
}

impl fmt::Display for Vector4 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe { write!(f, "<{}, {}, {}, {}>", self.x, self.y, self.z, self.w) }
    }
}
