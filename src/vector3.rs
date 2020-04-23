use crate::common;
use crate::vector2::Vector2;
use crate::vector4::Vector4;

use std::cmp;
use std::convert::From;
use std::f32::EPSILON;
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

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3 {
    /// Creates a vector <0.0, 0.0, 0.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::new();
    /// let expected = Vector3 { x: 0.0, y: 0.0, z: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn new() -> Vector3 {
        Vector3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Creates a vector <1.0, 1.0, 1.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::one();
    /// let expected = Vector3 { x: 1.0, y: 1.0, z: 1.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn one() -> Vector3 {
        Vector3 { x: 1.0, y: 1.0, z: 1.0 }
    }

    /// Creates a right vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::right();
    /// let expected = Vector3 { x: 1.0, y: 0.0, z: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn right() -> Vector3 {
        Vector3 { x: 1.0, y: 0.0, z: 0.0 }
    }

    /// Creates an up vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;

    /// let actual = Vector3::up();
    /// let expected = Vector3 { x: 0.0, y: 1.0, z: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn up() -> Vector3 {
        Vector3 { x: 0.0, y: 1.0, z: 0.0 }
    }

    /// Creates a forward vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::forward();
    /// let expected = Vector3 { x: 0.0, y: 0.0, z: -1.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn forward() -> Vector3 {
        Vector3 { x: 0.0, y: 0.0, z: -1.0 }
    }

    /// Creates a vector from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::make(1.0, 2.0, 3.0);
    /// let expected = Vector3 { x: 1.0, y: 2.0, z: 3.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn make(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    /// Find the dot product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(1.0, 0.0, 0.0);
    /// let b = Vector3::make(0.0, 0.0, 1.0);
    /// let actual = Vector3::dot(&a, &b);
    /// let expected = 0.0;
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn dot(a: &Vector3, b: &Vector3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    /// Find the cross product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(0.0, 0.0, 1.0);
    /// let b = Vector3::make(1.0, 0.0, 0.0);
    /// let actual = Vector3::cross(&a, &b);
    /// let expected = Vector3::make(0.0, 1.0, 0.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn cross(a: &Vector3, b: &Vector3) -> Vector3 {
        Vector3::make(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    }

    /// Find the minimum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(1.0, 4.0, 5.0);
    /// let b = Vector3::make(2.0, 3.0, 6.0);
    /// let actual = Vector3::min(&a, &b);
    /// let expected = Vector3::make(1.0, 3.0, 5.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn min(a: &Vector3, b: &Vector3) -> Vector3 {
        Vector3::make(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
    }

    /// Find the maximum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(1.0, 4.0, 5.0);
    /// let b = Vector3::make(2.0, 3.0, 6.0);
    /// let actual = Vector3::max(&a, &b);
    /// let expected = Vector3::make(2.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn max(a: &Vector3, b: &Vector3) -> Vector3 {
        Vector3::make(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
    }

    /// Find the clamped (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(1.0, 3.0, 5.0);
    /// let b = Vector3::make(2.0, 4.0, 6.0);
    /// let mut actual = Vector3::make(0.0, 5.0, 10.0);
    /// actual.clamp(&a, &b);
    /// let expected = Vector3::make(1.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn clamp(&mut self, a: &Vector3, b: &Vector3) {
        let low = Self::min(a, b);
        let high = Self::max(a, b);
        let result = Self::max(&low, &Self::min(self, &high));
        self.set(result.x, result.y, result.z);
    }

    /// Set the components of a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::new();
    /// actual.set(1.0, 2.0, 3.0);
    /// let expected = Vector3::make(1.0, 2.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    /// Get the magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::make(1.0, 2.0, 3.0).mag();
    /// let expected = 3.74165738677;
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
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::make(1.0, 2.0, 3.0).mag_sq();
    /// let expected = 14.0;
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn mag_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalize the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 3.0);
    /// actual.norm();
    /// let expected = Vector3::make(0.26726124191, 0.53452248382, 0.8017837);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn norm(&mut self) -> f32 {
        let length = self.mag();
        if length > EPSILON {
            self.x /= length;
            self.y /= length;
            self.z /= length;
            length
        } else {
            0.0
        }
    }

    /// Set the components of a vector to their absolute values
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(-1.0, -2.0, -3.0);
    /// actual.abs();
    /// let expected = Vector3::make(1.0, 2.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn abs(&mut self) {
        self.x = self.x.abs();
        self.y = self.y.abs();
        self.z = self.z.abs();
    }

    /// Determine whether or not all components of the vector are valid
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::make(1.0, 2.0, 3.0);
    /// assert!(actual.is_valid());
    /// ```
    #[inline]
    pub fn is_valid(&self) -> bool {
        for i in 0..3 {
            if !common::is_valid(self[i]) {
                return false;
            }
        }

        true
    }
}

impl From<Vector2> for Vector3 {
    /// Creates a Vector3 from the components of a Vector2
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// use vex::Vector3;
    /// 
    /// let input = Vector2::make(1.0, 2.0);
    /// let actual = Vector3::from(input);
    /// let expected = Vector3 { x: 1.0, y: 2.0, z: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn from(item: Vector2) -> Vector3 {
        Vector3 {
            x: item.x,
            y: item.y,
            z: 0.0,
        }
    }
}

impl From<Vector4> for Vector3 {
    /// Creates a Vector3 from the components of a Vector4
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// use vex::Vector4;
    /// 
    /// let input = Vector4::make(1.0, 2.0, 3.0, 4.0);
    /// let actual = Vector3::from(input);
    /// let expected = Vector3 { x: 1.0, y: 2.0, z: 3.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn from(item: Vector4) -> Vector3 {
        Vector3 {
            x: item.x,
            y: item.y,
            z: item.z,
        }
    }
}

impl Index<u32> for Vector3 {
    type Output = f32;

    /// Looks up a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut v = Vector3::make(1.0, 2.0, 3.0);
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v[1], 2.0);
    /// assert_eq!(v[2], 3.0);
    /// ```
    #[inline]
    fn index(&self, index: u32) -> &f32 {
        unsafe {
            match index {
                0 => &self.x,
                1 => &self.y,
                2 => &self.z,
                _ => panic!("Invalid index for Vector3: {}", index),
            }
        }
    }
}

impl IndexMut<u32> for Vector3 {
    /// Mutate a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut v = Vector3::new();
    /// v[0] = 4.0;
    /// v[1] = 5.0;
    /// v[2] = 6.0;
    /// assert_eq!(v[0], 4.0);
    /// assert_eq!(v[1], 5.0);
    /// assert_eq!(v[2], 6.0);
    /// ```
    #[inline]
    fn index_mut<'a>(&'a mut self, index: u32) -> &'a mut f32 {
        unsafe {
            match index {
                0 => &mut self.x,
                1 => &mut self.y,
                2 => &mut self.z,
                _ => panic!("Invalid index for Vector3: {}", index),
            }
        }
    }
}

impl Neg for Vector3 {
    type Output = Vector3;

    /// Negates all components in a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = -Vector3::make(1.0, 2.0, 3.0);
    /// let expected = Vector3::make(-1.0, -2.0, -3.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn neg(self) -> Vector3 {
        Vector3::make(-self.x, -self.y, -self.z)
    }
}

impl Add<f32> for Vector3 {
    type Output = Vector3;

    /// Find the resulting vector by adding a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::make(1.0, 2.0, 3.0) + 1.0;
    /// let expected = Vector3::make(2.0, 3.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: f32) -> Vector3 {
        Vector3::make(self.x + _rhs, self.y + _rhs, self.z + _rhs)
    }
}

impl Add<Vector3> for Vector3 {
    type Output = Vector3;

    /// Add two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(1.0, 2.0, 3.0);
    /// let b = Vector3::make(4.0, 5.0, 6.0);
    /// let actual = a + b;
    /// let expected = Vector3::make(5.0, 7.0, 9.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: Vector3) -> Vector3 {
        Vector3::make(self.x + _rhs.x, self.y + _rhs.y, self.z + _rhs.z)
    }
}

impl AddAssign<f32> for Vector3 {
    /// Increment a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 3.0);
    /// actual += 10.0;
    /// let expected = Vector3::make(11.0, 12.0, 13.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: f32) {
        self.x += _rhs;
        self.y += _rhs;
        self.z += _rhs;
    }
}

impl AddAssign<Vector3> for Vector3 {
    /// Increment a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 3.0);
    /// actual += Vector3::make(1.0, 2.0, 3.0);
    /// let expected = Vector3::make(2.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: Vector3) {
        self.x += _rhs.x;
        self.y += _rhs.y;
        self.z += _rhs.z;
    }
}

impl Sub<f32> for Vector3 {
    type Output = Vector3;

    /// Find the resulting vector by subtracting a scalar from a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::make(1.0, 2.0, 3.0) - 10.0;
    /// let expected = Vector3::make(-9.0, -8.0, -7.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: f32) -> Vector3 {
        Vector3::make(self.x - _rhs, self.y - _rhs, self.z - _rhs)
    }
}

impl Sub<Vector3> for Vector3 {
    type Output = Vector3;

    /// Subtract two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(1.0, 2.0, 3.0);
    /// let b = Vector3::make(5.0, 4.0, 3.0);
    /// let actual = a - b;
    /// let expected = Vector3::make(-4.0, -2.0, 0.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: Vector3) -> Vector3 {
        Vector3::make(self.x - _rhs.x, self.y - _rhs.y, self.z - _rhs.z)
    }
}

impl SubAssign<f32> for Vector3 {
    /// Decrement a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 3.0);
    /// actual -= 1.0;
    /// let expected = Vector3::make(0.0, 1.0, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: f32) {
        self.x -= _rhs;
        self.y -= _rhs;
        self.z -= _rhs;
    }
}

impl SubAssign<Vector3> for Vector3 {
    /// Decrement a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 3.0);
    /// actual -= Vector3::make(1.0, 2.0, 3.0);
    /// assert_eq!(actual, Vector3::new());
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: Vector3) {
        self.x -= _rhs.x;
        self.y -= _rhs.y;
        self.z -= _rhs.z;
    }
}

impl Mul<f32> for Vector3 {
    type Output = Vector3;

    /// Find the resulting vector by multiplying a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::make(1.0, 2.0, 3.0) * 2.0;
    /// let expected = Vector3::make(2.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: f32) -> Vector3 {
        Vector3::make(self.x * _rhs, self.y * _rhs, self.z * _rhs)
    }
}

impl Mul<Vector3> for Vector3 {
    type Output = Vector3;

    /// Multiply two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(1.0, 2.0, 3.0);
    /// let b = Vector3::make(3.0, 4.0, 5.0);
    /// let actual = a * b;
    /// let expected = Vector3::make(3.0, 8.0, 15.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: Vector3) -> Vector3 {
        Vector3::make(self.x * _rhs.x, self.y * _rhs.y, self.z * _rhs.z)
    }
}

impl MulAssign<f32> for Vector3 {
    /// Multiply a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 3.0);
    /// actual *= 2.0;
    /// let expected = Vector3::make(2.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: f32) {
        self.x *= _rhs;
        self.y *= _rhs;
        self.z *= _rhs;
    }
}

impl MulAssign<Vector3> for Vector3 {
    /// Multiply a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 3.0);
    /// actual *= Vector3::make(2.0, 3.0, 6.0);
    /// let expected = Vector3::make(2.0, 6.0, 18.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: Vector3) {
        self.x *= _rhs.x;
        self.y *= _rhs.y;
        self.z *= _rhs.z;
    }
}

impl Div<f32> for Vector3 {
    type Output = Vector3;

    /// Find the resulting vector by dividing a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let actual = Vector3::make(1.0, 2.0, 3.0) / 2.0;
    /// let expected = Vector3::make(0.5, 1.0, 1.5);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div(self, _rhs: f32) -> Vector3 {
        Vector3::make(self.x / _rhs, self.y / _rhs, self.z / _rhs)
    }
}

impl Div<Vector3> for Vector3 {
    type Output = Vector3;

    /// Divide two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let a = Vector3::make(1.0, 2.0, 4.0);
    /// let b = Vector3::make(2.0, 8.0, 32.0);
    /// let actual = a / b;
    /// let expected = Vector3::make(0.5, 0.25, 0.125);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div(self, _rhs: Vector3) -> Vector3 {
        Vector3::make(self.x / _rhs.x, self.y / _rhs.y, self.z / _rhs.z)
    }
}

impl DivAssign<f32> for Vector3 {
    /// Divide a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 3.0);
    /// actual /= 2.0;
    /// let expected = Vector3::make(0.5, 1.0, 1.5);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div_assign(&mut self, _rhs: f32) {
        self.x /= _rhs;
        self.y /= _rhs;
        self.z /= _rhs;
    }
}

impl DivAssign<Vector3> for Vector3 {
    /// Divide a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// let mut actual = Vector3::make(1.0, 2.0, 4.0);
    /// actual /= Vector3::make(2.0, 8.0, 32.0);
    /// let expected = Vector3::make(0.5, 0.25, 0.125);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div_assign(&mut self, _rhs: Vector3) {
        self.x /= _rhs.x;
        self.y /= _rhs.y;
        self.z /= _rhs.z;
    }
}

impl cmp::PartialEq for Vector3 {
    /// Determines if two vectors' components are equivalent
    ///
    /// # Examples
    /// ```
    /// use vex::Vector3;
    /// 
    /// assert!(Vector3::new() == Vector3::new());
    /// ```
    #[inline]
    fn eq(&self, _rhs: &Vector3) -> bool {
        for i in 0..3 {
            if self[i] != _rhs[i] {
                return false;
            }
        }

        true
    }
}

impl Display for Vector3 {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        unsafe { write!(f, "<{}  {}  {}>", self.x, self.y, self.z) }
    }
}
