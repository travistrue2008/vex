use crate::common;
use crate::vector3::Vector3;

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
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    /// Creates a vector <0.0, 0.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::new();
    /// let expected = Vector2 { x: 0.0, y: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn new() -> Vector2 {
        Vector2 { x: 0.0, y: 0.0 }
    }

    /// Creates a vector <1.0, 1.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::one();
    /// let expected = Vector2 { x: 1.0, y: 1.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn one() -> Vector2 {
        Vector2 { x: 1.0, y: 1.0 }
    }

    /// Creates a vector from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0);
    /// let expected = Vector2 { x: 1.0, y: 2.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn make(x: f32, y: f32) -> Vector2 {
        Vector2 { x, y }
    }

    /// Find the dot product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 0.0);
    /// let b = Vector2::make(0.0, 1.0);
    /// let actual = Vector2::dot(&a, &b);
    /// let expected = 0.0;
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn dot(a: &Vector2, b: &Vector2) -> f32 {
        a.x * b.x + a.y * b.y
    }

    /// Find the cross product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 0.0);
    /// let b = Vector2::make(0.0, 1.0);
    /// let actual = Vector2::cross(&a, &b);
    /// let expected = 1.0;
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn cross(a: &Vector2, b: &Vector2) -> f32 {
        a.x * b.y - a.y * b.x
    }

    /// Find the cross product between a scalar (left) and vector (right)
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let s = 1.0;
    /// let v = Vector2::make(1.0, 0.0);
    /// let actual = Vector2::cross_scalar_vec(s, &v);
    /// let expected = Vector2::make(0.0, 1.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn cross_scalar_vec(s: f32, v: &Vector2) -> Vector2 {
        Vector2::make(-s * v.y, s * v.x)
    }

    /// Find the cross product between a vector (left) and scalar (right)
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let s = 1.0;
    /// let v = Vector2::make(1.0, 0.0);
    /// let actual = Vector2::cross_vec_scalar(&v, s);
    /// let expected = Vector2::make(0.0, -1.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn cross_vec_scalar(v: &Vector2, s: f32) -> Vector2 {
        Vector2::make(s * v.y, -s * v.x)
    }

    /// Find the minimum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 4.0);
    /// let b = Vector2::make(2.0, 3.0);
    /// let actual = Vector2::min(&a, &b);
    /// let expected = Vector2::make(1.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn min(a: &Vector2, b: &Vector2) -> Vector2 {
        Vector2::make(a.x.min(b.x), a.y.min(b.y))
    }

    /// Find the maximum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 4.0);
    /// let b = Vector2::make(2.0, 3.0);
    /// let actual = Vector2::max(&a, &b);
    /// let expected = Vector2::make(2.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn max(a: &Vector2, b: &Vector2) -> Vector2 {
        Vector2::make(a.x.max(b.x), a.y.max(b.y))
    }

    /// Find the clamped (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 3.0);
    /// let b = Vector2::make(2.0, 4.0);
    /// let mut actual = Vector2::make(0.0, 5.0);
    /// actual.clamp(&a, &b);
    /// let expected = Vector2::make(1.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn clamp(&mut self, a: &Vector2, b: &Vector2) {
        let low = Self::min(a, b);
        let high = Self::max(a, b);
        let result = Self::max(&low, &Self::min(self, &high));
        self.set(result.x, result.y);
    }

    /// Set the components of a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::new();
    /// actual.set(1.0, 2.0);
    /// let expected = Vector2::make(1.0, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn set(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
    }

    /// Get the magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0).mag();
    /// let expected = 2.2360679775;
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
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0).mag_sq();
    /// let expected = 5.0;
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn mag_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Normalize the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual.norm();
    /// let expected = Vector2::make(0.4472135955, 0.894427191);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn norm(&mut self) -> f32 {
        let length = self.mag();
        if length > EPSILON {
            self.x /= length;
            self.y /= length;
            length
        } else {
            0.0
        }
    }

    /// Set the components of a vector to their absolute values
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(-1.0, -2.0);
    /// actual.abs();
    /// let expected = Vector2::make(1.0, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn abs(&mut self) {
        self.x = self.x.abs();
        self.y = self.y.abs();
    }

    /// Skew the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual.skew();
    /// let expected = Vector2::make(-2.0, 1.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    pub fn skew(&mut self) {
        let x = self.x;
        self.x = -self.y;
        self.y = x;
    }

    /// Determine whether or not all components of the vector are valid
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0);
    /// assert!(actual.is_valid());
    /// ```
    #[inline]
    pub fn is_valid(&self) -> bool {
        for i in 0..2 {
            if !common::is_valid(self[i]) {
                return false;
            }
        }

        true
    }
}

impl From<Vector3> for Vector2 {
    /// Creates a Vector2 from the components of a Vector3
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// use vex::Vector3;
    /// let input = Vector3::make(1.0, 2.0, 3.0);
    /// let actual = Vector2::from(input);
    /// let expected = Vector2 { x: 1.0, y: 2.0 };
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn from(item: Vector3) -> Self {
        Vector2 {
            x: item.x,
            y: item.y,
        }
    }
}

impl Index<u32> for Vector2 {
    type Output = f32;

    /// Looks up a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut v = Vector2::make(1.0, 2.0);
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v[1], 2.0);
    /// ```
    #[inline]
    fn index(&self, index: u32) -> &f32 {
        unsafe {
            match index {
                0 => &self.x,
                1 => &self.y,
                _ => panic!("Invalid index for Vector2: {}", index),
            }
        }
    }
}

impl IndexMut<u32> for Vector2 {
    /// Mutate a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut v = Vector2::new();
    /// v[0] = 3.0;
    /// v[1] = 4.0;
    /// assert_eq!(v[0], 3.0);
    /// assert_eq!(v[1], 4.0);
    /// ```
    #[inline]
    fn index_mut<'a>(&'a mut self, index: u32) -> &'a mut f32 {
        unsafe {
            match index {
                0 => &mut self.x,
                1 => &mut self.y,
                _ => panic!("Invalid index for Vector2: {}", index),
            }
        }
    }
}

impl Neg for Vector2 {
    type Output = Vector2;

    /// Negates all components in a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = -Vector2::make(1.0, 2.0);
    /// let expected = Vector2::make(-1.0, -2.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn neg(self) -> Vector2 {
        Vector2::make(-self.x, -self.y)
    }
}

impl Add<f32> for Vector2 {
    type Output = Vector2;

    /// Find the resulting vector by adding a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0) + 1.0;
    /// let expected = Vector2::make(2.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: f32) -> Vector2 {
        Vector2::make(self.x + _rhs, self.y + _rhs)
    }
}

impl Add<Vector2> for Vector2 {
    type Output = Vector2;

    /// Add two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 2.0);
    /// let b = Vector2::make(3.0, 4.0);
    /// let actual = a + b;
    /// let expected = Vector2::make(4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add(self, _rhs: Vector2) -> Vector2 {
        Vector2::make(self.x + _rhs.x, self.y + _rhs.y)
    }
}

impl AddAssign<f32> for Vector2 {
    /// Increment a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual += 10.0;
    /// let expected = Vector2::make(11.0, 12.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: f32) {
        self.x += _rhs;
        self.y += _rhs;
    }
}

impl AddAssign<Vector2> for Vector2 {
    /// Increment a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual += Vector2::make(1.0, 2.0);
    /// let expected = Vector2::make(2.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn add_assign(&mut self, _rhs: Vector2) {
        self.x += _rhs.x;
        self.y += _rhs.y;
    }
}

impl Sub<f32> for Vector2 {
    type Output = Vector2;

    /// Find the resulting vector by subtracting a scalar from a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0) - 10.0;
    /// let expected = Vector2::make(-9.0, -8.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: f32) -> Vector2 {
        Vector2::make(self.x - _rhs, self.y - _rhs)
    }
}

impl Sub<Vector2> for Vector2 {
    type Output = Vector2;

    /// Subtract two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 2.0);
    /// let b = Vector2::make(4.0, 3.0);
    /// let actual = a - b;
    /// let expected = Vector2::make(-3.0, -1.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub(self, _rhs: Vector2) -> Vector2 {
        Vector2::make(self.x - _rhs.x, self.y - _rhs.y)
    }
}

impl SubAssign<f32> for Vector2 {
    /// Decrement a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual -= 1.0;
    /// let expected = Vector2::make(0.0, 1.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: f32) {
        self.x -= _rhs;
        self.y -= _rhs;
    }
}

impl SubAssign<Vector2> for Vector2 {
    /// Decrement a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual -= Vector2::make(1.0, 2.0);
    /// assert_eq!(actual, Vector2::new());
    /// ```
    #[inline]
    fn sub_assign(&mut self, _rhs: Vector2) {
        self.x -= _rhs.x;
        self.y -= _rhs.y;
    }
}

impl Mul<f32> for Vector2 {
    type Output = Vector2;

    /// Find the resulting vector by multiplying a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0) * 2.0;
    /// let expected = Vector2::make(2.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: f32) -> Vector2 {
        Vector2::make(self.x * _rhs, self.y * _rhs)
    }
}

impl Mul<Vector2> for Vector2 {
    type Output = Vector2;

    /// Multiply two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 2.0);
    /// let b = Vector2::make(2.0, 3.0);
    /// let actual = a * b;
    /// let expected = Vector2::make(2.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul(self, _rhs: Vector2) -> Vector2 {
        Vector2::make(self.x * _rhs.x, self.y * _rhs.y)
    }
}

impl MulAssign<f32> for Vector2 {
    /// Multiply a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual *= 2.0;
    /// let expected = Vector2::make(2.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: f32) {
        self.x *= _rhs;
        self.y *= _rhs;
    }
}

impl MulAssign<Vector2> for Vector2 {
    /// Multiply a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual *= Vector2::make(2.0, 3.0);
    /// let expected = Vector2::make(2.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn mul_assign(&mut self, _rhs: Vector2) {
        self.x *= _rhs.x;
        self.y *= _rhs.y;
    }
}

impl Div<f32> for Vector2 {
    type Output = Vector2;

    /// Find the resulting vector by dividing a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0) / 2.0;
    /// let expected = Vector2::make(0.5, 1.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div(self, _rhs: f32) -> Vector2 {
        Vector2::make(self.x / _rhs, self.y / _rhs)
    }
}

impl Div<Vector2> for Vector2 {
    type Output = Vector2;

    /// Divide two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let a = Vector2::make(1.0, 2.0);
    /// let b = Vector2::make(2.0, 8.0);
    /// let actual = a / b;
    /// let expected = Vector2::make(0.5, 0.25);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div(self, _rhs: Vector2) -> Vector2 {
        Vector2::make(self.x / _rhs.x, self.y / _rhs.y)
    }
}

impl DivAssign<f32> for Vector2 {
    /// Divide a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual /= 2.0;
    /// let expected = Vector2::make(0.5, 1.0);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div_assign(&mut self, _rhs: f32) {
        self.x /= _rhs;
        self.y /= _rhs;
    }
}

impl DivAssign<Vector2> for Vector2 {
    /// Divide a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual /= Vector2::make(2.0, 8.0);
    /// let expected = Vector2::make(0.5, 0.25);
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    fn div_assign(&mut self, _rhs: Vector2) {
        self.x /= _rhs.x;
        self.y /= _rhs.y;
    }
}

impl cmp::PartialEq for Vector2 {
    /// Determines if two vectors' components are equivalent
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// assert!(Vector2::new() == Vector2::new());
    /// ```
    #[inline]
    fn eq(&self, _rhs: &Vector2) -> bool {
        for i in 0..2 {
            if self[i] != _rhs[i] {
                return false;
            }
        }

        true
    }
}

impl Display for Vector2 {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        unsafe { write!(f, "<{}  {}>", self.x, self.y) }
    }
}
