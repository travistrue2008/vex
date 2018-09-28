use super::common;
use super::vector3::Vector3;
use std::cmp;
use std::convert::From;
use std::f32::EPSILON;
use std::fmt;
use std::ops;

pub const ZERO: Vector2 = Vector2 { x: 0.0, y: 0.0 };
pub const ONE: Vector2 = Vector2 { x: 1.0, y: 1.0 };

#[derive(Copy, Clone)]
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
    pub fn new() -> Vector2 {
        ZERO
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
    pub fn set(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
    }

    /// Get the magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0).magnitude();
    /// let expected = 2.2360679775;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Get the squared magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let actual = Vector2::make(1.0, 2.0).magnitude_squared();
    /// let expected = 5.0;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn magnitude_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Normalize the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual.normalize();
    /// let expected = Vector2::make(0.4472135955, 0.894427191);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn normalize(&mut self) -> f32 {
        let length = self.magnitude();
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
    pub fn is_valid(&self) -> bool {
        for i in 0..2 {
            if !common::is_valid(self[i]) {
                return false;
            }
        }

        true
    }

    fn print(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}, {}>", self.x, self.y)
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
    fn from(item: Vector3) -> Self {
        Vector2 {
            x: item.x,
            y: item.y,
        }
    }
}

impl ops::Index<u32> for Vector2 {
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
    fn index(&self, index: u32) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Invalid index for Vector2: {}", index),
        }
    }
}

impl ops::IndexMut<u32> for Vector2 {
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
    fn index_mut<'a>(&'a mut self, index: u32) -> &'a mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Invalid index for Vector2: {}", index),
        }
    }
}

impl ops::Neg for Vector2 {
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
    fn neg(self) -> Vector2 {
        Vector2::make(-self.x, -self.y)
    }
}

impl ops::Add<f32> for Vector2 {
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
    fn add(self, _rhs: f32) -> Vector2 {
        Vector2::make(self.x + _rhs, self.y + _rhs)
    }
}

impl ops::Add<Vector2> for Vector2 {
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
    fn add(self, _rhs: Vector2) -> Vector2 {
        Vector2::make(self.x + _rhs.x, self.y + _rhs.y)
    }
}

impl ops::AddAssign<f32> for Vector2 {
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
    fn add_assign(&mut self, _rhs: f32) {
        self.x += _rhs;
        self.y += _rhs;
    }
}

impl ops::AddAssign<Vector2> for Vector2 {
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
    fn add_assign(&mut self, _rhs: Vector2) {
        self.x += _rhs.x;
        self.y += _rhs.y;
    }
}

impl ops::Sub<f32> for Vector2 {
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
    fn sub(self, _rhs: f32) -> Vector2 {
        Vector2::make(self.x - _rhs, self.y - _rhs)
    }
}

impl ops::Sub<Vector2> for Vector2 {
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
    fn sub(self, _rhs: Vector2) -> Vector2 {
        Vector2::make(self.x - _rhs.x, self.y - _rhs.y)
    }
}

impl ops::SubAssign<f32> for Vector2 {
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
    fn sub_assign(&mut self, _rhs: f32) {
        self.x -= _rhs;
        self.y -= _rhs;
    }
}

impl ops::SubAssign<Vector2> for Vector2 {
    /// Decrement a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vector2;
    /// let mut actual = Vector2::make(1.0, 2.0);
    /// actual -= Vector2::make(1.0, 2.0);
    /// assert_eq!(actual, Vector2::new());
    /// ```
    fn sub_assign(&mut self, _rhs: Vector2) {
        self.x -= _rhs.x;
        self.y -= _rhs.y;
    }
}

impl ops::Mul<f32> for Vector2 {
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
    fn mul(self, _rhs: f32) -> Vector2 {
        Vector2::make(self.x * _rhs, self.y * _rhs)
    }
}

impl ops::Mul<Vector2> for Vector2 {
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
    fn mul(self, _rhs: Vector2) -> Vector2 {
        Vector2::make(self.x * _rhs.x, self.y * _rhs.y)
    }
}

impl ops::MulAssign<f32> for Vector2 {
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
    fn mul_assign(&mut self, _rhs: f32) {
        self.x *= _rhs;
        self.y *= _rhs;
    }
}

impl ops::MulAssign<Vector2> for Vector2 {
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
    fn mul_assign(&mut self, _rhs: Vector2) {
        self.x *= _rhs.x;
        self.y *= _rhs.y;
    }
}

impl ops::Div<f32> for Vector2 {
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
    fn div(self, _rhs: f32) -> Vector2 {
        Vector2::make(self.x / _rhs, self.y / _rhs)
    }
}

impl ops::Div<Vector2> for Vector2 {
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
    fn div(self, _rhs: Vector2) -> Vector2 {
        Vector2::make(self.x / _rhs.x, self.y / _rhs.y)
    }
}

impl ops::DivAssign<f32> for Vector2 {
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
    fn div_assign(&mut self, _rhs: f32) {
        self.x /= _rhs;
        self.y /= _rhs;
    }
}

impl ops::DivAssign<Vector2> for Vector2 {
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
    fn eq(&self, _rhs: &Vector2) -> bool {
        for i in 0..2 {
            if self[i] != _rhs[i] {
                return false;
            }
        }

        true
    }
}

impl fmt::Debug for Vector2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl fmt::Display for Vector2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}
