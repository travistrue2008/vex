use super::math;
use super::vec2::Vec2;
use super::vec4::Vec4;
use std::cmp;
use std::convert::From;
use std::f32::EPSILON;
use std::fmt;
use std::ops;

#[derive(Copy, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Creates a vector <0.0, 0.0, 0.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::new();
    /// let expected = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn new() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Creates a vector <1.0, 1.0, 1.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::one();
    /// let expected = Vec3 { x: 1.0, y: 1.0, z: 1.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn one() -> Vec3 {
        Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }

    /// Creates a right-pointing vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::right();
    /// let expected = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn right() -> Vec3 {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Creates a right-pointing vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::up();
    /// let expected = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn up() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }

    /// Creates a right-pointing vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::forward();
    /// let expected = Vec3 { x: 0.0, y: 0.0, z: -1.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn forward() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        }
    }

    /// Creates a vector from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::construct(1.0, 2.0, 3.0);
    /// let expected = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn construct(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// Find the dot product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(1.0, 0.0, 0.0);
    /// let b = Vec3::construct(0.0, 0.0, 1.0);
    /// let actual = Vec3::dot(&a, &b);
    /// let expected = 0.0;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn dot(a: &Vec3, b: &Vec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    /// Find the cross product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(0.0, 0.0, 1.0);
    /// let b = Vec3::construct(1.0, 0.0, 0.0);
    /// let actual = Vec3::cross(&a, &b);
    /// let expected = Vec3::construct(0.0, 1.0, 0.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
        Vec3::construct(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    }

    /// Find the minimum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(1.0, 4.0, 5.0);
    /// let b = Vec3::construct(2.0, 3.0, 6.0);
    /// let actual = Vec3::min(&a, &b);
    /// let expected = Vec3::construct(1.0, 3.0, 5.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn min(a: &Vec3, b: &Vec3) -> Vec3 {
        Vec3::construct(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
    }

    /// Find the maximum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(1.0, 4.0, 5.0);
    /// let b = Vec3::construct(2.0, 3.0, 6.0);
    /// let actual = Vec3::max(&a, &b);
    /// let expected = Vec3::construct(2.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn max(a: &Vec3, b: &Vec3) -> Vec3 {
        Vec3::construct(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
    }

    /// Find the clamped (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(1.0, 3.0, 5.0);
    /// let b = Vec3::construct(2.0, 4.0, 6.0);
    /// let mut actual = Vec3::construct(0.0, 5.0, 10.0);
    /// actual.clamp(&a, &b);
    /// let expected = Vec3::construct(1.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn clamp(&mut self, a: &Vec3, b: &Vec3) {
        let low = Self::min(a, b);
        let high = Self::max(a, b);
        let result = Self::max(&low, &Self::min(self, &high));
        self.set(result.x, result.y, result.z);
    }

    /// Set the components of a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::new();
    /// actual.set(1.0, 2.0, 3.0);
    /// let expected = Vec3::construct(1.0, 2.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn set(&mut self, x: f32, y: f32, z: f32) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    /// Set all components of the vector to zero
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    //  actual.zero();
    //  assert_eq!(actual, Vec3::new());
    /// ```
    pub fn zero(&mut self) {
        self.set(0.0, 0.0, 0.0);
    }

    /// Get the magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::construct(1.0, 2.0, 3.0).magnitude();
    /// let expected = 3.74165738677;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Get the squared magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::construct(1.0, 2.0, 3.0).magnitude_squared();
    /// let expected = 14.0;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn magnitude_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalize the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    /// actual.normalize();
    /// let expected = Vec3::construct(0.26726124191, 0.53452248382, 0.8017837);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn normalize(&mut self) -> f32 {
        let length = self.magnitude();
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
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(-1.0, -2.0, -3.0);
    /// actual.abs();
    /// let expected = Vec3::construct(1.0, 2.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn abs(&mut self) {
        self.x = self.x.abs();
        self.y = self.y.abs();
        self.z = self.z.abs();
    }

    /// Determine whether or not all components of the vector are valid
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::construct(1.0, 2.0, 3.0);
    /// assert!(actual.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        math::is_valid(self.x) && math::is_valid(self.y) && math::is_valid(self.z)
    }

    fn print(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}, {}, {}>", self.x, self.y, self.z)
    }
}

impl From<Vec2> for Vec3 {
    /// Creates a Vec3 from the components of a Vec2
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// use vex::Vec3;
    /// let input = Vec2::construct(1.0, 2.0);
    /// let actual = Vec3::from(input);
    /// let expected = Vec3 { x: 1.0, y: 2.0, z: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    fn from(item: Vec2) -> Vec3 {
        Vec3 {
            x: item.x,
            y: item.y,
            z: 0.0,
        }
    }
}

impl From<Vec4> for Vec3 {
    /// Creates a Vec3 from the components of a Vec4
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// use vex::Vec4;
    /// let input = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// let actual = Vec3::from(input);
    /// let expected = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
    /// assert_eq!(actual, expected);
    /// ```
    fn from(item: Vec4) -> Vec3 {
        Vec3 {
            x: item.x,
            y: item.y,
            z: item.z,
        }
    }
}

impl ops::Index<u32> for Vec3 {
    type Output = f32;

    /// Looks up a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut v = Vec3::construct(1.0, 2.0, 3.0);
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v[1], 2.0);
    /// assert_eq!(v[2], 3.0);
    /// ```
    fn index(&self, index: u32) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Invalid index for Vec3: {}", index),
        }
    }
}

impl ops::IndexMut<u32> for Vec3 {
    /// Mutate a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut v = Vec3::new();
    /// v[0] = 4.0;
    /// v[1] = 5.0;
    /// v[2] = 6.0;
    /// assert_eq!(v[0], 4.0);
    /// assert_eq!(v[1], 5.0);
    /// assert_eq!(v[2], 6.0);
    /// ```
    fn index_mut<'a>(&'a mut self, index: u32) -> &'a mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Invalid index for Vec3: {}", index),
        }
    }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;

    /// Negates all components in a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = -Vec3::construct(1.0, 2.0, 3.0);
    /// let expected = Vec3::construct(-1.0, -2.0, -3.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn neg(self) -> Vec3 {
        Vec3::construct(-self.x, -self.y, -self.z)
    }
}

impl ops::Add<f32> for Vec3 {
    type Output = Vec3;

    /// Find the resulting vector by adding a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::construct(1.0, 2.0, 3.0) + 1.0;
    /// let expected = Vec3::construct(2.0, 3.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add(self, _rhs: f32) -> Vec3 {
        Vec3::construct(self.x + _rhs, self.y + _rhs, self.z + _rhs)
    }
}

impl ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    /// Add two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(1.0, 2.0, 3.0);
    /// let b = Vec3::construct(4.0, 5.0, 6.0);
    /// let actual = a + b;
    /// let expected = Vec3::construct(5.0, 7.0, 9.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add(self, _rhs: Vec3) -> Vec3 {
        Vec3::construct(self.x + _rhs.x, self.y + _rhs.y, self.z + _rhs.z)
    }
}

impl ops::AddAssign<f32> for Vec3 {
    /// Increment a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    /// actual += 10.0;
    /// let expected = Vec3::construct(11.0, 12.0, 13.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add_assign(&mut self, _rhs: f32) {
        self.x += _rhs;
        self.y += _rhs;
        self.z += _rhs;
    }
}

impl ops::AddAssign<Vec3> for Vec3 {
    /// Increment a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    /// actual += Vec3::construct(1.0, 2.0, 3.0);
    /// let expected = Vec3::construct(2.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add_assign(&mut self, _rhs: Vec3) {
        self.x += _rhs.x;
        self.y += _rhs.y;
        self.z += _rhs.z;
    }
}

impl ops::Sub<f32> for Vec3 {
    type Output = Vec3;

    /// Find the resulting vector by subtracting a scalar from a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::construct(1.0, 2.0, 3.0) - 10.0;
    /// let expected = Vec3::construct(-9.0, -8.0, -7.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub(self, _rhs: f32) -> Vec3 {
        Vec3::construct(self.x - _rhs, self.y - _rhs, self.z - _rhs)
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    /// Subtract two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(1.0, 2.0, 3.0);
    /// let b = Vec3::construct(5.0, 4.0, 3.0);
    /// let actual = a - b;
    /// let expected = Vec3::construct(-4.0, -2.0, 0.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub(self, _rhs: Vec3) -> Vec3 {
        Vec3::construct(self.x - _rhs.x, self.y - _rhs.y, self.z - _rhs.z)
    }
}

impl ops::SubAssign<f32> for Vec3 {
    /// Decrement a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    /// actual -= 1.0;
    /// let expected = Vec3::construct(0.0, 1.0, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub_assign(&mut self, _rhs: f32) {
        self.x -= _rhs;
        self.y -= _rhs;
        self.z -= _rhs;
    }
}

impl ops::SubAssign<Vec3> for Vec3 {
    /// Decrement a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    /// actual -= Vec3::construct(1.0, 2.0, 3.0);
    /// assert_eq!(actual, Vec3::new());
    /// ```
    fn sub_assign(&mut self, _rhs: Vec3) {
        self.x -= _rhs.x;
        self.y -= _rhs.y;
        self.z -= _rhs.z;
    }
}

impl ops::Mul<f32> for Vec3 {
    type Output = Vec3;

    /// Find the resulting vector by multiplying a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::construct(1.0, 2.0, 3.0) * 2.0;
    /// let expected = Vec3::construct(2.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul(self, _rhs: f32) -> Vec3 {
        Vec3::construct(self.x * _rhs, self.y * _rhs, self.z * _rhs)
    }
}

impl ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;

    /// Multiply two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(1.0, 2.0, 3.0);
    /// let b = Vec3::construct(3.0, 4.0, 5.0);
    /// let actual = a * b;
    /// let expected = Vec3::construct(3.0, 8.0, 15.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul(self, _rhs: Vec3) -> Vec3 {
        Vec3::construct(self.x * _rhs.x, self.y * _rhs.y, self.z * _rhs.z)
    }
}

impl ops::MulAssign<f32> for Vec3 {
    /// Multiply a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    /// actual *= 2.0;
    /// let expected = Vec3::construct(2.0, 4.0, 6.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul_assign(&mut self, _rhs: f32) {
        self.x *= _rhs;
        self.y *= _rhs;
        self.z *= _rhs;
    }
}

impl ops::MulAssign<Vec3> for Vec3 {
    /// Multiply a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    /// actual *= Vec3::construct(2.0, 3.0, 6.0);
    /// let expected = Vec3::construct(2.0, 6.0, 18.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul_assign(&mut self, _rhs: Vec3) {
        self.x *= _rhs.x;
        self.y *= _rhs.y;
        self.z *= _rhs.z;
    }
}

impl ops::Div<f32> for Vec3 {
    type Output = Vec3;

    /// Find the resulting vector by dividing a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let actual = Vec3::construct(1.0, 2.0, 3.0) / 2.0;
    /// let expected = Vec3::construct(0.5, 1.0, 1.5);
    /// assert_eq!(actual, expected);
    /// ```
    fn div(self, _rhs: f32) -> Vec3 {
        Vec3::construct(self.x / _rhs, self.y / _rhs, self.z / _rhs)
    }
}

impl ops::Div<Vec3> for Vec3 {
    type Output = Vec3;

    /// Divide two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let a = Vec3::construct(1.0, 2.0, 4.0);
    /// let b = Vec3::construct(2.0, 8.0, 32.0);
    /// let actual = a / b;
    /// let expected = Vec3::construct(0.5, 0.25, 0.125);
    /// assert_eq!(actual, expected);
    /// ```
    fn div(self, _rhs: Vec3) -> Vec3 {
        Vec3::construct(self.x / _rhs.x, self.y / _rhs.y, self.z / _rhs.z)
    }
}

impl ops::DivAssign<f32> for Vec3 {
    /// Divide a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 3.0);
    /// actual /= 2.0;
    /// let expected = Vec3::construct(0.5, 1.0, 1.5);
    /// assert_eq!(actual, expected);
    /// ```
    fn div_assign(&mut self, _rhs: f32) {
        self.x /= _rhs;
        self.y /= _rhs;
        self.z /= _rhs;
    }
}

impl ops::DivAssign<Vec3> for Vec3 {
    /// Divide a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// let mut actual = Vec3::construct(1.0, 2.0, 4.0);
    /// actual /= Vec3::construct(2.0, 8.0, 32.0);
    /// let expected = Vec3::construct(0.5, 0.25, 0.125);
    /// assert_eq!(actual, expected);
    /// ```
    fn div_assign(&mut self, _rhs: Vec3) {
        self.x /= _rhs.x;
        self.y /= _rhs.y;
        self.z /= _rhs.z;
    }
}

impl cmp::PartialEq for Vec3 {
    /// Determines if two vectors' components are equivalent
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// assert!(Vec3::new() == Vec3::new());
    /// ```
    fn eq(&self, _rhs: &Vec3) -> bool {
        self.x == _rhs.x && self.y == _rhs.y && self.z == _rhs.z
    }
}

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}
