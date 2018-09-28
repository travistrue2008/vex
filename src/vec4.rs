use super::math;
use super::vec3::Vec3;
use std::cmp;
use std::convert::From;
use std::f32::EPSILON;
use std::fmt;
use std::ops;

#[derive(Copy, Clone)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    /// Creates a new vector <0.0, 0.0, 0.0, 0.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::new();
    /// let expected = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn new() -> Vec4 {
        Vec4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    /// Creates a new vector <1.0, 1.0, 1.0, 1.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::one();
    /// let expected = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn one() -> Vec4 {
        Vec4 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: 1.0,
        }
    }

    /// Creates a new vector from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// let expected = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn construct(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4 { x, y, z, w }
    }

    /// Find the dot product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let a = Vec4::construct(1.0, 0.0, 0.0, 0.0);
    /// let b = Vec4::construct(0.0, 0.0, 1.0, 0.0);
    /// let actual = Vec4::dot(&a, &b);
    /// let expected = 0.0;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn dot(a: &Vec4, b: &Vec4) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    }

    /// Find the minimum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let a = Vec4::construct(1.0, 4.0, 5.0, 7.0);
    /// let b = Vec4::construct(2.0, 3.0, 6.0, 8.0);
    /// let actual = Vec4::min(&a, &b);
    /// let expected = Vec4::construct(1.0, 3.0, 5.0, 7.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn min(a: &Vec4, b: &Vec4) -> Vec4 {
        Vec4::construct(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z), a.w.min(b.w))
    }

    /// Find the maximum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let a = Vec4::construct(1.0, 4.0, 5.0, 7.0);
    /// let b = Vec4::construct(2.0, 3.0, 6.0, 8.0);
    /// let actual = Vec4::max(&a, &b);
    /// let expected = Vec4::construct(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn max(a: &Vec4, b: &Vec4) -> Vec4 {
        Vec4::construct(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z), a.w.max(b.w))
    }

    /// Find the clamped (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let a = Vec4::construct(1.0, 3.0, 5.0, 7.0);
    /// let b = Vec4::construct(2.0, 4.0, 6.0, 8.0);
    /// let mut actual = Vec4::construct(0.0, 5.0, 10.0, 20.0);
    /// actual.clamp(&a, &b);
    /// let expected = Vec4::construct(1.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn clamp(&mut self, a: &Vec4, b: &Vec4) {
        let low = Self::min(a, b);
        let high = Self::max(a, b);
        let result = Self::max(&low, &Self::min(self, &high));
        self.set(result.x, result.y, result.z, result.w);
    }

    /// Set the components of a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::new();
    /// actual.set(1.0, 2.0, 3.0, 4.0);
    /// let expected = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn set(&mut self, x: f32, y: f32, z: f32, w: f32) {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
    }

    /// Set all components of the vector to zero
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    //  actual.zero();
    //  assert_eq!(actual, Vec4::new());
    /// ```
    pub fn zero(&mut self) {
        self.set(0.0, 0.0, 0.0, 0.0);
    }

    /// Get the magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::construct(1.0, 2.0, 3.0, 4.0).magnitude();
    /// let expected = 5.47722557505;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Get the squared magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::construct(1.0, 2.0, 3.0, 4.0).magnitude_squared();
    /// let expected = 30.0;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn magnitude_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Normalize the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// actual.normalize();
    /// let expected = Vec4::construct(0.18257418, 0.36514837, 0.5477225, 0.73029673);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn normalize(&mut self) -> f32 {
        let length = self.magnitude();
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
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(-1.0, -2.0, -3.0, -4.0);
    /// actual.abs();
    /// let expected = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
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
    /// use vex::Vec4;
    /// let actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// assert!(actual.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        math::is_valid(self.x) && math::is_valid(self.y) && math::is_valid(self.z)
    }

    fn print(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}, {}, {}, {}>", self.x, self.y, self.z, self.w)
    }
}

impl From<Vec3> for Vec4 {
    /// Creates a new Vec4 from the components of a Vec3
    ///
    /// # Examples
    /// ```
    /// use vex::Vec3;
    /// use vex::Vec4;
    /// let input = Vec3::construct(1.0, 2.0, 3.0);
    /// let actual = Vec4::from(input);
    /// let expected = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    fn from(item: Vec3) -> Vec4 {
        Vec4 {
            x: item.x,
            y: item.y,
            z: item.z,
            w: 0.0,
        }
    }
}

impl ops::Index<u32> for Vec4 {
    type Output = f32;

    /// Looks up a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut v = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v[1], 2.0);
    /// assert_eq!(v[2], 3.0);
    /// assert_eq!(v[3], 4.0);
    /// ```
    fn index(&self, index: u32) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Invalid index for Vec4: {}", index),
        }
    }
}

impl ops::IndexMut<u32> for Vec4 {
    /// Mutate a component by index
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut v = Vec4::new();
    /// v[0] = 4.0;
    /// v[1] = 5.0;
    /// v[2] = 6.0;
    /// v[3] = 7.0;
    /// assert_eq!(v[0], 4.0);
    /// assert_eq!(v[1], 5.0);
    /// assert_eq!(v[2], 6.0);
    /// assert_eq!(v[3], 7.0);
    /// ```
    fn index_mut<'a>(&'a mut self, index: u32) -> &'a mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Invalid index for Vec4: {}", index),
        }
    }
}

impl ops::Neg for Vec4 {
    type Output = Vec4;

    /// Negates all components in a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = -Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// let expected = Vec4::construct(-1.0, -2.0, -3.0, -4.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn neg(self) -> Vec4 {
        Vec4::construct(-self.x, -self.y, -self.z, -self.w)
    }
}

impl ops::Add<f32> for Vec4 {
    type Output = Vec4;

    /// Find the resulting vector by adding a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::construct(1.0, 2.0, 3.0, 4.0) + 1.0;
    /// let expected = Vec4::construct(2.0, 3.0, 4.0, 5.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add(self, _rhs: f32) -> Vec4 {
        Vec4::construct(self.x + _rhs, self.y + _rhs, self.z + _rhs, self.w + _rhs)
    }
}

impl ops::Add<Vec4> for Vec4 {
    type Output = Vec4;

    /// Add two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let a = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4::construct(5.0, 6.0, 7.0, 8.0);
    /// let actual = a + b;
    /// let expected = Vec4::construct(6.0, 8.0, 10.0, 12.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add(self, _rhs: Vec4) -> Vec4 {
        Vec4::construct(
            self.x + _rhs.x,
            self.y + _rhs.y,
            self.z + _rhs.z,
            self.w + _rhs.w,
        )
    }
}

impl ops::AddAssign<f32> for Vec4 {
    /// Increment a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// actual += 10.0;
    /// let expected = Vec4::construct(11.0, 12.0, 13.0, 14.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add_assign(&mut self, _rhs: f32) {
        self.x += _rhs;
        self.y += _rhs;
        self.z += _rhs;
        self.w += _rhs;
    }
}

impl ops::AddAssign<Vec4> for Vec4 {
    /// Increment a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// actual += Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// let expected = Vec4::construct(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn add_assign(&mut self, _rhs: Vec4) {
        self.x += _rhs.x;
        self.y += _rhs.y;
        self.z += _rhs.z;
        self.w += _rhs.w;
    }
}

impl ops::Sub<f32> for Vec4 {
    type Output = Vec4;

    /// Find the resulting vector by subtracting a scalar from a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::construct(1.0, 2.0, 3.0, 4.0) - 10.0;
    /// let expected = Vec4::construct(-9.0, -8.0, -7.0, -6.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub(self, _rhs: f32) -> Vec4 {
        Vec4::construct(self.x - _rhs, self.y - _rhs, self.z - _rhs, self.w - _rhs)
    }
}

impl ops::Sub<Vec4> for Vec4 {
    type Output = Vec4;

    /// Subtract two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let a = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4::construct(5.0, 4.0, 3.0, 2.0);
    /// let actual = a - b;
    /// let expected = Vec4::construct(-4.0, -2.0, 0.0, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub(self, _rhs: Vec4) -> Vec4 {
        Vec4::construct(
            self.x - _rhs.x,
            self.y - _rhs.y,
            self.z - _rhs.z,
            self.w - _rhs.w,
        )
    }
}

impl ops::SubAssign<f32> for Vec4 {
    /// Decrement a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// actual -= 1.0;
    /// let expected = Vec4::construct(0.0, 1.0, 2.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn sub_assign(&mut self, _rhs: f32) {
        self.x -= _rhs;
        self.y -= _rhs;
        self.z -= _rhs;
        self.w -= _rhs;
    }
}

impl ops::SubAssign<Vec4> for Vec4 {
    /// Decrement a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// actual -= Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(actual, Vec4::new());
    /// ```
    fn sub_assign(&mut self, _rhs: Vec4) {
        self.x -= _rhs.x;
        self.y -= _rhs.y;
        self.z -= _rhs.z;
        self.w -= _rhs.w;
    }
}

impl ops::Mul<f32> for Vec4 {
    type Output = Vec4;

    /// Find the resulting vector by multiplying a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::construct(1.0, 2.0, 3.0, 4.0) * 2.0;
    /// let expected = Vec4::construct(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul(self, _rhs: f32) -> Vec4 {
        Vec4::construct(self.x * _rhs, self.y * _rhs, self.z * _rhs, self.w * _rhs)
    }
}

impl ops::Mul<Vec4> for Vec4 {
    type Output = Vec4;

    /// Multiply two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let a = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4::construct(5.0, 6.0, 7.0, 8.0);
    /// let actual = a * b;
    /// let expected = Vec4::construct(5.0, 12.0, 21.0, 32.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul(self, _rhs: Vec4) -> Vec4 {
        Vec4::construct(
            self.x * _rhs.x,
            self.y * _rhs.y,
            self.z * _rhs.z,
            self.w * _rhs.w,
        )
    }
}

impl ops::MulAssign<f32> for Vec4 {
    /// Multiply a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// actual *= 2.0;
    /// let expected = Vec4::construct(2.0, 4.0, 6.0, 8.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul_assign(&mut self, _rhs: f32) {
        self.x *= _rhs;
        self.y *= _rhs;
        self.z *= _rhs;
        self.w *= _rhs;
    }
}

impl ops::MulAssign<Vec4> for Vec4 {
    /// Multiply a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// actual *= Vec4::construct(2.0, 3.0, 6.0, 8.0);
    /// let expected = Vec4::construct(2.0, 6.0, 18.0, 32.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn mul_assign(&mut self, _rhs: Vec4) {
        self.x *= _rhs.x;
        self.y *= _rhs.y;
        self.z *= _rhs.z;
        self.w *= _rhs.w;
    }
}

impl ops::Div<f32> for Vec4 {
    type Output = Vec4;

    /// Find the resulting vector by dividing a scalar to a vector's components
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::construct(1.0, 2.0, 3.0, 4.0) / 2.0;
    /// let expected = Vec4::construct(0.5, 1.0, 1.5, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn div(self, _rhs: f32) -> Vec4 {
        Vec4::construct(self.x / _rhs, self.y / _rhs, self.z / _rhs, self.w / _rhs)
    }
}

impl ops::Div<Vec4> for Vec4 {
    type Output = Vec4;

    /// Divide two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let a = Vec4::construct(2.0, 4.0, 6.0, 8.0);
    /// let b = Vec4::construct(1.0, 4.0, 12.0, 32.0);
    /// let actual = a / b;
    /// let expected = Vec4::construct(2.0, 1.0, 0.5, 0.25);
    /// assert_eq!(actual, expected);
    /// ```
    fn div(self, _rhs: Vec4) -> Vec4 {
        Vec4::construct(
            self.x / _rhs.x,
            self.y / _rhs.y,
            self.z / _rhs.z,
            self.w / _rhs.w,
        )
    }
}

impl ops::DivAssign<f32> for Vec4 {
    /// Divide a vector by a scalar
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(1.0, 2.0, 3.0, 4.0);
    /// actual /= 2.0;
    /// let expected = Vec4::construct(0.5, 1.0, 1.5, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    fn div_assign(&mut self, _rhs: f32) {
        self.x /= _rhs;
        self.y /= _rhs;
        self.z /= _rhs;
        self.w /= _rhs;
    }
}

impl ops::DivAssign<Vec4> for Vec4 {
    /// Divide a vector by another vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let mut actual = Vec4::construct(2.0, 4.0, 6.0, 8.0);
    /// actual /= Vec4::construct(1.0, 4.0, 12.0, 32.0);
    /// let expected = Vec4::construct(2.0, 1.0, 0.5, 0.25);
    /// assert_eq!(actual, expected);
    /// ```
    fn div_assign(&mut self, _rhs: Vec4) {
        self.x /= _rhs.x;
        self.y /= _rhs.y;
        self.z /= _rhs.z;
        self.w /= _rhs.w;
    }
}

impl cmp::PartialEq for Vec4 {
    /// Determines if two vectors' components are equivalent
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// assert!(Vec4::new() == Vec4::new());
    /// ```
    fn eq(&self, _rhs: &Vec4) -> bool {
        self.x == _rhs.x && self.y == _rhs.y && self.z == _rhs.z && self.w == _rhs.w
    }
}

impl fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}
