use super::math;
use std::cmp;
use std::f32::EPSILON;
use std::fmt;
use std::ops;

#[derive(Copy, Clone)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    /// Creates a new vector <0.0, 0.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let actual = Vec2::new();
    /// let expected = Vec2 { x: 0.0, y: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn new() -> Vec2 {
        Vec2 { x: 0.0, y: 0.0 }
    }

    /// Creates a new vector <1.0, 1.0>
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let actual = Vec2::one();
    /// let expected = Vec2 { x: 1.0, y: 1.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn one() -> Vec2 {
        Vec2 { x: 1.0, y: 1.0 }
    }

    /// Creates a new vector from the provided values
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let actual = Vec2::from(1.0, 2.0);
    /// let expected = Vec2 { x: 1.0, y: 2.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn from(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    /// Find the dot product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let a = Vec2::from(1.0, 0.0);
    /// let b = Vec2::from(0.0, 1.0);
    /// let actual = Vec2::dot(&a, &b);
    /// let expected = 0.0;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn dot(a: &Vec2, b: &Vec2) -> f32 {
        a.x * b.x + a.y * b.y
    }

    /// Find the cross product between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let a = Vec2::from(1.0, 0.0);
    /// let b = Vec2::from(0.0, 1.0);
    /// let actual = Vec2::cross(&a, &b);
    /// let expected = 1.0;
    /// assert_eq!(actual, expected);
    /// ```
    pub fn cross(a: &Vec2, b: &Vec2) -> f32 {
        a.x * b.y - a.y * b.x
    }

    /// Find the cross product between a scalar (left) and vector (right)
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let s = 1.0;
    /// let v = Vec2::from(1.0, 0.0);
    /// let actual = Vec2::cross_scalar_vec(s, &v);
    /// let expected = Vec2::from(0.0, 1.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn cross_scalar_vec(s: f32, v: &Vec2) -> Vec2 {
        Vec2::from(-s * v.y, s * v.x)
    }

    /// Find the cross product between a vector (left) and scalar (right)
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let s = 1.0;
    /// let v = Vec2::from(1.0, 0.0);
    /// let actual = Vec2::cross_vec_scalar(&v, s);
    /// let expected = Vec2::from(0.0, -1.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn cross_vec_scalar(v: &Vec2, s: f32) -> Vec2 {
        Vec2::from(s * v.y, -s * v.x)
    }

    /// Find the minimum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let a = Vec2::from(1.0, 4.0);
    /// let b = Vec2::from(2.0, 3.0);
    /// let actual = Vec2::min(&a, &b);
    /// let expected = Vec2::from(1.0, 3.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn min(a: &Vec2, b: &Vec2) -> Vec2 {
        Vec2::from(a.x.min(b.x), a.y.min(b.y))
    }

    /// Find the maximum (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let a = Vec2::from(1.0, 4.0);
    /// let b = Vec2::from(2.0, 3.0);
    /// let actual = Vec2::max(&a, &b);
    /// let expected = Vec2::from(2.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn max(a: &Vec2, b: &Vec2) -> Vec2 {
        Vec2::from(a.x.max(b.x), a.y.max(b.y))
    }

    /// Find the clamped (component-wise) vector between two vectors
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let a = Vec2::from(1.0, 3.0);
    /// let b = Vec2::from(2.0, 4.0);
    /// let mut actual = Vec2::from(0.0, 5.0);
    /// actual.clamp(&a, &b);
    /// let expected = Vec2::from(1.0, 4.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn clamp(&mut self, a: &Vec2, b: &Vec2) {
        let low = Self::min(a, b);
        let high = Self::max(a, b);
        let result = Self::max(&low, &Self::min(self, &high));
        self.set(result.x, result.y);
    }

    /// Set the components of a vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let mut actual = Vec2::new();
    /// actual.set(1.0, 2.0);
    /// let expected = Vec2::from(1.0, 2.0);
    /// assert_eq!(actual, expected);
    /// ```
    pub fn set(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
    }

    /// Set all components of the vector to zero
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let mut actual = Vec2::from(1.0, 2.0);
    //  actual.zero();
    //  assert_eq!(actual, Vec2::new());
    /// ```
    pub fn zero(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }

    /// Get the magnitude of the vector
    ///
    /// # Examples
    /// ```
    /// use vex::Vec2;
    /// let actual = Vec2::from(1.0, 2.0).magnitude();
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
    /// use vex::Vec2;
    /// let actual = Vec2::from(1.0, 2.0).magnitude_squared();
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
    /// use vex::Vec2;
    /// let mut actual = Vec2::from(1.0, 2.0);
    /// actual.normalize();
    /// let expected = Vec2::from(0.4472135955, 0.894427191);
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
    /// use vex::Vec2;
    /// let mut actual = Vec2::from(-1.0, -2.0);
    /// actual.abs();
    /// let expected = Vec2::from(1.0, 2.0);
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
    /// use vex::Vec2;
    /// let mut actual = Vec2::from(1.0, 2.0);
    /// actual.skew();
    /// let expected = Vec2::from(-2.0, 1.0);
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
    /// use vex::Vec2;
    /// let actual = Vec2::from(1.0, 2.0);
    /// assert!(actual.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        math::is_valid(self.x) && math::is_valid(self.y)
    }

    fn print(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}, {}>", self.x, self.y)
    }
}

impl fmt::Debug for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

/// Looks up a component by index
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut v = Vec2::from(1.0, 2.0);
/// assert_eq!(v[0], 1.0);
/// assert_eq!(v[1], 2.0);
/// ```
impl ops::Index<u32> for Vec2 {
    type Output = f32;

    fn index(&self, index: u32) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Invalid index for Vec2: {}", index),
        }
    }
}

/// Mutate a component by index
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut v = Vec2::new();
/// v[0] = 3.0;
/// v[1] = 4.0;
/// assert_eq!(v[0], 3.0);
/// assert_eq!(v[1], 4.0);
/// ```
impl ops::IndexMut<u32> for Vec2 {
    fn index_mut<'a>(&'a mut self, index: u32) -> &'a mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Invalid index for Vec2: {}", index),
        }
    }
}

/// Negates all components in a vector
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let actual = -Vec2::from(1.0, 2.0);
/// let expected = Vec2::from(-1.0, -2.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Vec2 {
        Vec2::from(-self.x, -self.y)
    }
}

/// Find the resulting vector by adding a scalar to a vector's components
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let actual = Vec2::from(1.0, 2.0) + 1.0;
/// let expected = Vec2::from(2.0, 3.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Add<f32> for Vec2 {
    type Output = Vec2;

    fn add(self, _rhs: f32) -> Vec2 {
        Vec2::from(self.x + _rhs, self.y + _rhs)
    }
}

/// Add two vectors
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let a = Vec2::from(1.0, 2.0);
/// let b = Vec2::from(3.0, 4.0);
/// let actual = a + b;
/// let expected = Vec2::from(4.0, 6.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, _rhs: Vec2) -> Vec2 {
        Vec2::from(self.x + _rhs.x, self.y + _rhs.y)
    }
}

/// Increment a vector by a scalar
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut actual = Vec2::from(1.0, 2.0);
/// actual += 10.0;
/// let expected = Vec2::from(11.0, 12.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::AddAssign<f32> for Vec2 {
    fn add_assign(&mut self, _rhs: f32) {
        self.x += _rhs;
        self.y += _rhs;
    }
}

/// Increment a vector by another vector
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut actual = Vec2::from(1.0, 2.0);
/// actual += Vec2::from(1.0, 2.0);
/// let expected = Vec2::from(2.0, 4.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::AddAssign<Vec2> for Vec2 {
    fn add_assign(&mut self, _rhs: Vec2) {
        self.x += _rhs.x;
        self.y += _rhs.y;
    }
}

/// Find the resulting vector by subtracting a scalar from a vector's components
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let actual = Vec2::from(1.0, 2.0) - 10.0;
/// let expected = Vec2::from(-9.0, -8.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Sub<f32> for Vec2 {
    type Output = Vec2;

    fn sub(self, _rhs: f32) -> Vec2 {
        Vec2::from(self.x - _rhs, self.y - _rhs)
    }
}

/// Subtract two vectors
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let a = Vec2::from(1.0, 2.0);
/// let b = Vec2::from(4.0, 3.0);
/// let actual = a - b;
/// let expected = Vec2::from(-3.0, -1.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Sub<Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, _rhs: Vec2) -> Vec2 {
        Vec2::from(self.x - _rhs.x, self.y - _rhs.y)
    }
}

/// Decrement a vector by a scalar
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut actual = Vec2::from(1.0, 2.0);
/// actual -= 1.0;
/// let expected = Vec2::from(0.0, 1.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::SubAssign<f32> for Vec2 {
    fn sub_assign(&mut self, _rhs: f32) {
        self.x -= _rhs;
        self.y -= _rhs;
    }
}

/// Decrement a vector by another vector
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut actual = Vec2::from(1.0, 2.0);
/// actual -= Vec2::from(1.0, 2.0);
/// assert_eq!(actual, Vec2::new());
/// ```
impl ops::SubAssign<Vec2> for Vec2 {
    fn sub_assign(&mut self, _rhs: Vec2) {
        self.x -= _rhs.x;
        self.y -= _rhs.y;
    }
}

/// Find the resulting vector by multiplying a scalar to a vector's components
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let actual = Vec2::from(1.0, 2.0) * 2.0;
/// let expected = Vec2::from(2.0, 4.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, _rhs: f32) -> Vec2 {
        Vec2::from(self.x * _rhs, self.y * _rhs)
    }
}

/// Multiply two vectors
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let a = Vec2::from(1.0, 2.0);
/// let b = Vec2::from(2.0, 3.0);
/// let actual = a * b;
/// let expected = Vec2::from(2.0, 6.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Mul<Vec2> for Vec2 {
    type Output = Vec2;

    fn mul(self, _rhs: Vec2) -> Vec2 {
        Vec2::from(self.x * _rhs.x, self.y * _rhs.y)
    }
}

/// Multiply a vector by a scalar
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut actual = Vec2::from(1.0, 2.0);
/// actual *= 2.0;
/// let expected = Vec2::from(2.0, 4.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, _rhs: f32) {
        self.x *= _rhs;
        self.y *= _rhs;
    }
}

/// Multiply a vector by another vector
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut actual = Vec2::from(1.0, 2.0);
/// actual *= Vec2::from(2.0, 3.0);
/// let expected = Vec2::from(2.0, 6.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::MulAssign<Vec2> for Vec2 {
    fn mul_assign(&mut self, _rhs: Vec2) {
        self.x *= _rhs.x;
        self.y *= _rhs.y;
    }
}

/// Find the resulting vector by dividing a scalar to a vector's components
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let actual = Vec2::from(1.0, 2.0) / 2.0;
/// let expected = Vec2::from(0.5, 1.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::Div<f32> for Vec2 {
    type Output = Vec2;

    fn div(self, _rhs: f32) -> Vec2 {
        Vec2::from(self.x / _rhs, self.y / _rhs)
    }
}

/// Divide two vectors
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let a = Vec2::from(1.0, 2.0);
/// let b = Vec2::from(2.0, 8.0);
/// let actual = a / b;
/// let expected = Vec2::from(0.5, 0.25);
/// assert_eq!(actual, expected);
/// ```
impl ops::Div<Vec2> for Vec2 {
    type Output = Vec2;

    fn div(self, _rhs: Vec2) -> Vec2 {
        Vec2::from(self.x / _rhs.x, self.y / _rhs.y)
    }
}

/// Divide a vector by a scalar
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut actual = Vec2::from(1.0, 2.0);
/// actual /= 2.0;
/// let expected = Vec2::from(0.5, 1.0);
/// assert_eq!(actual, expected);
/// ```
impl ops::DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, _rhs: f32) {
        self.x /= _rhs;
        self.y /= _rhs;
    }
}

/// Divide a vector by another vector
///
/// # Examples
/// ```
/// use vex::Vec2;
/// let mut actual = Vec2::from(1.0, 2.0);
/// actual /= Vec2::from(2.0, 8.0);
/// let expected = Vec2::from(0.5, 0.25);
/// assert_eq!(actual, expected);
/// ```
impl ops::DivAssign<Vec2> for Vec2 {
    fn div_assign(&mut self, _rhs: Vec2) {
        self.x /= _rhs.x;
        self.y /= _rhs.y;
    }
}

/// Determines if two vectors components are equivalent
///
/// # Examples
/// ```
/// use vex::Vec2;
/// assert!(Vec2::new() == Vec2::new());
/// ```
impl cmp::PartialEq for Vec2 {
    fn eq(&self, _rhs: &Vec2) -> bool {
        self.x == _rhs.x && self.y == _rhs.y
    }
}
