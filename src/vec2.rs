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

pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

impl Vec2 {
    pub fn new() -> Vec2 {
        Vec2 { x: 0.0, y: 0.0 }
    }

    pub fn from(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    pub fn dot(a: &Vec2, b: &Vec2) -> f32 {
        a.x * b.x + a.y * b.y
    }

    pub fn cross_vec(a: &Vec2, b: &Vec2) -> f32 {
        a.x * b.y - a.y * b.x
    }

    pub fn cross_pre_scalar(s: f32, v: &Vec2) -> Vec2 {
        Vec2::from(-s * v.y, s * v.x)
    }

    pub fn cross_post_scalar(v: &Vec2, s: f32) -> Vec2 {
        Vec2::from(s * v.y, -s * v.x)
    }

    pub fn min(a: &Vec2, b: &Vec2) -> Vec2 {
        Vec2::from(a.x.min(b.x), a.y.min(b.y))
    }

    pub fn max(a: &Vec2, b: &Vec2) -> Vec2 {
        Vec2::from(a.x.max(b.x), a.y.max(b.y))
    }

    pub fn clamp(v: &Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
        let low = Self::min(a, b);
        let high = Self::max(a, b);
        Self::max(&low, &Self::min(v, &high))
    }

    pub fn set(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
    }

    pub fn zero(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }

    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    pub fn magnitude_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

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

    pub fn abs(&mut self) {
        self.x = self.x.abs();
        self.y = self.y.abs();
    }

    pub fn skew(&mut self) {
        let x = self.x;
        self.x = -self.y;
        self.y = x;
    }

    pub fn is_valid(&self) -> bool {
        math::is_valid(self.x) && math::is_valid(self.y)
    }

    fn print(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}, {}>", self.x, self.y)
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl fmt::Debug for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl ops::Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Vec2 {
        Vec2::from(-self.x, -self.y)
    }
}

impl ops::Index<u32> for Vec2 {
    type Output = f32;

    fn index(&self, index: u32) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Invalid index for Vec2"),
        }
    }
}

impl ops::IndexMut<u32> for Vec2 {
    fn index_mut<'a>(&'a mut self, index: u32) -> &'a mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Invalid index for Vec2"),
        }
    }
}

impl ops::Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, _rhs: Vec2) -> Vec2 {
        Vec2::from(self.x + _rhs.x, self.y + _rhs.y)
    }
}

impl ops::AddAssign<f32> for Vec2 {
    fn add_assign(&mut self, _rhs: f32) {
        self.x += _rhs;
        self.y += _rhs;
    }
}

impl ops::AddAssign<Vec2> for Vec2 {
    fn add_assign(&mut self, _rhs: Vec2) {
        self.x += _rhs.x;
        self.y += _rhs.y;
    }
}

impl ops::Sub<Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, _rhs: Vec2) -> Vec2 {
        Vec2::from(self.x - _rhs.x, self.y - _rhs.y)
    }
}

impl ops::SubAssign<f32> for Vec2 {
    fn sub_assign(&mut self, _rhs: f32) {
        self.x -= _rhs;
        self.y -= _rhs;
    }
}

impl ops::SubAssign<Vec2> for Vec2 {
    fn sub_assign(&mut self, _rhs: Vec2) {
        self.x -= _rhs.x;
        self.y -= _rhs.y;
    }
}

impl ops::Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, _rhs: f32) -> Vec2 {
        Vec2::from(self.x * _rhs, self.y * _rhs)
    }
}

impl ops::Mul<Vec2> for Vec2 {
    type Output = Vec2;

    fn mul(self, _rhs: Vec2) -> Vec2 {
        Vec2::from(self.x * _rhs.x, self.y * _rhs.y)
    }
}

impl ops::MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, _rhs: f32) {
        self.x *= _rhs;
        self.y *= _rhs;
    }
}

impl ops::MulAssign<Vec2> for Vec2 {
    fn mul_assign(&mut self, _rhs: Vec2) {
        self.x *= _rhs.x;
        self.y *= _rhs.y;
    }
}

impl ops::Div<f32> for Vec2 {
    type Output = Vec2;

    fn div(self, _rhs: f32) -> Vec2 {
        Vec2::from(self.x / _rhs, self.y / _rhs)
    }
}

impl ops::Div<Vec2> for Vec2 {
    type Output = Vec2;

    fn div(self, _rhs: Vec2) -> Vec2 {
        Vec2::from(self.x / _rhs.x, self.y / _rhs.y)
    }
}

impl ops::DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, _rhs: f32) {
        self.x /= _rhs;
        self.y /= _rhs;
    }
}

impl ops::DivAssign<Vec2> for Vec2 {
    fn div_assign(&mut self, _rhs: Vec2) {
        self.x /= _rhs.x;
        self.y /= _rhs.y;
    }
}

impl cmp::PartialEq for Vec2 {
    fn eq(&self, _rhs: &Vec2) -> bool {
        self.x == _rhs.x && self.y == _rhs.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let actual = Vec2::new();
        let expected = Vec2 { x: 0.0, y: 0.0 };
        assert_eq!(actual, expected);
    }

    #[test]
    fn from() {
        let actual = Vec2::from(1.0, 2.0);
        let expected = Vec2 { x: 1.0, y: 2.0 };
        assert_eq!(actual, expected);
    }

    #[test]
    fn dot() {
        let a = Vec2::from(1.0, 0.0);
        let b = Vec2::from(0.0, 1.0);
        let actual = Vec2::dot(&a, &b);
        let expected = 0.0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn cross_vec() {
        let a = Vec2::from(1.0, 0.0);
        let b = Vec2::from(0.0, 1.0);
        let actual = Vec2::cross_vec(&a, &b);
        let expected = 1.0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn cross_pre_scalar() {
        let s = 1.0;
        let v = Vec2::from(1.0, 0.0);
        let actual = Vec2::cross_pre_scalar(s, &v);
        let expected = Vec2::from(0.0, 1.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn cross_post_scalar() {
        let s = 1.0;
        let v = Vec2::from(1.0, 0.0);
        let actual = Vec2::cross_post_scalar(&v, s);
        let expected = Vec2::from(0.0, -1.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn min() {
        let a = Vec2::from(1.0, 4.0);
        let b = Vec2::from(2.0, 3.0);
        let actual = Vec2::min(&a, &b);
        let expected = Vec2::from(1.0, 3.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn max() {
        let a = Vec2::from(1.0, 4.0);
        let b = Vec2::from(2.0, 3.0);
        let actual = Vec2::max(&a, &b);
        let expected = Vec2::from(2.0, 4.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn clamp() {
        let a = Vec2::from(1.0, 3.0);
        let b = Vec2::from(2.0, 4.0);
        let c = Vec2::from(0.0, 5.0);
        let actual = Vec2::clamp(&c, &a, &b);
        let expected = Vec2::from(1.0, 4.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn set() {
        let mut actual = Vec2::new();
        actual.set(1.0, 2.0);
        let expected = Vec2::from(1.0, 2.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn zero() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual.zero();
        assert_eq!(actual, ZERO);
    }

    #[test]
    fn magnitude() {
        let actual = Vec2::from(1.0, 2.0).magnitude();
        let expected = 2.2360679775;
        assert_eq!(actual, expected);
    }

    #[test]
    fn magnitude_squared() {
        let actual = Vec2::from(1.0, 2.0).magnitude_squared();
        let expected = 5.0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn normalize() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual.normalize();
        let expected = Vec2::from(0.4472135955, 0.894427191);
        assert_eq!(actual, expected);
    }

    #[test]
    fn abs() {
        let mut actual = Vec2::from(-1.0, -2.0);
        actual.abs();
        let expected = Vec2::from(1.0, 2.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn skew() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual.skew();
        let expected = Vec2::from(-2.0, 1.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn is_valid() {
        let actual = Vec2::from(1.0, 2.0);
        assert!(actual.is_valid());
    }

    #[test]
    fn op_neg() {
        let actual = -Vec2::from(1.0, 2.0);
        let expected = Vec2::from(-1.0, -2.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn index() {
        let mut v = Vec2::from(1.0, 2.0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);

        v[0] = 3.0;
        v[1] = 4.0;
        assert_eq!(v[0], 3.0);
        assert_eq!(v[1], 4.0);
    }

    #[test]
    fn add() {
        let a = Vec2::from(1.0, 2.0);
        let b = Vec2::from(3.0, 4.0);
        let actual = a + b;
        let expected = Vec2::from(4.0, 6.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn add_assign_scalar() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual += 1.0;
        let expected = Vec2::from(2.0, 3.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn add_assign_vec() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual += Vec2::from(1.0, 2.0);
        let expected = Vec2::from(2.0, 4.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn sub() {
        let a = Vec2::from(1.0, 2.0);
        let b = Vec2::from(4.0, 3.0);
        let actual = a - b;
        let expected = Vec2::from(-3.0, -1.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn sub_assign_scalar() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual -= 1.0;
        let expected = Vec2::from(0.0, 1.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn sub_assign_vec() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual -= Vec2::from(1.0, 2.0);
        assert_eq!(actual, ZERO);
    }

    #[test]
    fn mul_scalar() {
        let v = Vec2::from(1.0, 2.0);
        let actual = v * 2.0;
        let expected = Vec2::from(2.0, 4.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn mul_vec2() {
        let actual = Vec2::from(1.0, 2.0) * Vec2::from(2.0, 3.0);
        let expected = Vec2::from(2.0, 6.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn mul_assign_scalar() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual *= 2.0;
        let expected = Vec2::from(2.0, 4.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn mul_assign_vec() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual *= Vec2::from(2.0, 3.0);
        let expected = Vec2::from(2.0, 6.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn div_scalar() {
        let v = Vec2::from(1.0, 2.0);
        let actual = v / 2.0;
        let expected = Vec2::from(0.5, 1.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn div_vec2() {
        let actual = Vec2::from(1.0, 2.0) / Vec2::from(2.0, 8.0);
        let expected = Vec2::from(0.5, 0.25);
        assert_eq!(actual, expected);
    }

    #[test]
    fn div_assign_scalar() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual /= 2.0;
        let expected = Vec2::from(0.5, 1.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn div_assign_vec() {
        let mut actual = Vec2::from(1.0, 2.0);
        actual /= Vec2::from(2.0, 8.0);
        let expected = Vec2::from(0.5, 0.25);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eq() {
        assert!(Vec2::new() == ZERO);
    }
}
