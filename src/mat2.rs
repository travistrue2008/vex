use super::math;
use super::vec2::Vec2;
use std::cmp;
use std::fmt;
use std::ops;

#[derive(Copy, Clone)]
pub struct Mat2 {
    m: [f32; 4],
}

impl Mat2 {
    /// Creates a new matrix
    ///
    /// # Examples
    /// ```
    /// use vex::Vec4;
    /// let actual = Vec4::new();
    /// let expected = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    /// assert_eq!(actual, expected);
    /// ```
    pub fn new() -> Mat2 {
        Mat2 {
            m: [1.0, 0.0, 1.0, 0.0],
        }
    }

    pub fn construct(m11: f32, m21: f32, m12: f32, m22: f32) -> Mat2 {
        Mat2 {
            m: [m11, m21, m12, m22],
        }
    }

    pub fn set_m11(&mut self, v: f32) {
        self.m[0] = v;
    }

    pub fn set_m21(&mut self, v: f32) {
        self.m[1] = v;
    }

    pub fn set_m12(&mut self, v: f32) {
        self.m[2] = v;
    }

    pub fn set_m22(&mut self, v: f32) {
        self.m[3] = v;
    }

    pub fn m11(&self) -> f32 {
        self.m[0]
    }

    pub fn m21(&self) -> f32 {
        self.m[1]
    }

    pub fn m12(&self) -> f32 {
        self.m[2]
    }

    pub fn m22(&self) -> f32 {
        self.m[3]
    }

    pub fn set(&mut self, m11: f32, m12: f32, m21: f32, m22: f32) {
        self.m[0] = m11;
        self.m[1] = m21;
        self.m[2] = m12;
        self.m[3] = m22;
    }

    pub fn identity(&mut self) {
        self.m[0] = 1.0;
        self.m[1] = 0.0;
        self.m[2] = 0.0;
        self.m[3] = 1.0;
    }

    pub fn transpose(&mut self) {
        let temp = self.m[1];
        self.m[1] = self.m[2];
        self.m[2] = temp;
    }

    pub fn determinant(&self) -> f32 {
        (self.m[0] * self.m[3]) - (self.m[2] * self.m[1])
    }

    pub fn inverse(&mut self) -> bool {
        let det = self.determinant();
        if det == 0.0 {
            return false;
        }

        let inv_det = 1.0 / det;
        let values = [self.m[3], -self.m[1], -self.m[2], self.m[0]];
        for (i, elem) in self.m.iter_mut().enumerate() {
            *elem = values[i] * inv_det;
        }

        true
    }

    pub fn is_valid(&self) -> bool {
        math::is_valid(self.m[0])
            && math::is_valid(self.m[1])
            && math::is_valid(self.m[2])
            && math::is_valid(self.m[3])
    }

    fn print(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}, {}]\n[{}, {}]",
            self.m[0], self.m[1], self.m[2], self.m[3]
        )
    }
}

impl ops::Add<f32> for Mat2 {
    type Output = Mat2;

    fn add(self, _rhs: f32) -> Mat2 {
        Mat2::construct(
            self.m[0] + _rhs,
            self.m[1] + _rhs,
            self.m[2] + _rhs,
            self.m[3] + _rhs,
        )
    }
}

impl ops::Add<Mat2> for Mat2 {
    type Output = Mat2;

    fn add(self, _rhs: Mat2) -> Mat2 {
        Mat2::construct(
            self.m[0] + _rhs.m[0],
            self.m[1] + _rhs.m[1],
            self.m[2] + _rhs.m[2],
            self.m[3] + _rhs.m[3],
        )
    }
}

impl ops::AddAssign<f32> for Mat2 {
    fn add_assign(&mut self, _rhs: f32) {
        self.m[0] += _rhs;
        self.m[1] += _rhs;
        self.m[2] += _rhs;
        self.m[3] += _rhs;
    }
}

impl ops::AddAssign<Mat2> for Mat2 {
    fn add_assign(&mut self, _rhs: Mat2) {
        self.m[0] += _rhs.m[0];
        self.m[1] += _rhs.m[1];
        self.m[2] += _rhs.m[2];
        self.m[3] += _rhs.m[3];
    }
}

impl ops::Sub<f32> for Mat2 {
    type Output = Mat2;

    fn sub(self, _rhs: f32) -> Mat2 {
        Mat2::construct(
            self.m[0] - _rhs,
            self.m[1] - _rhs,
            self.m[2] - _rhs,
            self.m[3] - _rhs,
        )
    }
}

impl ops::Sub<Mat2> for Mat2 {
    type Output = Mat2;

    fn sub(self, _rhs: Mat2) -> Mat2 {
        Mat2::construct(
            self.m[0] - _rhs.m[0],
            self.m[1] - _rhs.m[1],
            self.m[2] - _rhs.m[2],
            self.m[3] - _rhs.m[3],
        )
    }
}

impl ops::SubAssign<f32> for Mat2 {
    fn sub_assign(&mut self, _rhs: f32) {
        self.m[0] -= _rhs;
        self.m[1] -= _rhs;
        self.m[2] -= _rhs;
        self.m[3] -= _rhs;
    }
}

impl ops::SubAssign<Mat2> for Mat2 {
    fn sub_assign(&mut self, _rhs: Mat2) {
        self.m[0] -= _rhs.m[0];
        self.m[1] -= _rhs.m[1];
        self.m[2] -= _rhs.m[2];
        self.m[3] -= _rhs.m[3];
    }
}

impl ops::Mul<f32> for Mat2 {
    type Output = Mat2;

    fn mul(self, _rhs: f32) -> Mat2 {
        Mat2::construct(
            self.m[0] * _rhs,
            self.m[1] * _rhs,
            self.m[2] * _rhs,
            self.m[3] * _rhs,
        )
    }
}

impl ops::Mul<Vec2> for Mat2 {
    type Output = Vec2;

    fn mul(self, _rhs: Vec2) -> Vec2 {
        Vec2::construct(
            self.m[0] * _rhs.x + self.m[2] * _rhs.y,
            self.m[1] * _rhs.x + self.m[3] * _rhs.y,
        )
    }
}

impl ops::Mul<Mat2> for Mat2 {
    type Output = Mat2;

    fn mul(self, _rhs: Mat2) -> Mat2 {
        Mat2 {
            m: [
                (self.m[0] * _rhs.m[0]) + (self.m[2] * _rhs.m[1]),
                (self.m[1] * _rhs.m[0]) + (self.m[3] * _rhs.m[1]),
                (self.m[0] * _rhs.m[2]) + (self.m[2] * _rhs.m[3]),
                (self.m[1] * _rhs.m[2]) + (self.m[3] * _rhs.m[3]),
            ],
        }
    }
}

impl ops::MulAssign<f32> for Mat2 {
    fn mul_assign(&mut self, _rhs: f32) {
        self.m[0] *= _rhs;
        self.m[1] *= _rhs;
        self.m[2] *= _rhs;
        self.m[3] *= _rhs;
    }
}

impl ops::MulAssign<Mat2> for Mat2 {
    fn mul_assign(&mut self, _rhs: Mat2) {
        self.m[0] = (self.m[0] * _rhs.m[0]) + (self.m[2] * _rhs.m[1]);
        self.m[1] = (self.m[1] * _rhs.m[0]) + (self.m[3] * _rhs.m[1]);
        self.m[2] = (self.m[0] * _rhs.m[2]) + (self.m[2] * _rhs.m[3]);
        self.m[3] = (self.m[1] * _rhs.m[2]) + (self.m[3] * _rhs.m[3]);
    }
}

impl ops::Div<f32> for Mat2 {
    type Output = Mat2;

    fn div(self, _rhs: f32) -> Mat2 {
        Mat2::construct(
            self.m[0] / _rhs,
            self.m[1] / _rhs,
            self.m[2] / _rhs,
            self.m[3] / _rhs,
        )
    }
}

impl ops::Div<Mat2> for Mat2 {
    type Output = Mat2;

    fn div(self, _rhs: Mat2) -> Mat2 {
        let mut rhs_inv = _rhs.clone();
        rhs_inv.inverse();
        self * rhs_inv
    }
}

impl ops::DivAssign<f32> for Mat2 {
    fn div_assign(&mut self, _rhs: f32) {
        self.m[0] /= _rhs;
        self.m[1] /= _rhs;
        self.m[2] /= _rhs;
        self.m[3] /= _rhs;
    }
}

impl ops::DivAssign<Mat2> for Mat2 {
    fn div_assign(&mut self, _rhs: Mat2) {
        let lhs = self.clone();
        let mut rhs = _rhs.clone();
        rhs.inverse();
        self.m = (lhs * rhs).m;
    }
}

impl cmp::PartialEq for Mat2 {
    fn eq(&self, _rhs: &Mat2) -> bool {
        self.m[0] == _rhs.m[0]
            && self.m[1] == _rhs.m[1]
            && self.m[2] == _rhs.m[2]
            && self.m[3] == _rhs.m[3]
    }
}

impl fmt::Debug for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}

impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f)
    }
}
