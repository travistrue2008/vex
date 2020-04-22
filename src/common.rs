#[inline]
pub fn is_valid(x: f32) -> bool {
    !(x.is_nan() || x.is_infinite())
}

/// Gets the next power of two for a given value
///
/// # Examples
/// ```
/// use vex::next_power_of_two;
/// let n = next_power_of_two(1);
/// assert_eq!(n, 2);
/// ```
/// ```
/// use vex::next_power_of_two;
/// let n = next_power_of_two(2);
/// assert_eq!(n, 4);
/// ```
#[inline]
pub fn next_power_of_two(x: i32) -> i32 {
    let mut r = x;
    r |= r >> 1;
    r |= r >> 2;
    r |= r >> 4;
    r |= r >> 8;
    r |= r >> 16;
    r + 1
}

/// Determines whether or not a given value is a power of two
///
/// # Examples
/// ```
/// use vex::is_power_of_two;
/// let n = is_power_of_two(1);
/// assert_eq!(n, true);
/// ```
/// ```
/// use vex::is_power_of_two;
/// let n = is_power_of_two(2);
/// assert_eq!(n, true);
/// ```
/// ```
/// use vex::is_power_of_two;
/// let n = is_power_of_two(3);
/// assert_eq!(n, false);
/// ```
#[inline]
pub fn is_power_of_two(x: i32) -> bool {
    x > 0 && (x & (x - 1)) == 0
}

/// Returns 1 or -1 depending on the sign of the input value
///
/// # Examples
/// ```
/// use vex::sign;
/// let mul = sign(1234.0);
/// assert_eq!(mul, 1.0);
/// ```
/// ```
/// use vex::sign;
/// let mul = sign(-1234.0);
/// assert_eq!(mul, -1.0);
/// ```
#[inline]
pub fn sign(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

pub trait TransformPoint<T> {
    fn transform_point(&self, point: &T) -> T;
}
