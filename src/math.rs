pub fn is_valid(x: f32) -> bool {
    !(x.is_nan() || x.is_infinite())
}

pub fn next_power_of_two(x: &i32) -> i32 {
    let mut r = x.clone();
    r |= r >> 1;
    r |= r >> 2;
    r |= r >> 4;
    r |= r >> 8;
    r |= r >> 16;
    r + 1
}

pub fn is_power_of_two(x: i32) -> bool {
    x > 0 && (x & (x - 1)) == 0
}
