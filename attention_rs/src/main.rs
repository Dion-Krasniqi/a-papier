use std::ops::{Add, Mul};

fn main() {
    let a = Val::new(4.0);
    let b = Val::new(2.0);
    let c = a * b;
    println!("{}", c.data );
}



// move this to its own file but here for now
pub struct Val {
    data: f32,
    grad: f32
}

impl Val {
    fn new(value: f32) -> Val {
        Val {
            data: value,
            grad: 0.0,
        }
    }
}

impl Add for Val {
    type Output = Val;
    fn add(self, other: Val) -> Val {
        Val {
            data: self.data + other.data,
            grad: 0.0,
        }
    }
}

impl Mul for Val {
    type Output = Val;
    fn mul(self, other: Val) -> Val {
        Val {
            data: self.data * other.data,
            grad: 0.0,
        }
    }
}