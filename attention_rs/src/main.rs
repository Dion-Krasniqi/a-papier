use std::ops::{Add, Mul};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    let x1 = &Val::new(2.0);
    let x2 = &Val::new(0.0);
    let w1 = &Val::new(-3.0);
    let w2 = &Val::new(1.0);
    let b = &Val::new(6.7);
    let x1w1 = x1*w1;
    let x2w2 = x2*w2;
    let x1w1x2w2 = &x1w1 + &x2w2;
    let n = &x1w1x2w2 + b;
    let O = n.tanh();
    O.print();
    // do manual back prop

}



// move this to its own file but here for now

enum Op {
    Add,
    Mul,
    Tanh,
    Leaf,
}
impl fmt::Display for Op{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Add => write!(f, "+"),
            Op::Mul => write!(f, "*"),
            Op::Tanh => write!(f, "tanh"),
            _ => write!(f, ""),
        }
    }
}

struct ValRef(Rc<RefCell<Val>>);
struct Val {
    data: f32,
    grad: f32,
    opp: Op,
    children: Vec<ValRef>, 
    // the naming kinda confusing to me, also type so its mut and shared 
                                    
}

impl Val {
    fn new(value: f32) -> ValRef {
        ValRef(Rc::new(RefCell::new(
            Val {
                data: value,
                grad: 0.0,
                opp: Op::Leaf,
                children: Vec::new(),
            }
        )))
    }
}

impl ValRef {
    fn print(&self) {
        println!("({},{})", self.0.borrow().data,
                         self.0.borrow().opp);
    }
    // just playing
    fn child(&self) {
        for child in &self.0.borrow().children {
            &child.print();
        }
    }
    // tanh with direct tanh
    fn tanh(&self) -> ValRef {
        ValRef(Rc::new(RefCell::new(
            Val {
                data: self.0.borrow().data.tanh(),
                grad: 0.0,
                opp: Op::Tanh,
                children: vec![ValRef(Rc::clone(&self.0))],
            }
        )))
    }
}

impl Add for &ValRef {
    type Output = ValRef;
    fn add(self, other: &ValRef) -> ValRef {
        ValRef(Rc::new(RefCell::new(Val {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            opp: Op::Add,
            children: vec![ValRef(Rc::clone(&self.0)), 
                           ValRef(Rc::clone(&other.0))],
        })))
    }
}

impl Mul for &ValRef {
    type Output = ValRef;
    fn mul(self, other: &ValRef) -> ValRef {
        ValRef(Rc::new(RefCell::new( Val {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            opp: Op::Mul,
            children: vec![ValRef(Rc::clone(&self.0)), 
                           ValRef(Rc::clone(&other.0))],
        })))
    }
}