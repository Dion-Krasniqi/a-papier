use std::ops::{Add, Mul};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    let x1 = &Val::new(2.0);
    let x2 = &Val::new(0.0);
    let w1 = &Val::new(-3.0);
    let w2 = &Val::new(1.0);
    let b = &Val::new(6.881373);
    let x1w1 = x1*w1;
    let x2w2 = x2*w2;
    let x1w1x2w2 = &x1w1 + &x2w2;
    let n = &x1w1x2w2 + b;
    let mut O = ValRef::tanh(&n);

    // manual back prop
    O.get_grad();
    O.set_grad(1.0);
    n.set_grad(1.0-O.0.borrow().data.powf(2.0));
    x1w1x2w2.set_grad(1.0 * n.0.borrow().grad);
    b.set_grad(1.0 * n.0.borrow().grad);
    x1w1.set_grad(1.0 * x1w1x2w2.0.borrow().grad);
    x2w2.set_grad(1.0 * x1w1x2w2.0.borrow().grad);
    x1.set_grad(w1.0.borrow().data * /*cuz chain rule*/ x1w1.0.borrow().grad);
    w1.set_grad(x1.0.borrow().data * x1w1.0.borrow().grad);
    x2.set_grad(w2.0.borrow().data * x2w2.0.borrow().grad);
    w2.set_grad(x2.0.borrow().data * x2w2.0.borrow().grad);

    O.print();
    n.print();
    x1w1x2w2.print();
    b.print();
    x1.print();
    w1.print();
    x2.print();
    w2.print();

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
pub struct Val {
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
        println!("({},{},{})", self.0.borrow().data,
                               self.0.borrow().grad,
                               self.0.borrow().opp);
    }
    // just playing
    fn child(&self) {
        for child in &self.0.borrow().children {
            &child.print();
        }
    }
    fn get_grad(&self) {
        println!("{}", self.0.borrow().grad);
    }
    fn set_grad(&self, val: f32) {
        self.0.borrow_mut().grad = val;
    }
    // tanh with direct tanh
    fn tanh(value: &ValRef) -> ValRef {
        ValRef(Rc::new(RefCell::new(
            Val {
                data: value.0.borrow().data.tanh(),
                grad: 0.0,
                opp: Op::Tanh,
                children: vec![ValRef(Rc::clone(&value.0))],
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