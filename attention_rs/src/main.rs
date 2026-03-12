use std::ops::{Add, Mul};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    let a = Val::new(4.0);
    let b = Val::new(4.0);
    let c = a + b;
    let d = Val::new(4.0) * c;
    &d.print();
    let e = d + Val::new(3.0);
    &e.print();

}



// move this to its own file but here for now

enum Op {
    Add,
    Mul,
    Leaf,
}
impl fmt::Display for Op{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Add => write!(f, "+"),
            Op::Mul => write!(f, "*"),
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
    fn child(self) {
        for child in &self.0.borrow().children {
            println!("1");
        }
    }
}

impl Add for ValRef {
    type Output = ValRef;
    fn add(self, other: ValRef) -> ValRef {
        ValRef(Rc::new(RefCell::new(Val {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            opp: Op::Add,
            children: vec![ValRef(Rc::clone(&self.0)), 
                           ValRef(Rc::clone(&other.0))],
        })))
    }
}

impl Mul for ValRef {
    type Output = ValRef;
    fn mul(self, other: ValRef) -> ValRef {
        ValRef(Rc::new(RefCell::new( Val {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            opp: Op::Mul,
            children: vec![ValRef(Rc::clone(&self.0)), 
                           ValRef(Rc::clone(&other.0))],
        })))
    }
}