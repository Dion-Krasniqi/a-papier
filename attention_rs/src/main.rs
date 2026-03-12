use std::ops::{Add, Mul};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    let a: ValRef = Rc::new(RefCell::new(Val::new(4.0)));
    let b: ValRef = Rc::new(RefCell::new(Val::new(4.0)));
    let c = a + b;
    println!("{}", c.data );
    println!("{}", c.opp);

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

type ValRef = Rc<RefCell<Val>>;
struct Val {
    data: f32,
    grad: f32,
    opp: Op,
    children: Vec<ValRef>, 
    // the naming kinda confusing to me, also type so its mut and shared 
                                    
}

impl Val {
    fn new(value: f32) -> Val {
        Val {
                data: value,
                grad: 0.0,
                opp: Op::Leaf,
                children: Vec::new(),
        }
    }
}

impl Add for ValRef {
    type Output = ValRef;
    fn add(self, other: ValRef) -> ValRef {
        Rc::new(RefCell::new(
            Val {
                data: self.borrow().data + other.borrow().data,
                grad: 0.0,
                opp: Op::Add,
                children: vec![Rc::clone(&self), Rc::clone(&other)],
            }
        ))
    }
}

impl Mul for Val {
    type Output = Val;
    fn mul(self, other: Val) -> Val {
        Val {
            data: self.data * other.data,
            grad: 0.0,
            opp: Op::Mul,
            children: self.children,
        }
    }
}