use std::ops::{Add, Mul, Div};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

fn main() {
    // example
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

    // end of manual back prop
    // Same example using backward function
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
    O.backward();
    O.print();
    n.print();
    x1w1x2w2.print();
    b.print();
    x1.print();
    w1.print();
    x2.print();
    w2.print();

}

enum Op {
    Add,
    Mul,
    Pow,
    Tanh,
    ReLU,
    Exp,
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

pub struct Val {
    data: f32,
    grad: f32,
    opp: Op,
    children: Vec<ValRef>, 
    // the naming kinda confusing to me, also type ValRef with Rc and Refcell so its mut and shared 
    backward: Rc<dyn Fn()>,

                                    
}
impl Val {
    pub fn new(value: f32) -> ValRef {
        ValRef(Rc::new(RefCell::new(
            Val {
                data: value,
                grad: 0.0,
                opp: Op::Leaf,
                children: Vec::new(),
                backward: Rc::new(|| {}),
            }
        )))
    }
}

pub struct ValRef(Rc<RefCell<Val>>);
// for hashmap
impl Hash for ValRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}
impl PartialEq for ValRef {
    fn eq(&self, other: &ValRef) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for ValRef {}

impl ValRef {
    pub fn print(&self) {
        println!("({},{},{})", self.0.borrow().data,
                               self.0.borrow().grad,
                               self.0.borrow().opp);
    }
    // just playing
    pub fn child(&self) {
        for child in &self.0.borrow().children {
            &child.print();
        }
    }
    pub fn get_grad(&self) -> f32 {
        self.0.borrow().grad
    }
    pub fn set_grad(&self, val: f32) {
        self.0.borrow_mut().grad = val;
    }
    pub fn get_data(&self) -> f32 {
        self.0.borrow().data
    }
    pub fn set_data(&self, val: f32) {
        self.0.borrow_mut().data = val;
    }
    fn topo_sort(&self, visited: &mut HashSet<ValRef>, topo: &mut Vec<ValRef>){
        if !(visited.contains(&self)) {
            visited.insert(ValRef(Rc::clone(&self.0)));
            for child in &self.0.borrow().children {
                &child.topo_sort(visited, topo);
            }
            topo.push(ValRef(Rc::clone(&self.0)));
        }
    }
    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        &self.topo_sort(&mut visited, &mut topo);
        &self.set_grad(1.0);
        for node in topo.iter().rev(){
            let f = Rc::clone(&node.0.borrow().backward);
            f();
        };
    }
    pub fn powf(&self, exp: f32) -> ValRef {
        let output = ValRef(Rc::new(RefCell::new(
            Val {
                data: self.0.borrow().data.powf(exp),
                grad: 0.0,
                opp: Op::Pow,
                children: vec![ValRef(Rc::clone(&self.0))],
                backward: Rc::new(|| {}),
            }
        )));
        // backward func
        let self_clone = Rc::clone(&self.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            let self_data = self_clone.borrow().data;
            self_clone.borrow_mut().grad += (exp * self_data.powf(exp-1.0)) * out_grad;
        });

        output
    }
    // exponential func
    pub fn expo(&self) -> ValRef {
        let output = ValRef(Rc::new(RefCell::new(
            Val {
                data: self.0.borrow().data.exp(),
                grad: 0.0,
                opp: Op::Exp,
                children: vec![ValRef(Rc::clone(&self.0))],
                backward: Rc::new(|| {}),
            }
        )));
        // backward func
        let self_clone = Rc::clone(&self.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            let out_data = out_clone.borrow().data;
            self_clone.borrow_mut().grad += out_data * out_grad;
        });

        output
    }
    // tanh with direct tanh
    pub fn tanh(value: &ValRef) -> ValRef {
        let output = ValRef(Rc::new(RefCell::new(
            Val {
                data: value.0.borrow().data.tanh(),
                grad: 0.0,
                opp: Op::Tanh,
                children: vec![ValRef(Rc::clone(&value.0))],
                backward: Rc::new(|| {}),
            }
        )));
        // backward func
        let self_clone = Rc::clone(&value.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            self_clone.borrow_mut().grad += (1.0 - (out_clone.borrow().data).powf(2.0)) * out_grad;
        });

        output
    }
    // composite tanh
    pub fn _tanh(&self) -> ValRef {
        // kinda funky
        let output = &(&(self*2.0).expo()+(-1.0))/
                     &(&(self*2.0).expo()+(1.0));
        // backward func
        let self_clone = Rc::clone(&self.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            self_clone.borrow_mut().grad += (1.0 - (out_clone.borrow().data).powf(2.0)) * out_grad;
        });

        output
    }
    pub fn _sigmoid(&self) -> ValRef {
        let output = 1.0/&(1.0 + &(self*(-1.0)).expo());
        // backward function
        let self_clone = Rc::clone(&self.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            let out_data =  out_clone.borrow().data;
            // f = 1/(1-e(-x)) =(d/dx)=> d(f)/dx = e(-x)/((1 - e(-x))^2)
            self_clone.borrow_mut().grad += out_data * (1.0 - out_data) * out_grad;
        });

        output
    }
    pub fn _relu(&self) -> ValRef {
        let output = ValRef(Rc::new(RefCell::new(Val {
            data: self.0.borrow().data.max(0.0),
            grad: 0.0,
            opp: Op::ReLU,
            children: vec![ValRef(Rc::clone(&self.0)),],
            backward: Rc::new(|| {}),
        })));
        // backward function
        let self_clone = Rc::clone(&self.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            let out_data =  out_clone.borrow().data;
            self_clone.borrow_mut().grad += if out_data>0.0 {1.0} else {0.0} * out_grad;
        });

        output
    }
}
//Addition
impl Add for &ValRef {
    type Output = ValRef;
    fn add(self, other: &ValRef) -> ValRef {
        let output = ValRef(Rc::new(RefCell::new(Val {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            opp: Op::Add,
            children: vec![ValRef(Rc::clone(&self.0)), 
                           ValRef(Rc::clone(&other.0))],
            backward: Rc::new(|| {}),
        })));
        // backward func
        let self_clone = Rc::clone(&self.0);
        let other_clone = Rc::clone(&other.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            self_clone.borrow_mut().grad += out_grad;
            other_clone.borrow_mut().grad += out_grad;
        });

        output
    }
}
impl Add<f32> for &ValRef {
    type Output = ValRef;
    fn add(self, rhs: f32) -> ValRef {
        let other = &Val::new(rhs);
        let output = ValRef(Rc::new(RefCell::new(Val {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            opp: Op::Add,
            children: vec![ValRef(Rc::clone(&self.0)), 
                           ValRef(Rc::clone(&other.0))],
            backward: Rc::new(|| {}),
        })));
        // backward func
        let self_clone = Rc::clone(&self.0);
        let other_clone = Rc::clone(&other.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            self_clone.borrow_mut().grad += out_grad;
            other_clone.borrow_mut().grad += out_grad;
        });

        output
    }

}
impl Add<&ValRef> for f32 {
    type Output = ValRef;
    fn add(self, other: &ValRef) -> ValRef {
        let lhs = &Val::new(self);
        let output = ValRef(Rc::new(RefCell::new(Val {
            data: lhs.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            opp: Op::Add,
            children: vec![ValRef(Rc::clone(&lhs.0)), 
                           ValRef(Rc::clone(&other.0))],
            backward: Rc::new(|| {}),
        })));
        // backward func
        let self_clone = Rc::clone(&lhs.0);
        let other_clone = Rc::clone(&other.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            self_clone.borrow_mut().grad += out_grad;
            other_clone.borrow_mut().grad += out_grad;
        });

        output
    }

}

//Multiplication
impl Mul for &ValRef {
    type Output = ValRef;
    fn mul(self, other: &ValRef) -> ValRef {
        let output = ValRef(Rc::new(RefCell::new( Val {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            opp: Op::Mul,
            children: vec![ValRef(Rc::clone(&self.0)), 
                           ValRef(Rc::clone(&other.0))],
            backward: Rc::new(|| {}),
        })));
        // backward func
        let self_clone = Rc::clone(&self.0);
        let other_clone = Rc::clone(&other.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            self_clone.borrow_mut().grad += other_clone.borrow().data * out_grad;
            other_clone.borrow_mut().grad += self_clone.borrow().data * out_grad;
        });

        output
    }
}
impl Mul<f32> for &ValRef {
    type Output = ValRef;
    fn mul(self, rhs: f32) -> ValRef {
        let other = &Val::new(rhs);
        let output = ValRef(Rc::new(RefCell::new( Val {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            opp: Op::Mul,
            children: vec![ValRef(Rc::clone(&self.0)), 
                           ValRef(Rc::clone(&other.0))],
            backward: Rc::new(|| {}),
        })));
        // backward func
        let self_clone = Rc::clone(&self.0);
        let other_clone = Rc::clone(&other.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            self_clone.borrow_mut().grad += other_clone.borrow().data * out_grad;
            other_clone.borrow_mut().grad += self_clone.borrow().data * out_grad;
        });

        output
    }
}
impl Mul<&ValRef> for f32 {
    type Output = ValRef;
    fn mul(self, other: &ValRef) -> ValRef {
        let lhs = &Val::new(self);
        let output = ValRef(Rc::new(RefCell::new( Val {
            data: lhs.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            opp: Op::Mul,
            children: vec![ValRef(Rc::clone(&lhs.0)), 
                           ValRef(Rc::clone(&other.0))],
            backward: Rc::new(|| {}),
        })));
        // backward func
        let self_clone = Rc::clone(&lhs.0);
        let other_clone = Rc::clone(&other.0);
        let out_clone = Rc::clone(&output.0);

        output.0.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            self_clone.borrow_mut().grad += other_clone.borrow().data * out_grad;
            other_clone.borrow_mut().grad += self_clone.borrow().data * out_grad;
        });

        output
    }
}
//Division
impl Div for &ValRef {
    type Output = ValRef;
    fn div(self, other: &ValRef) -> ValRef {
        self*&(other.powf(-1.0))
    }
}
impl Div<f32> for &ValRef {
    type Output = ValRef;
    fn div(self, denominator: f32) -> ValRef {
        let other = &Val::new(denominator);
        self*&(other.powf(-1.0))
    }
}
impl Div<&ValRef> for f32 {
    type Output = ValRef;
    fn div(self, denominator: &ValRef) -> ValRef {
        self*&(denominator.powf(-1.0))
    }
}

impl Clone for ValRef {
    fn clone(&self) -> ValRef {
        ValRef(Rc::clone(&self.0))
    }
}