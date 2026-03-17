use std::ops::{Add, Mul, Div};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

mod scalar;
use crate::scalar::neuron::definitions::*;

fn main(){
    let n = Neuron::new(5);
    let O = &n.forward(vec![0.002,0.02,0.01,0.01,0.02]);
    for w in &n.weights {
        w.print();
    };
    &O.backward();
    for w in &n.weights {
        w.print();
    };

}