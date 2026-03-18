use std::ops::{Add, Mul, Div};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

mod scalar;
use crate::scalar::neuron::definitions::*;
use crate::scalar::value::definitions::*;

fn main(){

    let x = vec![Val::new(2.0),Val::new(3.0),Val::new(-1.0)];
    let L = MLP::new(3, [4,4,1].to_vec());
    let O = &L.forward(x);

}