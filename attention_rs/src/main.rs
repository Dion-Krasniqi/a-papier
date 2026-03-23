use std::ops::{Add, Mul, Div};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

mod tensor;
mod scalar;
use crate::scalar::neuron::definitions::*;
use crate::scalar::neuron::definitions::NONL as NONL;
use crate::scalar::value::definitions::*;
use crate::tensor::tensor::*;

fn main(){
    let t = Tensor::tensor([2,4].to_vec());
    let b = Tensor::tensor_one(1.0,[2,4].to_vec());
    let sum = matadd_forward(&t,&b);
    matadd_backward(&sum, &t, &b);
    &t.print();

}