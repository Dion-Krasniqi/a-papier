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
    let sum = add_forward(&t,&b);
    add_backward(&sum, &t, &b);
    &t.print();

    let b1 = Tensor::tensor_one(2.0,[2,3].to_vec());
    let b2 = Tensor::tensor_one(1.0,[3,1].to_vec());
    &matmul_forward(&b1, &b2).print();
    

}