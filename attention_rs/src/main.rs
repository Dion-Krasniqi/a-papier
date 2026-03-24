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
    let t1 = Tensor::tensor([2,4].to_vec());
    let t2 = Tensor::tensor([2,4].to_vec());
    let ot = add_forward(&t1,&t2);
    let b = Tensor::tensor([4,3].to_vec());
    let bot = matmul_forward(&ot, &b);
    let loss = tanh_forward(&bot);
    set_grad(&loss, 1.0);
    tanh_backward(&loss, &bot);
    matmul_backward(&bot, &b, &ot);
    add_backward(&ot, &t1, &t2);
    &loss.print();
    &bot.print();
    &b.print();
    &ot.print();
    &t1.print();
    &t2.print();    

}