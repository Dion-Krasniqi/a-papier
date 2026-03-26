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
    let t1 = Tensor::tensor_rand([2,4].to_vec());
    let t2 = Tensor::tensor_rand([2,4].to_vec());
    let ot = add_forward(&t1,&t2);
    let b = Tensor::tensor_rand([4,3].to_vec());
    let bot = matmul_forward(&ot, &b);
    let tot = tanh_forward(&bot);
    let loss = softmax_forward(&tot);
    &loss.set_grad(1.0);
    softmax_backward(&loss, &tot);
    tanh_backward(&tot, &bot);
    matmul_backward(&bot, &b, &ot);
    add_backward(&ot, &t1, &t2);
    &loss.print();
    &tot.print();
    &bot.print();
    &b.print();
    &ot.print();
    &t1.print();
    &t2.print();    

}