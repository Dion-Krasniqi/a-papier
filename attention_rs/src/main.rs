use std::ops::{Add, Mul, Div};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

mod scalar;
use crate::scalar::neuron::definitions::*;
use crate::scalar::neuron::definitions::NONL as NONL;
use crate::scalar::value::definitions::*;

fn main(){

    // test
    let L = MLP::new(3, [4,4,1].to_vec());

    let xs = vec![[Val::new(2.0),Val::new(3.0),Val::new(-1.0)],
              [Val::new(3.0),Val::new(-1.0),Val::new(0.5)],
              [Val::new(0.5),Val::new(1.0),Val::new(1.0)],
              [Val::new(1.0),Val::new(1.0),Val::new(-1.0)]];

    let ys = vec![Val::new(1.0),Val::new(-1.0),Val::new(-1.0),Val::new(1.0)];
    let mut ypred = Vec::new();
    // eeh training loop
    for i in 0..50 {
        // forward pass
        ypred = Vec::new();
        for i in &xs {
            ypred.push(L.call(i.to_vec(), NONL::tanh)[0].clone());
        }
        // loss calculation
        let mut loss = Val::new(0.0);
        for (y_, y) in ypred.iter().zip(ys.clone()) {
            loss = &loss + &((y_ + &(-1.0 * &y)).powf(2.0));
        }
        // backward pass
        for w in &L.parameters() {
            w.set_grad(0.0);
        }
        loss.backward();
        println!("Iter: {}", i);
        for w in &L.parameters() {
            w.set_data(w.get_data() + (-0.15)*w.get_grad());
        }
        loss.print();
    }
    for y in ypred {
        y.print();
    }

    let xtest = vec![Val::new(2.0),Val::new(2.0),Val::new(-2.0)];
    &(L.call(xtest, NONL::tanh))[0].print();

}