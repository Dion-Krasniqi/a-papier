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
    &O[0].backward();
    for p in &L.parameters() {
        p.print();
    };

    // test
    let xs = vec![[Val::new(2.0),Val::new(3.0),Val::new(-1.0)],
              [Val::new(3.0),Val::new(-1.0),Val::new(0.5)],
              [Val::new(0.5),Val::new(1.0),Val::new(1.0)],
              [Val::new(1.0),Val::new(1.0),Val::new(-1.0)]];

    let ys = vec![Val::new(1.0),Val::new(-1.0),Val::new(-1.0),Val::new(1.0)];
    let mut ypred = Vec::new();
    for i in xs {
        ypred.push(L.forward(i.to_vec())[0].clone());
    }

    let mut loss = Val::new(0.0);
    for (y_, y) in ypred.iter().zip(ys) {
        loss = &loss + &(&(y_ + &(-1.0 * &y))*&(y_ + &(-1.0 * &y)));
    }
    loss.print()

}