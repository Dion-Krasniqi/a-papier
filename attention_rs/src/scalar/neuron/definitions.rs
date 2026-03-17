use crate::scalar::value::definitions::*;
use rand::random;

fn main(){

}

pub struct Neuron {
    inputs: Vec<f32>,
    weights: Vec<ValRef>,
    bias: ValRef,
}

impl Neuron {
    pub fn new(x: Vec<f32>) -> Neuron {
        Neuron {
            inputs: x,
            weights: Vec::new(),
            bias: Val::new(random::<f32>()),
        }
    }
}