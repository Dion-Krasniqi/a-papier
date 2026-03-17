use crate::scalar::value::definitions::*;
use rand::random;

fn main(){
    
}

pub struct Neuron {
    pub weights: Vec<ValRef>,
    pub bias: ValRef,
}

impl Neuron {
    pub fn new(nin: u32) -> Neuron {
        let mut init_weights = Vec::new();
        for _ in 0..nin {
            init_weights.push(Val::new(random::<f32>()));
        };
        Neuron {
            weights: init_weights,
            bias: Val::new(random::<f32>()),
        }
    }
    pub fn forward(&self, inputs: Vec<f32>) -> ValRef{
        let mut s = Val::new(0.0); // ValRef 
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            s = &s + &(w * &Val::new(*x)); 
        }
        s = &s + &self.bias;
        s._tanh()
    }
}
