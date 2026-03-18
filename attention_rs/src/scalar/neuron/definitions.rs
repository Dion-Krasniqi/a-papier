use crate::scalar::value::definitions::*;
use rand::random;

fn main(){}

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
    pub fn forward(&self, inputs: Vec<ValRef>) -> ValRef{
        let mut s = Val::new(0.0); // ValRef 
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            s = &s + &(w * x); 
        }
        s = &s + &self.bias;
        s._tanh()
    }
}

pub struct Neuron_Layer {
    pub nodes: Vec<Neuron>
}

impl Neuron_Layer{
    pub fn new(nin: u32, nout: u32) -> Neuron_Layer {
        let mut nodes = Vec::new();
        for _ in 0..nout {
            nodes.push(Neuron::new(nin))
        };
        Neuron_Layer { nodes }
    }
    pub fn forward(&self, inputs: Vec<ValRef>) -> Vec<ValRef> {
        let mut output = Vec::new();
        for node in &self.nodes {
            output.push(node.forward(inputs.clone()));
        }
        output
    }
}

pub struct MLP {
    pub layers: Vec<Neuron_Layer>,
}
impl MLP {
    pub fn new(mut nin: u32, nout: Vec<u32>) -> MLP {
        let mut layers = Vec::new();
        for x in 0..nout.len() {
            layers.push(Neuron_Layer::new(nin, nout[x]));
            nin = nout[x];
        };
        MLP { layers }
    }
    pub fn forward(&self, mut inputs: Vec<ValRef>) -> Vec<ValRef> {
        for x in 0..(&self.layers.len()-1) {
            inputs = self.layers[x].forward(inputs.clone());
        }
        self.layers.last().expect("iterated until before last").forward(inputs.clone())
    }
}