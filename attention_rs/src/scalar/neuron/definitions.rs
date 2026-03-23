use crate::scalar::value::definitions::*;
use rand::random;

pub fn example() {
    // example
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

pub struct Neuron {
    pub weights: Vec<ValRef>,
    pub bias: ValRef,
}
#[derive(Clone)]
pub enum NONL {
    tanh,
    sigmoid,
    relu,
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
    pub fn forward(&self, inputs: Vec<ValRef>, nl: NONL) -> ValRef{
        let mut s = Val::new(0.0); // ValRef 
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            s = &s + &(w * x); 
        }
        s = &s + &self.bias;
        match nl {
            NONL::tanh => s._tanh(),
            _ => s._sigmoid(),
        }
    }
    pub fn parameters(&self) -> Vec<ValRef> {
        let mut params: Vec<ValRef> = self.weights.iter().map(|w| w.clone()).collect();
        params.push(self.bias.clone());
        params
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
    pub fn forward(&self, inputs: Vec<ValRef>, nl: NONL) -> Vec<ValRef> {
        let mut output = Vec::new();
        for node in &self.nodes {
            output.push(node.forward(inputs.clone(), nl.clone()));
        }
        output
    }
    pub fn parameters(&self) -> Vec<ValRef> {
        let mut params = Vec::new();
        for n in &self.nodes {
            for w in n.parameters() {
                params.push(w);
            }
        }
        params
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
    pub fn call(&self, mut inputs: Vec<ValRef>, nl: NONL) -> Vec<ValRef> {
        for x in 0..(&self.layers.len()-1) {
            inputs = self.layers[x].forward(inputs.clone(), nl.clone());
        }
        self.layers.last().expect("iterated until before last").forward(inputs.clone(), nl.clone())
    }
    pub fn parameters(&self) -> Vec<ValRef> {
        let mut params = Vec::new();
        for l in &self.layers {
            for p in l.parameters() {
                params.push(p);
            }
        }
        params
    }
}