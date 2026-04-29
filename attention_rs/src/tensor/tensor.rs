use std::ops::{Add, Mul};
use std::rc::Rc;
use std::cell::RefCell;
use rand::random;
use std::collections::HashMap;
use std::fs;
use std::io::prelude::*;

pub fn norm_random(mean: f32, std: f32) -> f32{
    let a = 2.*random::<f32>() -1.;
    let b = 2.*random::<f32>() -1.;
    let s: f32 = a.powf(2.) + b.powf(2.);
    if s >= 1. {return norm_random(mean,std)} else if s == 0.0 { return 0.0 };
    return mean + std * (a * (-2.0 * s.ln()/s).sqrt())
}
struct TensorData {
    data: Vec<f32>, // no matter the dim, the data is stored in order and shape "determines" dimension
    grad: Vec<f32>,
    shape: Vec<usize>,
    // going to leave for now, for the same reason going to leave the current Tensor struct as is
    children: Vec<Tensor>,
}
pub struct Tensor(Rc<RefCell<TensorData>>);
impl Tensor {
    // add seed option
    pub fn rand(shape: Vec<usize>) -> Tensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_|norm_random(0.0,0.02)).collect(); //(0..size).map(|_| (random::<f32>()-0.5)*0.2).collect();
        let output = TensorData {
            data,
            grad: vec![0.0f32;size],
            shape: shape,
            children: Vec::new(),
        };
        Tensor(Rc::new(RefCell::new(output)))
    }
    pub fn tensor(value: f32, shape: Vec<usize>) -> Tensor {
        let size: usize = shape.iter().product();
        let data = vec![value;size];
        let output = TensorData {
            data,
            grad: vec![0.0f32;size],
            shape: shape,
            children: Vec::new(),
        };
        Tensor(Rc::new(RefCell::new(output)))
    }
    pub fn zero(shape: Vec<usize>) -> Tensor {
        Tensor::tensor(0.0, shape)
    }
    pub fn one(shape: Vec<usize>) -> Tensor {
        Tensor::tensor(1.0, shape)
    }
    pub fn like_tensor(tensor: &Tensor) -> Tensor {
        let shape = tensor.shape();
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| random::<f32>()).collect();
        let output = TensorData {
            data,
            grad: vec![0.0f32;size],
            shape: shape,
            children: Vec::new(),
        };
        Tensor(Rc::new(RefCell::new(output)))

    }
    pub fn print(&self) {
        let data = self.0.borrow();
        println!("shape: {:?}", data.shape);
        println!("data: {:?}", data.data);
        println!("grad: {:?}", data.grad);
    }
    pub fn set_grad(&self, val: f32) {
        for g in self.0.borrow_mut().grad.iter_mut() {
            *g = val;
        };
    }
    pub fn adjust_data(&self, multiplier: f32) {
        let grad = self.grad();
        for (d, g) in self.0.borrow_mut().data.iter_mut().zip(grad) {
            *d += multiplier*g;
        };
    }
    pub fn data(&self) -> Vec<f32> {
        self.0.borrow().data.clone()
    }
    pub fn grad(&self) -> Vec<f32> {
        self.0.borrow().grad.clone()
    }
    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().shape.clone()
    }
}
impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor(Rc::clone(&self.0))
    }
}
impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        let shape = self.shape();
        let output = TensorData{
            data: self.0.borrow_mut().data.iter().enumerate().map(|(e,i)| i * other.0.borrow().data[e]).collect(),
            grad: vec![0.0f32;self.shape().clone().iter().product()],
            shape,
            children: vec![self.clone(), other.clone()],
        };
        Tensor(Rc::new(RefCell::new(output)))
    }
}
impl Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: f32) -> Tensor {
        let shape = self.shape();
        let output = TensorData{
            data: self.0.borrow_mut().data.iter().map(|a| a * other).collect(),
            grad: vec![0.0f32;shape.iter().product()],
            shape,
            children: vec![self.clone()],
        };
        Tensor(Rc::new(RefCell::new(output)))
    }
}
impl Add<f32> for &Tensor {
    type Output = Tensor;
    fn add(self, other: f32) -> Tensor {
        let shape = self.shape();
        let output = TensorData{
            data: self.0.borrow_mut().data.iter().map(|a| a + other).collect(),
            grad: vec![0.0f32;shape.iter().product()],
            shape,
            children: vec![self.clone()],
        };
        Tensor(Rc::new(RefCell::new(output)))
    }
}
/*
impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor{
        if self.0.borrow().shape != other.0.borrow().shape {
            println!("Shapes are not compatibile, result is LHS.");
            return self.clone()
        };
        let shape = self.0.borrow().shape.clone();
        let size: usize = shape.iter().product();
        let output = TensorData {
            data: self.0.borrow().data.iter()
                  .zip(other.0.borrow().data.iter())
                  .map(|(a,b)| a+b ).collect(),
            grad: vec![0.0f32; size],
            shape: shape,
            children: vec![self.clone(), other.clone()],
        };
        Tensor(Rc::new(RefCell::new(output)))
    }
}
*/
pub fn add_forward(a: &Tensor, b: &Tensor) -> Tensor {
    if a.0.borrow().shape != b.0.borrow().shape {
            println!("Shapes are not compatibile, result is LHS.");
            return a.clone()
        };
    let shape = a.0.borrow().shape.clone();
    let size: usize = shape.iter().product();
    let output = TensorData {
        data: a.0.borrow().data.iter()
            .zip(b.0.borrow().data.iter())
            .map(|(a,b)| a+b ).collect(),
        grad: vec![0.0f32; size],
        shape: shape,
        children: vec![a.clone(), b.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn add_backward(out: &Tensor, a: &Tensor, b: &Tensor) {
    let out_grad = &out.0.borrow().grad;
    for (a_grad, o_grad) in a.0.borrow_mut().grad.iter_mut().zip(out_grad) {
        *a_grad += o_grad;
    }
    for (b_grad, o_grad) in b.0.borrow_mut().grad.iter_mut().zip(out_grad) {
        *b_grad += o_grad;
    }
}
pub fn add_forward_vec(a: &Vec<Tensor>, b: &Vec<Tensor>) -> Vec<Tensor> {
    let shape = a[0].0.borrow().shape.clone();
    let size: usize = shape.iter().product();
    let mut output = vec![Tensor::zero(shape.clone());a.len()];
    for i in 0..output.len() {
        if a[i].0.borrow().shape != b[i].0.borrow().shape {
                println!("Shapes are not compatibile, result is LHS1.");
                return a.clone()
            };
        
        let temp = TensorData {
            data: a[i].0.borrow().data.iter()
                .zip(b[i].0.borrow().data.iter())
                .map(|(a,b)| a+b ).collect(),
            grad: vec![0.0f32; size],
            shape: shape.clone(),
            children: vec![a[i].clone(), b[i].clone()],
        };
        output[i] = Tensor(Rc::new(RefCell::new(temp)));
    }
    output
}
pub fn add_backward_vec(out: &Vec<Tensor>, a: &Vec<Tensor>, b: &Vec<Tensor>) {
    let out_grad: Vec<Vec<f32>> = out.iter().map(|o|o.0.borrow().grad.clone()).collect();
    for i in 0..out.len() {
        for (a_grad, o_grad) in a[i].0.borrow_mut().grad.iter_mut().zip(&out_grad[i]) {
            *a_grad += o_grad;
        }
        for (b_grad, o_grad) in b[i].0.borrow_mut().grad.iter_mut().zip(&out_grad[i]) {
            *b_grad += o_grad;
        }
    }
}
pub fn matmul_forward(a: &Tensor, b: &Tensor) -> Tensor {
    // currently only for matrix
    if a.0.borrow().shape[1] != b.0.borrow().shape[0] {
        println!("failed");
        return a.clone();
    }
    let a_vec = a.0.borrow().data.clone();
    let b_vec = b.0.borrow().data.clone();

    let m = a.0.borrow().shape[0].clone();  
    let n = b.0.borrow().shape[0].clone();
    let p = b.0.borrow().shape[1].clone();

    let shape: Vec<usize> = vec![m, p];
    let size = m*p;
    let mut out_vec = vec![0.0f32;size];
    for i in 0..m{
        for j in 0..n {
            for k in 0..p {
                out_vec[i*p+k] += a_vec[i*n+j] * b_vec[j*p+k];
            }
        }
    }
    let output = TensorData {
        data: out_vec,
        grad: vec![0.0f32;size],
        shape,
        children: vec![a.clone(), b.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn matmul_backward(out: &Tensor, a: &Tensor, b: &Tensor) {
    let out_grad = out.0.borrow().grad.clone(); // a: mxn , b: nxp, out: mxp , out_grad: mxp
    // some build up for myself, the indexing is a bit confusing, should be either 0..n-1 or 1..n, but i think understandable
    // b = [b0, b1, b2, ... bp, bp+1, bp+2, ... b2p, ..., bn-1p, bn-1p +1, ... bnp]
    // out0 = [a0bp*0+0 + a1bp+0 + a2b2p+0 + ... + ambnp+0] 
    // outp = [a0bp*0+p + a1bp*1+p + a2b2p+p + ... + ambnp+p] 
    // a0 contribs to out0 ... outp
    // the dl/da0 notation isnt exactly correct
    // for out0: d(L)/d(a0) = (d(L)/d(out0)=out_grad0) * bp*0+0
    // for out1: d(L)/d(a0) = (d(L)/d(out1)=out_grad1) * bp*0+1
    // for outp: d(L)/d(a0) = (d(L)/d(outp)=out_gradp) * bp*0+p
    // d(L)/d(a0) = out_grad0 * bp*0+0 + out_grad1 * bp*0+1 + ... + out_gradp * bp*0+p
    // a_grad = [dl/da0, dl/da1, ..., dldan] ...
    // a_grad = out_grad row * b-s row => a_grad = out_grad @ bT
    let m = out.0.borrow().shape[0];
    let n = b.0.borrow().shape[0];
    let p = out.0.borrow().shape[1];
    let b_vec = b.0.borrow().data.clone();
    for i in 0..m {
        for j in 0..n{
            for k in 0..p{
                a.0.borrow_mut().grad [i*n+j] += out_grad[i*p+k] * b_vec[j*p+k];
            }
        }
    }
    // b_grad = aT @ out_grad
    let m = out.0.borrow().shape[0];
    let n = b.0.borrow().shape[0];
    let p = out.0.borrow().shape[1];
    let a_vec = a.0.borrow().data.clone();
    for i in 0..n {
        for j in 0..m{
            for k in 0..p{
                b.0.borrow_mut().grad[i*p+k] += out_grad[j*p+k] * a_vec[i*m+j];
            }
        }
    }
}
pub fn tanh_forward(a: &Tensor) -> Tensor {
    let size = a.0.borrow().shape.iter().product();
    //data.iter_mut().zip(a.0.borrow().data).map(|(o,a)| *o + a);
    let output = TensorData {
        data: a.0.borrow().data.clone().iter().map(|o| o.tanh() ).collect(),
        shape: a.0.borrow().shape.clone(),
        grad: vec![0.0f32;size],
        children: vec![a.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn tanh_backward(out: &Tensor, a: &Tensor) {
    let out_data = out.0.borrow().data.clone();
    let out_grad = out.0.borrow().grad.clone();
    for i in 0..out_data.len() {
        a.0.borrow_mut().grad[i] += (1.0 - out_data[i].powf(2.0)) * out_grad[i];
    };
}
pub fn relu_forward(a: &Tensor) -> Tensor {
    let size: usize = a.0.borrow().shape.iter().product();
    let output = TensorData {
        data: a.0.borrow().data.clone().iter().map(|o| if *o > 0.0 { *o } else { 0.0 }).collect(),
        grad: vec![0.0f32;size],
        shape: a.0.borrow().shape.clone(),
        children: vec![a.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn relu_backward(out: &Tensor, a: &Tensor) {
    let out_data = out.0.borrow().data.clone();
    let out_grad = out.0.borrow().grad.clone();
    for i in 0..out_data.len() {
        a.0.borrow_mut().grad[i] += (if out_data[i] > 0.0 { 1.0 } else { 0.0 }) * out_grad[i];
    }
}
pub fn sigmoid_forward(a: &Tensor) -> Tensor {
    let size: usize = a.0.borrow().shape.iter().product();
    let output = TensorData {
        data: a.0.borrow().data.clone().iter_mut().map(|o| 1.0/(1.0+(-(*o)).exp())).collect(),
        grad: vec![0.0f32;size],
        shape: a.0.borrow().shape.clone(),
        children: vec![a.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn sigmoid_backward(out: &Tensor, a: &Tensor) {
    let out_data = out.0.borrow().data.clone();
    let out_grad = out.0.borrow().grad.clone();
    for i in 0..out_data.len() {
        a.0.borrow_mut().grad[i] += (out_data[i] - out_data[i].powf(2.0)) * out_grad[i];
    }
}
pub fn softmax_forward(a: &Vec<Tensor>) -> Vec<Tensor> {
    let shape = a[0].shape(); // mxn
    let size: usize = shape.iter().product();
    let cols = shape[1].clone();
    let rows = shape[0].clone();

    //let mut exp_sum: Vec<f32> = vec![0.0f32;a.len()];
    let mut exp_sums: Vec<f32> = vec![0.0f32;a.len()*shape[0].clone()];
    let mut shifted: Vec<Vec<f32>> = vec![vec![0.0f32;size];a.len()];
    for i in 0..a.len() {
        let max = a[i].0.borrow().data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        shifted[i] = a[i].0.borrow().data.iter().map(|x| x - max).collect();
        for j in 0..shape[0].clone() {
            for k in 0..shape[1].clone() {
                exp_sums[i*rows+j] += shifted[i][j*cols+k].exp();
            }
        }
    }
    let mut output = vec![Tensor::zero(shape.clone());a.len()];
    for i in 0..a.len() {
        for j in 0..rows {
            for k in 0..cols {
                shifted[i][j*cols+k] = shifted[i][j*cols+k].exp()/exp_sums[i*rows+j];
            }
        }
        let temp = TensorData {
                data: shifted[i].clone(),
                grad: vec![0.0f32;size],
                shape: shape.clone(),
                children: vec![a[i].clone()],
            };
        output[i] = Tensor(Rc::new(RefCell::new(temp)));
    };
    output
}
pub fn softmax_backward(out: &Vec<Tensor>, a: &Vec<Tensor>) {
    
    // for one tensor
    // fx = e(ai)/Sum(aj.exp())
    // k = i dout/ak = (e(ak)* sum(aj.exp()) - e(ak)*e(ak))/sum(aj.exp())^2 * out_grad
    // = (e(ak)/sum(aj.exp()) - ak.exp()^2/ sum^2) * out_grad
    // = (out_data - out_data^2) * out_grad = out_data(1 - out_data) * out_grad
    //
    //k =/ i dout/ak = (0 - ak.exp*ai.exp())/sum^2 * out_grad 
    // = - out_data[k] * out_data[i] * out_grad
    let shape = a[0].shape();
    let rows = shape[0];
    let cols = shape[1];
    let out_data: Vec<Vec<f32>> = out.iter().map(|o|o.0.borrow().data.clone()).collect();
    let out_grad: Vec<Vec<f32>> = out.iter().map(|o|o.0.borrow().grad.clone()).collect();
    let length = out_data[0].len();
    for k in 0..a.len() {
        for i in 0..length{
                for j in 0..rows {
                    for l in 0..cols {
                        if i == j*cols + l {
                            a[k].0.borrow_mut().grad[i] += (out_data[k][i]*(1.0-out_data[k][i]))
                             * out_grad[k][i];
                        } else {
                            a[k].0.borrow_mut().grad[i] += -out_data[k][i] * out_data[k][j*cols+l]
                            * out_grad[k][j*cols+l];
                        }
                    }
                }
            }
    }
}
pub fn layernorm_forward(a: &Tensor, betta: &Tensor, gamma: f32) -> Tensor {
    let rows = a.0.borrow().shape[0];
    let cols = a.0.borrow().shape[1];
    let a_vec = a.0.borrow().data.clone();
    let mut sum = vec![(0.0f32,0.0f32);rows];
    for i in 0..rows {
        for j in 0..cols {
            sum[i].0 += a_vec[i*cols+j];
            sum[i].1 += a_vec[i*cols+j].powf(2.0);
        }
        sum[i].0 /= cols as f32; //E[X]
        sum[i].1 /= cols as f32; //E[X^2]
        sum[i].1 -= sum[i].0.powf(2.0); //Var(x) = E[X^2] - E[X]^2
    }
    let mut a_norm = vec![0.0f32;rows*cols];
    for i in 0..rows {
        for j in 0..cols {
            a_norm[i*cols+j] += gamma*(a_vec[i*cols+j] - sum[i].0)/(sum[i].1 + 1e-5).sqrt() + betta.0.borrow().data[i*cols+j];
        }
    }
    let output = TensorData {
        data: a_norm,
        grad: vec![0.0f32;rows*cols],
        shape: a.0.borrow().shape.clone(),
        children: vec![a.clone(),betta.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn layernorm_backward(out: &Tensor, a: &Tensor, betta: &Tensor, gamma: f32) {
    // TODO gamma will be a wieght tensor so update its gradient, think about mean aswell
    let eps = 1e-5;
    let rows = out.0.borrow().shape[0];
    let cols = out.0.borrow().shape[1];
    let out_data = out.0.borrow().data.clone();
    let out_grad = out.0.borrow().grad.clone();
    let a_vec = a.0.borrow().data.clone();
    // dl/dbetta
    // dl/dbetta = dl/dout * dout/dbetta; dout/dbetta = d(out_data*gamma + betta)/dbetta = 1
    let mut betta_grad = vec![0.0f32;rows*cols];
    for i in 0..betta_grad.len() {
        betta_grad[i] += out_grad[i];
    }
    // dl/d(out/gamma)
    // kinda useless, but might use in final formula
    let mut out_wo_gamma_grad = vec![0.0f32;rows*cols];
    for (o_w, o_g) in out_wo_gamma_grad.iter_mut().zip(&out_grad) {
        *o_w += o_g * gamma;
    }
    let mut varexs = vec![(0.0f32, 0.0f32);rows];
    // varexs.0 - means per row, varexs.1 - Var[x] per row
    for i in 0..rows {
        for j in 0..cols {
            varexs[i].0 += a_vec[i*cols+j];
            varexs[i].1 += a_vec[i*cols+j].powf(2.0);
        }
        varexs[i].0 /= cols as f32; //E[X]
        varexs[i].1 /= cols as f32; //E[X^2]
        varexs[i].1 -= varexs[i].0.powf(2.0); //Var(x) = E[X^2] - E[X]^2
    }
    //from here only need sqrt(varx-eps)
    for (_, var) in varexs.iter_mut() {
        *var = (*var + eps).sqrt();
    }
    // entries are all out grads summed per row
    let mut douti = vec![0.0f32;rows];
    for i in 0..rows {
        for j in 0..cols {
            douti[i] += out_grad[i*cols+j];
        }
    }
    // same thing as above just each grad * out/gamma
    let mut douti_row = vec![0.0f32;rows];
    for i in 0..rows {
        for j in 0..cols {
            douti_row[i] += out_grad[i*cols+j]*out_data[i*cols+j]/gamma;
        }
    }
    // dl/dai = dl/dout * dout/dai + dl/dmean*dmean/dai + dl/dvarx*dvarx/dai
    let mut a_grad = vec![0.0f32;rows*cols];
    for i in 0..rows {
        for j in 0..cols {
            let out_wo_gamma = (a_vec[i*cols+j]-varexs[i].0)/varexs[i].1;
            a_grad[i*cols+j] += gamma * (1.0/(cols as f32 * varexs[i].1)) * ((cols as f32) * out_grad[i*cols+j]
            - douti[i] - out_wo_gamma * douti_row[i]);

        }
    }
    // assigning
    betta.0.borrow_mut().grad = betta_grad;
    a.0.borrow_mut().grad = a_grad;
}
pub fn embedding_forward(tokens: &[usize], weight: &Tensor) -> Tensor {
    let cols = weight.0.borrow().shape[1];
    let w_vec = weight.0.borrow().data.clone();

    let mut out = vec![0.0f32;tokens.len()*cols];
    for (i, &token) in tokens.iter().enumerate() {
        for j in 0..cols {
            out[i*cols+j] = w_vec[token*cols+j];
        }
    }
    let output = TensorData {
        data: out,
        grad: vec![0.0f32;tokens.len()*cols],
        shape: vec![tokens.len(),cols],
        children: vec![weight.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn embedding_backward(out: &Tensor, weight: &Tensor, tokens: &[usize]) {
    // dl/dw = dl/dout * dout/dw = dl/dout
    let out_grad = out.0.borrow().grad.clone();
    let cols = out.0.borrow().shape[1];
    for (i, &token) in tokens.iter().enumerate() {
        for j in 0..cols {
            weight.0.borrow_mut().grad[token*cols+j] += out_grad[i*cols+j];
        }
    }
}
pub fn cross_entropy_forward(a: &Vec<Tensor>, targets: &Vec<&[usize]>) -> Tensor {
    let rows = a[0].0.borrow().shape[0];
    let cols = a[0].0.borrow().shape[1];
    let size: usize = a[0].shape().iter().product();
    let mut loss = 0.0;
    for i in 0..a.len(){
        let probs = a[i].0.borrow().data.clone();
        for (j, &target) in targets[i].iter().enumerate() {
            loss -=  (probs[j*cols+target] + 1.0e-8).ln();
        }
    }
    Tensor::tensor(loss/((rows*a.len()) as f32), vec![1,1])
}
pub fn cross_entropy_backward(a: &Vec<Tensor>, targets: &Vec<&[usize]>) {
    let rows = a[0].0.borrow().shape[0];
    let cols = a[0].0.borrow().shape[1];
    for i in 0..a.len(){
        for (j, &target) in targets[i].iter().enumerate() {
            a[i].0.borrow_mut().grad[j*cols+target] -= 1.0/(rows * a.len()) as f32; 
        }
    }
}
pub fn positional_encoding(seq_len: usize, emb_dim: usize) -> Tensor {
    let mut data = vec![0.0f32;seq_len*emb_dim];
    for i in 0..seq_len {
        for j in 0..emb_dim {
            data[i*emb_dim+j] = if j%2==0 {
                (i as f32/1000_f32.powf(2. * j as f32/emb_dim as f32)).sin()
            } else {
                (i as f32/1000_f32.powf(2. * j as f32/emb_dim as f32)).cos()
            };
        }
    }
    let output = TensorData {
        data,
        grad: vec![0.0f32;seq_len*emb_dim],
        shape: vec![seq_len, emb_dim],
        children: Vec::new(),
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn transpose(a: &Tensor) -> Tensor {
    let a_vec = a.0.borrow().data.clone();
    let a_grad = a.0.borrow().grad.clone();
    let rows = a.0.borrow().shape[0];
    let cols = a.0.borrow().shape[1];
    let mut data = vec![0.0f32;rows*cols];
    let mut grad = vec![0.0f32;rows*cols];
    for i in 0..rows {
        for j in 0..cols {
            data[i*cols + j] = a_vec[j*rows+i];
            grad[i*cols + j] = a_grad[j*rows+i];
        }
    }
    let output = TensorData {
        data,
        grad,
        shape: vec![cols,rows],
        children: vec![a.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn transpose_grad(trans: &Tensor, og: &Tensor) {
    let shape = trans.shape();
    let rows = shape[0].clone();
    let cols = shape[1].clone();
    for i in 0..rows {
        for j in 0..cols {
            og.0.borrow_mut().grad[i*cols+j] = trans.0.borrow().grad[j*rows+i];
        }
    }
}

pub struct Tokenizer {
    vocab: HashMap<char, usize>,
    reverse: HashMap<usize, char>,
}
impl Tokenizer {
    pub fn new(text: &str) -> Tokenizer {
        let mut vocab = HashMap::new();
        let mut reverse = HashMap::new();
        let mut id = 0;
        for t in text.chars() {
            if !vocab.contains_key(&t) {
                vocab.insert(t, id);
                reverse.insert(id, t);
                id += 1;
            }
        }
        Tokenizer { vocab, reverse}
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|t| self.vocab[&t]).collect()
    }
    pub fn decode(&self, token: &[usize]) -> String {
        token.iter().map(|t| self.reverse[&t]).collect()
    }
    pub fn get_vocab_len(&self) -> usize {
        self.vocab.len()
    }
}
pub struct AttentionHead {
    pub W_q: Tensor,
    pub W_k: Tensor,
    pub W_v: Tensor,
}
impl AttentionHead {
    pub fn new(dim_shape: Vec<usize>) -> AttentionHead {
        let W_q = Tensor::rand(dim_shape.clone());
        let W_k = Tensor::rand(dim_shape.clone());
        let W_v = Tensor::rand(dim_shape.clone());
        AttentionHead { W_q, W_k, W_v}
    }
    pub fn forward(&self, x: &Vec<Tensor>) -> Vec<Tensor> {
        let mut temp = Tensor::zero(x[0].shape());
        let mut output: Vec<Tensor> = (0..x.len()).map(|_| Tensor::rand(x[0].shape())).collect();
        for (mut out, tensor) in output.iter_mut().zip(x) {
            let Q = matmul_forward(&tensor, &self.W_q); // Q = x @ W_q = (block_size x emb_dim) @ (emb_dim x head_dim) = Q(block_size * head_dim), dim_shape = emb_dim * head_dim
            let K = matmul_forward(&tensor, &self.W_k);
            let V = matmul_forward(&tensor, &self.W_v);
            let dk: usize= K.shape()[1];   
            let Q_Kt = matmul_forward(&Q, &transpose(&K)); // (block_size * head_dim) @ (head_dim * block_size)
            let scaling_factor = 1./((dk as f32).sqrt()); //Tensor::tensor(1./((dk as f32).sqrt()), Q_Kt.shape()); 
            temp = softmax_forward(&vec![(&Q_Kt * scaling_factor)])[0].clone();
            *out = matmul_forward(&temp, &V);
        }
        output
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        // eeh for now
        vec![self.W_q.clone(), self.W_k.clone(), self.W_v.clone()]
    }
    pub fn backward(&self, out: &Vec<Tensor>, a: &Vec<Tensor>) {
        for i in 0..a.len() {
            // cache/own these
            let Q = matmul_forward(&a[i], &self.W_q);
            let K = matmul_forward(&a[i], &self.W_k);
            let V = matmul_forward(&a[i], &self.W_v);
            let Kt = transpose(&K);
            let Q_Kt = matmul_forward(&Q, &Kt);
            let dk: usize= K.shape()[1]; 
            let scaling_factor = 1./((dk as f32).sqrt());
            let softmax_res = softmax_forward(&vec![(&Q_Kt * scaling_factor)])[0].clone();
            matmul_backward(&out[i], &softmax_res, &V);
            matmul_backward(&Q_Kt, &Q, &Kt);
            transpose_grad(&Kt, &K);
            matmul_backward(&V, &a[i], &self.W_v);
            matmul_backward(&K, &a[i], &self.W_k);
            transpose_grad(&K, &Kt);
            matmul_backward(&Q, &a[i], &self.W_q);
        }
    }
}
pub struct MaskedAttentionHead {
    pub W_q: Tensor,
    pub W_k: Tensor,
    pub W_v: Tensor,
}
impl MaskedAttentionHead {
    pub fn new(dim_shape: Vec<usize>) -> MaskedAttentionHead {
        let W_q = Tensor::rand(dim_shape.clone());
        let W_k = Tensor::rand(dim_shape.clone());
        let W_v = Tensor::rand(dim_shape.clone());
        MaskedAttentionHead { W_q, W_k, W_v }
    }
    pub fn forward(&self, a: &Vec<Tensor>) -> Vec<Tensor> {
        let mut temp = Tensor::zero(a[0].shape());
        let mut output: Vec<Tensor> = (0..a.len()).map(|_|Tensor::rand(a[0].shape())).collect();
        for (mut out, tensor) in output.iter_mut().zip(a) {
            let Q = matmul_forward(&tensor, &self.W_q); // Q = a @ W_q = (block_size x emb_dim) @ (emb_dim x head_dim) = Q(block_size * head_dim), dim_shape = emb_dim * head_dim
            let K = matmul_forward(&tensor, &self.W_k);
            let V = matmul_forward(&tensor, &self.W_v);
            let dk: usize= K.shape()[1];   
            let Q_Kt = matmul_forward(&Q, &transpose(&K)); // (block_size * head_dim) @ (head_dim * block_size)
            let mask = Tensor::like_tensor(&Q_Kt);
            let scaling_factor = 1./((dk as f32).sqrt()); //Tensor::tensor(1./((dk as f32).sqrt()), Q_Kt.shape()); 
            let rows = Q_Kt.shape()[0];
            let cols = Q_Kt.shape()[1];
            //mask
            for i in 0..rows {
                for j in (i+1)..cols {
                    Q_Kt.0.borrow_mut().data[i*cols+j] += f32::NEG_INFINITY;
                }
            }
            temp = softmax_forward(&vec![(&Q_Kt * scaling_factor)])[0].clone();
            *out = matmul_forward(&temp, &V);
        }
        output
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        // eeh for now
        vec![self.W_q.clone(), self.W_k.clone(), self.W_v.clone()]
    }
    pub fn backward(&self, out: &Vec<Tensor>, a: &Vec<Tensor>) {
        for i in 0..a.len() {
            // cache/own these
            let Q = matmul_forward(&a[i], &self.W_q);
            let K = matmul_forward(&a[i], &self.W_k);
            let V = matmul_forward(&a[i], &self.W_v);
            let Kt = transpose(&K);
            let Q_Kt = matmul_forward(&Q, &Kt);
            let dk: usize= K.shape()[1]; 
            let scaling_factor = 1./((dk as f32).sqrt());
            let rows = Q_Kt.shape()[0];
            let cols = Q_Kt.shape()[1];
            //mask
            for i in 0..rows {
                for j in (i+1)..cols {
                    Q_Kt.0.borrow_mut().data[i*cols+j] += f32::NEG_INFINITY;
                }
            }
            let softmax_res = softmax_forward(&vec![(&Q_Kt * scaling_factor)])[0].clone();
            matmul_backward(&out[i], &softmax_res, &V);
            for i in 0..rows {
                for j in i..cols {
                    // DL/DQ_kt = dl/dsoft_max * dsoft_max/dQ_kt
                    // d(Q_kt * scaling_factor) / dQ_kt = scaling_factor
                    // DL/DQ_kt = dsoft_max * scaling_factor
                    Q_Kt.0.borrow_mut().grad[i*cols+j] += softmax_res.0.borrow().grad[i*cols+j] * scaling_factor;
                }
            }
            matmul_backward(&Q_Kt, &Q, &Kt);
            transpose_grad(&Kt, &K);
            matmul_backward(&V, &a[i], &self.W_v);
            matmul_backward(&K, &a[i], &self.W_k);
            transpose_grad(&K, &Kt);
            matmul_backward(&Q, &a[i], &self.W_q);
        }
    }
}
pub struct FeedForward {
    pub weights: (Tensor, Tensor),
    pub biases: (Tensor, Tensor),
    pub f1: Vec<Tensor>,
    pub f2: Vec<Tensor>,
}
impl FeedForward {
    pub fn new(shape: Vec<usize>, in_len: usize) -> FeedForward {
        // confused on the parametrization
        let weights = (Tensor::rand(vec![shape[1].clone(),shape[1].clone()]),Tensor::rand(vec![shape[1].clone(),shape[1].clone()]));
        let biases = (Tensor::zero(shape.clone()),Tensor::zero(shape.clone()));
        let f1: Vec<Tensor> = (0..in_len).map(|_|Tensor::rand(shape.clone())).collect();
        let f2: Vec<Tensor> = (0..in_len).map(|_|Tensor::rand(shape.clone())).collect();
        FeedForward { weights, biases, f1, f2 }
    }
    pub fn forward(&mut self, x: &Vec<Tensor>) -> Vec<Tensor> {
        let mut output: Vec<Tensor> = (0..x.len()).map(|_|Tensor::rand(x[0].shape())).collect();
        for i in 0..x.len() {
            self.f1[i] = relu_forward(&add_forward(&matmul_forward(&x[i],&self.weights.0), &self.biases.0)); 
            self.f2[i] = matmul_forward(&self.f1[i], &self.weights.1); 
            output[i] = add_forward(&self.f2[i], &self.biases.1);
        }
        output
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.push(self.weights.0.clone());
        params.push(self.weights.1.clone());
        params.push(self.biases.0.clone());
        params.push(self.biases.1.clone());
        params
    }
    pub fn backward(&self, out: &Vec<Tensor>, x: &Vec<Tensor>) {
        // f1 = weight.0 * x + bias.0
        // f1_relu = relu(f1)
        // f2 = f1_relu * weight.1
        // out = f2 + bias.1
        // out = (relu(weight.0 * x + bias.0) * weight.1) + bias.1
        // dL/dx = dl/dout * dout/dx
        // dL/db1 = dl/dout * dout/db1
        // dout/db1 = 1 * out_grad
        // dout/db0 =  (if (term>0.) 1.0 else 0.) * out_grad
        // dout/dw1 = relu(w0 * x + bias.0) * out_grad
        // dout/dw0 = (if term>0. 1. else 0.) * x * out_grad
        // dL/dx = out_grad * dout/dx , dout/dx = d((relu(weight.0 * x + bias.0) * weight.1) + bias.1)/dx
        // dout/dx = (if term>0. 1. else 0.) * w0 * out_grad
        for i in 0..x.len() {
            add_backward(&out[i], &self.f2[i], &self.biases.1);
            matmul_backward(&self.f2[i], &self.f1[i], &self.weights.1);
            let mul_res = matmul_forward(&x[i],&self.weights.0);
            let add_res = add_forward(&mul_res, &self.biases.0);
            relu_backward(&self.f1[i], &add_res);
            add_backward(&add_res, &mul_res, &self.biases.0);
            matmul_backward(&mul_res, &x[i], &self.weights.0);
        }
    }
}
pub struct LinearLayer {
    // pub weights: Vec<Tensor>,
    // pub biases: Vec<Tensor>,
    pub weights: Tensor,
    pub biases: Tensor,
}
impl LinearLayer {
    pub fn new(shape: Vec<usize>, block_size: usize, in_len: usize) -> LinearLayer {
        // matmul(x, weights) = (block_size x emb_dim) @ (emb_dim x vocab_size) = block_size x vocab_size
        // add(matmul, bias) = (block_size x vocab_size) + (block_size x vocab_size)
        let weights = Tensor::rand(shape.clone());//vec![Tensor::rand(shape.clone());in_len];
        let biases = Tensor::rand(vec![block_size, shape.clone()[1]]);//vec![Tensor::rand(vec![block_size, shape.clone()[1]]);in_len];
        LinearLayer { weights, biases}
    }
    pub fn forward(&self, a: &Vec<Tensor>) -> Vec<Tensor> {
        // let shape = a[0].shape();
        // let length = a.len();
        // let mut output: Vec<Tensor> = (0..a.len()).map(|_|Tensor::rand(shape.clone())).collect();
        // for i in 0..length {
        //     output[i] = add_forward(&matmul_forward(&a[i], &self.weights[i]), &self.biases[i]);
        // }
        // output
        a.iter().map(|x|add_forward(&matmul_forward(x, &self.weights), &self.biases)).collect()
    }
    pub fn backward(&self, out: &Vec<Tensor>, a: &Vec<Tensor>) {
            // for i in 0..out.len() {
            //     let matmul_res = matmul_forward(&a[i], &self.weights[i]);
            //     add_backward(&out[i], &matmul_res, &self.biases[i]);
            //     matmul_backward(&matmul_res, &a[i], &self.weights[i]);
            // }
            for i in 0..out.len() {
                let matmul_res = matmul_forward(&a[i], &self.weights);
                add_backward(&out[i], &matmul_res, &self.biases);
                matmul_backward(&matmul_res, &a[i], &self.weights);
            }
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        // let mut params: Vec<Tensor> = self.biases.clone();
        // params.extend(self.weights.clone());
        // params
        vec![self.weights.clone(), self.biases.clone()]
    }
}
pub struct LayerNorm {
    pub betta: Tensor,
    pub gamma: f32,
}
impl LayerNorm {
    pub fn new(shape: Vec<usize>) -> LayerNorm {
        LayerNorm {
            betta: Tensor::zero(shape.clone()),
            gamma: 1.0,
        }
    }
    pub fn forward(&self, x: &Vec<Tensor>) -> Vec<Tensor> {
        let output: Vec<Tensor> = x.iter().map(|o|layernorm_forward(&o, &self.betta, self.gamma)).collect();
        output
    }
    pub fn backward(&self, out: &Vec<Tensor>, x: &Vec<Tensor>) {
        for i in 0..x.len() {
            layernorm_backward(&out[i], &(x[i]), &self.betta, self.gamma);
        }
    }
}
pub fn save_model(params: &Vec<Tensor>, path: &str) {
    let mut file = fs::File::create(path).unwrap();
    for p in params {
        let data = p.data();
        //println!("{}", data.len());
        file.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
        for val in &data {
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}
pub fn load_model(params: &Vec<Tensor>, path: &str) {
    let mut file = fs::File::open(path).unwrap();
    for p in params {
        let mut buffer = [0u8;4];
        file.read_exact(&mut buffer).unwrap();
        let len = u32::from_le_bytes(buffer);
        let mut data = vec![0.0f32;len as usize]; 
        for i in 0..len {
             let mut data_buffer = [0u8;4];
             file.read_exact(&mut data_buffer).unwrap();
             data[i as usize] = f32::from_le_bytes(data_buffer);
        }
        p.0.borrow_mut().data = data;
    }
}
// pub enum Layer {
//     Attention(AttentionHead),
//     MaskedAttention(MaskedAttentionHead),
//     FeedForwardLayer(FeedForward),
//     Linear(LinearLayer),
//     ResidualFFN(FeedForward),
//     ResidualAttention(AttentionHead),
//     ResidualMaskedAttention(MaskedAttentionHead),
//     Softmax,
//     Norm(LayerNorm),
// }
// impl Layer {
//     pub fn forward(&mut self, x: Vec<Tensor>) -> Vec<Tensor> {
//         match self {
//             Layer::Attention(a) => a.forward(x),
//             Layer::MaskedAttention(a) => a.forward(x),
//             Layer::FeedForwardLayer(a) => a.forward(x),
//             Layer::Linear(a) => a.forward(x),
//             Layer::Softmax => softmax_forward(x),
//             Layer::Norm(a) => a.forward(x),
//             // idk
//             Layer::ResidualFFN(a) => {
//                 let out = a.forward(x.clone());
//                 add_forward_vec(x, out)
//             },
//             Layer::ResidualAttention(a) => {
//                 let out = a.forward(x.clone());
//                 add_forward_vec(x, out)
//             },
//             Layer::ResidualMaskedAttention(a) => {
//                 let out = a.forward(x.clone());
//                 add_forward_vec(x, out)
//             },
//         }
//     }
//     pub fn backward(&mut self, x: Vec<Tensor>) {
//         match self {
//             Layer::Attention(a) => a.backward(x),
//             Layer::MaskedAttention(a) => a.backward(x),
//             Layer::FeedForwardLayer(a) => a.backward(x),
//             Layer::Linear(a) => a.backward(x),
//             Layer::Softmax => softmax_backward(x.clone(), x.clone()),
//             Layer::Norm(a) => a.backward(x),
//             // idk
//             Layer::ResidualFFN(a) => {
//                 let new_o = add_forward_vec(x.clone(), a.out.clone());
//                 add_backward(&new_o[0], &x[0], &a.out[0]);
//                 a.backward(x);
//             },
//             Layer::ResidualAttention(a) => {
//                 let out = a.forward(x.clone());
//                 add_forward_vec(x, out);
//             },
//             Layer::ResidualMaskedAttention(a) => {
//                 let out = a.forward(x.clone());
//                 add_forward_vec(x, out);
//             },
//         }
//     }
// }
// pub struct Stack {
//     pub layers: Vec<Layer>,
// }
// impl Stack {
//     pub fn new(layers: Vec<Layer>) -> Stack {
//         Stack { layers }
//     }
//     pub fn add(&mut self, layer: Layer) {
//         self.layers.push(layer);
//     }
//     pub fn forward(&mut self, x: Vec<Tensor>) -> Vec<Tensor> {
//         // one in 
//         let mut output = x.clone();
//         for l in &mut self.layers {
//             output = l.forward(output);
//         }
//         output
//     }
//     pub fn backward(&mut self, x: Vec<Tensor>){
//         for mut l in self.layers.iter_mut().rev() {
//             l.backward(x.clone());
//         }
//     }
// }