use std::ops::{Add, Mul};
use std::rc::Rc;
use std::cell::RefCell;
use rand::random;

struct TensorData {
    data: Vec<f32>, // no matter the dim, the data is stored in order and shape "determines" dimension
    grad: Vec<f32>,
    shape: Vec<usize>,
    // going to leave for now, for the same reason going to leave the current Tensor struct as is
    children: Vec<Tensor>,
}
pub struct Tensor(Rc<RefCell<TensorData>>);
impl Tensor {
    pub fn tensor(shape: Vec<usize>) -> Tensor {
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
    pub fn tensor_one(value: f32, shape: Vec<usize>) -> Tensor {
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
    pub fn print(&self) {
        let data = self.0.borrow();
        println!("shape: {:?}", data.shape);
        println!("data: {:?}", data.data);
        println!("grad: {:?}", data.grad);
    }
}
impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor(Rc::clone(&self.0))
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
pub fn set_grad(a: &Tensor, val: f32) {
    for ag in a.0.borrow_mut().grad.iter_mut() {
        *ag = val;
    };
}
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
        *a_grad += 0.0;
    }
    for (b_grad, o_grad) in b.0.borrow_mut().grad.iter_mut().zip(out_grad) {
        *b_grad += o_grad;
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
                out_vec[i*p+k] += a_vec[i*m+j] * b_vec[k*n+j];
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
    let size = m*n;
    //a_grad
    let mut a_grad = vec![0.0f32;size];
    for i in 0..m {
        for j in 0..n{
            for k in 0..p{
                a_grad[i*n+j] += out_grad[i*p+k] * b_vec[j*p+k];
            }
        }
    }
    // b_grad = aT @ out_grad
    let m = out.0.borrow().shape[0];
    let n = b.0.borrow().shape[0];
    let p = out.0.borrow().shape[1];
    let a_vec = a.0.borrow().data.clone();
    let size = n*p;
    let mut b_grad = vec![0.0f32;size];
    for i in 0..n {
        for j in 0..m{
            for k in 0..p{
                b_grad[i*p+k] += out_grad[j*p+k] * a_vec[i*m+j];
            }
        }
    }
    a.0.borrow_mut().grad = a_vec.clone();
    b.0.borrow_mut().grad = b_vec.clone();
}
pub fn tanh_forward(a: &Tensor) -> Tensor {
    let size = a.0.borrow().shape.iter().product();
    //data.iter_mut().zip(a.0.borrow().data).map(|(o,a)| *o + a);
    let output = TensorData {
        data: a.0.borrow().data.clone().iter_mut().map(|o| o.tanh() ).collect(),
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
