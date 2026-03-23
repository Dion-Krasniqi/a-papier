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
        println!("grade: {:?}", data.grad);
    }
}
impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor(Rc::clone(&self.0))
    }
}/*
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
    let vec_a = a.0.borrow().data.clone();
    let vec_b = b.0.borrow().data.clone();
    let m = a.0.borrow().shape[0].clone();
    
    let n = b.0.borrow().shape[0].clone();
    let p = b.0.borrow().shape[1].clone();
    let shape: Vec<usize> = vec![m, p];
    let size = m*p;
    let mut vec_out = vec![0.0f32;size];
    print!("{},{},{} \n", m,n,p);
    for i in 0..m{
        for j in 0..n {
            for k in 0..p {
                println!("{}",i*p+k);
                vec_out[i*p+k] += vec_a[i*m+j] * vec_b[j+k*n];
            }
        }
    }
    print!("{:?}", vec_out);
    let output = TensorData {
        data: vec_out,
        grad: vec![0.0f32;size],
        shape,
        children: vec![a.clone(), b.clone()],
    };
    return Tensor(Rc::new(RefCell::new(output)))
}