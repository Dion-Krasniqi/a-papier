use std::ops::{Add, Mul};
use std::rc::Rc;
use std::cell::RefCell;
use rand::random;
use std::fs;
use std::io::prelude::*;

pub fn norm_random(mean: f32, std: f32) -> f32{
    let a = 2.*random::<f32>() -1.;
    let b = 2.*random::<f32>() -1.;
    let s: f32 = a.powf(2.) + b.powf(2.);
    if s >= 1. {return norm_random(mean,std)} else if s == 0.0 { return 0.0 };
    return mean + std * (a * (-2.0 * s.ln()/s).sqrt())
}
pub struct TensorData {
    pub data: Vec<f32>, // no matter the dim, the data is stored in order and shape "determines" dimension
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    // going to leave for now, for the same reason going to leave the current Tensor struct as is
    pub children: Vec<Tensor>,
}
pub struct Tensor(pub Rc<RefCell<TensorData>>);
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
pub fn save_model(params: &Vec<Tensor>, path: &str) {
    let mut file = fs::File::create(path).unwrap();
    for p in params {
        let data = p.data();
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