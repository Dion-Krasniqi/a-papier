use std::ops::{Add, Mul};
use std::rc::Rc;
use std::cell::RefCell;

struct TensorData {
    data: Vec<f32>, // no matter the dim, the data is stored in order and shape "determines" dimension
    grad: Vec<f32>,
    shape: Vec<usize>,
}

pub struct Tensor(Rc<RefCell<TensorData>>);

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor(Rc::clone(&self.0))
    }
}
impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        let shape = self.0.borrow().shape.clone();
        let size: usize = shape.iter().product();
        let output = TensorData {
            data: self.0.borrow().data.iter()
                  .zip(other.0.borrow().data.iter())
                  .map(|(a,b)| a+b ).collect(),
            grad: vec![0.0f32; size],
            shape: shape,
        };
        Tensor(Rc::new(RefCell::new(output)))
    }
}