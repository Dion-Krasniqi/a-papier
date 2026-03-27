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
    // add seed option
    pub fn tensor_rand(shape: Vec<usize>) -> Tensor {
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
    pub fn tensor_zero(shape: Vec<usize>) -> Tensor {
        Tensor::tensor(0.0, shape)
    }
    pub fn tensor_one(shape: Vec<usize>) -> Tensor {
        Tensor::tensor(1.0, shape)
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
                out_vec[i*p+k] += a_vec[i*n+j] * b_vec[k*n+j];
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
    a.0.borrow_mut().grad = a_grad;
    b.0.borrow_mut().grad = b_grad;
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
pub fn softmax_forward(a: &Tensor) -> Tensor {
    let size: usize = a.0.borrow().shape.iter().product();  
    let exp_sum: f32 = a.0.borrow().data.iter().map(|o|o.exp()).sum();
    let output = TensorData {
        data: a.0.borrow().data.clone().iter().map(|o|(*o).exp()/exp_sum).collect(),
        grad: vec![0.0f32;size],
        shape: a.0.borrow().shape.clone(),
        children: vec![a.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}
pub fn softmax_backward(out: &Tensor, a: &Tensor) {
    let exp_sum: f32 = a.0.borrow().data.iter().map(|o|o.exp()).sum();
    // fx = e(ai)/exp_sum, a/h - d(a/h)/dx -> d(a/h)/dx = a'h - ah' / h^2
    // when fx at i and ai
    // d(fx)/d(ai) = e(ai) * sum - (sum')* e(ai) / (sum^2), d(sum)/d(ai) = (e(a1) + e(a2) + ... + e(ai) + ... + e(an)) * e(ai) = e(ai)
    // d(fx)/d(ai) = e(ai)*sum - e(ai)^2 / sum^2
    // when fx at i and a j (j!=i)
    // d(fx)/d(aj) = e(ai)' * sum - (sum') * e(ai) / sum^2, e(ai)' = 0, since its d(fx) per d(aj) so e(ai) as const
    // d(fx)/d(aj) = 0 - e(aj)*e(ai)/ sum^2 -> e(aj)*e(ai)/sum^2
    let a_data = a.0.borrow().data.clone();
    let out_data = out.0.borrow().data.clone();
    let out_grad = out.0.borrow().grad.clone();
    //matrix
    let squared_sum = exp_sum.powf(2.0);
    for i in 0..out_data.len(){
        for j in 0..out_data.len(){
            if i==j {
                a.0.borrow_mut().grad[i] += (a_data[i] * exp_sum - a_data[i].powf(2.0))
                / squared_sum * out_grad[i];
            } else {
                a.0.borrow_mut().grad[i] += (a_data[j] * a_data[i]) / squared_sum * out_grad[j];
            }
        }
    }
}
// matrix
pub fn mean(a: &Tensor, dim: usize) -> Tensor{
    // keeps shape
    // 0 over all, 1 over rows, 2 over cols
    let mut a_vec = a.0.borrow().data.clone();
    let rows = a.0.borrow().shape[0];
    let cols = a.0.borrow().shape[1];
    match dim {
        0 => {
            let mut sum = 0.0;
            for a in a_vec.iter() {
                sum += *a;
            };
            let output = Tensor::tensor(sum/(a_vec.len() as f32), [rows,cols].to_vec());
            output

        }
        1 => {
            let mut sums = vec![0.0f32;rows];
            for i in 0..rows {
                for j in 0..cols {
                    sums[i] += a_vec[i*cols+j];
                    if i>0 {
                        a_vec[(i-1)*cols + j] = sums[i-1];
                    } 
                }
                sums[i] /= cols as f32;
            };
            for i in 0..cols {
                a_vec[i] = sums[0];
            }
            let output = TensorData {
                data: a_vec,
                grad: vec![0.0f32;rows*cols],
                shape: [rows,cols].to_vec(),
                children: vec![a.clone()],
            };
            Tensor(Rc::new(RefCell::new(output)))
        }
        _ => {
            let mut sums = vec![0.0f32;cols];
            for i in 0..cols {
                for j in 0..rows {
                    sums[i] += a_vec[i*rows+j];
                    if i>0 {
                        a_vec[(i-1)*rows + j] = sums[i-1];
                    } 
                }
                sums[i] /= rows as f32;
            };
            for i in 0..rows {
                a_vec[i+cols] = sums[0];
            }
            let output = TensorData {
                data: a_vec,
                grad: vec![0.0f32;rows*cols],
                shape: [rows,cols].to_vec(),
                children: vec![a.clone()],
            };
            Tensor(Rc::new(RefCell::new(output)))
        }

    }
}
pub fn layernorm_forward(a: &Tensor, betta: f32, gamma: f32) -> Tensor {
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
            a_norm[i*cols+j] += gamma*(a_vec[i*cols+j] - sum[i].0)/(sum[i].1 + 1e-5).sqrt() + betta;
        }
    }
    let output = TensorData {
        data: a_norm,
        grad: vec![0.0f32;rows*cols],
        shape: a.0.borrow().shape.clone(),
        children: vec![a.clone()],
    };
    Tensor(Rc::new(RefCell::new(output)))
}