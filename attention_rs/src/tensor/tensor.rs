use std::ops::{Add, Mul};
use std::rc::Rc;
use std::cell::RefCell;
use rand::random;
use std::collections::HashMap;

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
            grad: vec![0.0f32;32],
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
            grad: vec![0.0f32;32],
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
            grad: vec![0.0f32;32],
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
pub fn add_forward_vec(a: Vec<Tensor>, b: Vec<Tensor>) -> Vec<Tensor> {
    let shape = a[0].0.borrow().shape.clone();
    let size: usize = shape.iter().product();
    let mut output = vec![Tensor::zero(shape.clone());a.len()];
    for i in 0..output.len() {
        if a[i].0.borrow().shape != b[i].0.borrow().shape {
                println!("Shapes are not compatibile, result is LHS.");
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
pub fn softmax_forward(a: Vec<Tensor>) -> Vec<Tensor> {
    let shape = a[0].shape();
    let size: usize = shape.iter().product();  
    let mut exp_sum = vec![0.0f32;a.len()];
    for (e, tensor) in exp_sum.iter_mut().zip(a.clone()) {
        *e = tensor.0.borrow().data.iter().map(|o|o.exp()).sum();
    }
    let mut output = vec![Tensor::zero(shape.clone());a.len()];
    for ((out, tensor), exp) in output.iter_mut().zip(a).zip(exp_sum) {
        let temp = TensorData {
            data: tensor.0.borrow().data.clone().iter().map(|o|(*o).exp()/exp).collect(),
            grad: vec![0.0f32;size],
            shape: shape.clone(),
            children: vec![tensor.clone()],
        };
        *out = Tensor(Rc::new(RefCell::new(temp)));
    };
    output
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
            a_grad[i*cols+j] += gamma * (1.0/(cols as f32 * varexs[i/cols].1)) * ((cols as f32) * out_grad[i*cols+j]
            - douti[i] - out_wo_gamma * douti_row[i]);

        }
    }
    // assigning
    betta.0.borrow_mut().grad = betta_grad;
    a.0.borrow_mut().grad = a_grad;
}
pub fn layernorm_forward_vec(a: Vec<Tensor>/*, betta: &Tensor, gamma: f32*/) -> Vec<Tensor> {
    let rows = a[0].0.borrow().shape[0];
    let cols = a[0].0.borrow().shape[1];
    let shape = a[0].shape();
    let a_vecs: Vec<Vec<f32>> = a.iter().map(|vec|vec.0.borrow().data.clone()).collect();
    let mut sums = vec![vec![(0.0f32,0.0f32);rows];a.len()];
    for (sum, a_vec) in sums.iter_mut().zip(a_vecs.clone()) {
        for i in 0..rows {
            for j in 0..cols {
                sum[i].0 += a_vec[i*cols+j];
                sum[i].1 += a_vec[i*cols+j].powf(2.0);
            }
            sum[i].0 /= cols as f32; //E[X]
            sum[i].1 /= cols as f32; //E[X^2]
            sum[i].1 -= sum[i].0.powf(2.0); //Var(x) = E[X^2] - E[X]^2
        }
    }
    let mut a_norms = vec![vec![0.0f32;rows*cols];a.len()];
    for ((a_norm, sum), a_vec) in a_norms.iter_mut().zip(sums).zip(a_vecs) {
        for i in 0..rows {
            for j in 0..cols {
                a_norm[i*cols+j] += (a_vec[i*cols+j] - sum[i].0)/(sum[i].1 + 1e-5).sqrt();
            }
        }
    }
    let mut output = vec![Tensor::zero(shape.clone());a.len()];
    for i in 0..a_norms.len() {
        let temp = TensorData {
            data: a_norms[i].clone(),
            grad: vec![0.0f32;rows*cols],
            shape: shape.clone(),
            children: vec![a[i].clone()],
        };
        output[i] = Tensor(Rc::new(RefCell::new(temp)));
    }
    output
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
pub fn cross_entropy_forward(a: &Tensor, targets: &[usize]) -> f32 {
    let cols = a.0.borrow().shape[1];
    let probs = a.0.borrow().data.clone();
    let mut loss = 0.0;
    for (i, &target) in targets.iter().enumerate() {
        loss -= probs[i*cols+target].ln();
    }
    loss / targets.len() as f32
}
pub fn cross_entropy_backward(a: &Tensor, targets: &[usize]) {
    let rows = a.0.borrow().shape[0];
    let cols = a.0.borrow().shape[1];
    for (i, &target) in targets.iter().enumerate() {
        a.0.borrow_mut().grad[i*cols+target] -= 1.0; 
    }
}
pub fn positional_encoding(seq_len: usize, emb_dim: usize) -> Tensor {
    let mut data = vec![0.0f32;seq_len*emb_dim];
    for i in 0..seq_len {
        for j in 0..emb_dim {
            data[i*emb_dim+j] = if j%2==0 {
                (j as f32/1000_f32.powf(2. * j as f32/emb_dim as f32)).sin()
            } else {
                (j as f32/1000_f32.powf(2. * j as f32/emb_dim as f32)).cos()
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
        // doesnt use nout for now
        let W_q = Tensor::rand(dim_shape.clone());
        let W_k = Tensor::rand(dim_shape.clone());
        let W_v = Tensor::rand(dim_shape.clone());
        AttentionHead { W_q, W_k, W_v }
    }
    pub fn forward(&self, x: Vec<Tensor>) -> Vec<Tensor> {
        let mut temp = Tensor::zero(x[0].shape());
        let mut result = x.clone();
        for (mut out, tensor) in result.iter_mut().zip(x) {
            let Q = matmul_forward(&tensor, &self.W_q); // Q = x @ W_q = (block_size x emb_dim) @ (emb_dim x head_dim) = Q(block_size * head_dim), dim_shape = emb_dim * head_dim
            let K = matmul_forward(&tensor, &self.W_k);
            let V = matmul_forward(&tensor, &self.W_v);
            let dk: usize= K.shape().iter().product();   
            let Q_Kt = matmul_forward(&Q, &transpose(&K)); // (block_size * head_dim) @ (head_dim * block_size)
            let scaling_factor = 1./((dk as f32).sqrt()); //Tensor::tensor(1./((dk as f32).sqrt()), Q_Kt.shape()); 
            let rows = Q_Kt.shape()[0];
            let cols = Q_Kt.shape()[1];
            for i in 0..rows {
                for j in i..cols {
                    Q_Kt.0.borrow_mut().data[i*cols+j] += f32::NEG_INFINITY;
                }
            }
            temp = softmax_forward(vec![(&Q_Kt * scaling_factor)])[0].clone();
            *out = matmul_forward(&temp, &V);
        }
        result
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        // eeh for now
        vec![self.W_q.clone(), self.W_k.clone(), self.W_v.clone()]
    }
}
pub struct MaskedAttentionHead {
    pub W_q: Tensor,
    pub W_k: Tensor,
    pub W_v: Tensor,
}
impl MaskedAttentionHead {
    pub fn new(dim_shape: Vec<usize>) -> MaskedAttentionHead {
        // doesnt use nout for now
        let W_q = Tensor::rand(dim_shape.clone());
        let W_k = Tensor::rand(dim_shape.clone());
        let W_v = Tensor::rand(dim_shape.clone());
        MaskedAttentionHead { W_q, W_k, W_v }
    }
    pub fn forward(&self, x: Vec<Tensor>) -> Vec<Tensor> {
        let mut temp = Tensor::zero(x[0].shape());
        let mut result = x.clone();
        for (mut out, tensor) in result.iter_mut().zip(x) {
            let Q = matmul_forward(&tensor, &self.W_q); // Q = x @ W_q = (block_size x emb_dim) @ (emb_dim x head_dim) = Q(block_size * head_dim), dim_shape = emb_dim * head_dim
            let K = matmul_forward(&tensor, &self.W_k);
            let V = matmul_forward(&tensor, &self.W_v);
            let dk: usize= K.shape().iter().product();   
            let Q_Kt = matmul_forward(&Q, &transpose(&K)); // (block_size * head_dim) @ (head_dim * block_size)
            let mask = Tensor::like_tensor(&Q_Kt);
            let scaling_factor = 1./((dk as f32).sqrt()); //Tensor::tensor(1./((dk as f32).sqrt()), Q_Kt.shape()); 
            let rows = Q_Kt.shape()[0];
            let cols = Q_Kt.shape()[1];
            for i in 0..rows {
                for j in i..cols {
                    Q_Kt.0.borrow_mut().data[i*cols+j] += f32::NEG_INFINITY;
                }
            }
            temp = softmax_forward(vec![(&Q_Kt * scaling_factor)])[0].clone();
            *out = matmul_forward(&temp, &V);
        }
        result
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        // eeh for now
        vec![self.W_q.clone(), self.W_k.clone(), self.W_v.clone()]
    }
}
pub struct FeedForward {
    pub weights: (Tensor, Tensor),
    pub biases: (Tensor, Tensor),
}
impl FeedForward {
    pub fn new(shape: Vec<usize>) -> FeedForward {
        let weights = (Tensor::rand(shape.clone()),Tensor::rand(shape.clone()));
        let biases = (Tensor::rand(shape.clone()),Tensor::rand(shape.clone()));
        FeedForward { weights, biases }
    }
    pub fn forward(&self, x: Vec<Tensor>) -> Vec<Tensor> {
        let mut result = x.clone();
        let mut temp = Tensor::zero(x[0].shape());
        for mut res in &result {
            temp = add_forward(&(&self.weights.0 * res), &self.biases.0); //f1
            temp = relu_forward(&temp); //ffn_1
            temp = &temp * &self.weights.1; //f2
            res = &add_forward(&temp, &self.biases.1);
        }
        result
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.push(self.weights.0.clone());
        params.push(self.weights.1.clone());
        params.push(self.biases.0.clone());
        params.push(self.biases.1.clone());
        params
    }
}
pub struct LinearLayer {
    pub weights: Tensor,
    pub biases: Tensor,
}
impl LinearLayer {
    pub fn new(shape: Vec<usize>) -> LinearLayer {
        let weights = Tensor::rand(shape.clone());
        let biases = Tensor::rand(shape.clone());
        LinearLayer { weights, biases }
    }
    pub fn forward(&self, x: Vec<Tensor>) -> Vec<Tensor> {
        let shape = x[0].shape();
        let mut data = vec![Tensor::zero(shape.clone());x.len()];
        let length = shape.iter().product();
        for (out, tensor) in data.iter_mut().zip(x) {
            for i in 0..length {
                out.0.borrow_mut().data[i] = tensor.0.borrow().data[i] * self.weights.0.borrow().data[i] + self.biases.0.borrow().data[i];
            }
        }
        data
    }
}
pub enum Layer {
    Attention(AttentionHead),
    MaskedAttention(MaskedAttentionHead),
    FeedForwardLayer(FeedForward),
    Linear(LinearLayer),
    Softmax,
    Add,
    Norm,
}
impl Layer {
    pub fn forward(&self, x: Vec<Tensor>) -> Vec<Tensor> {
        match self {
            Layer::Attention(a) => a.forward(x),
            Layer::MaskedAttention(a) => a.forward(x),
            Layer::FeedForwardLayer(a) => a.forward(x),
            Layer::Linear(a) => a.forward(x),
            Layer::Softmax => softmax_forward(x),
            Layer::Add => add_forward_vec(x.clone(), x), //just for now
            Layer::Norm => layernorm_forward_vec(x),
        }
    }
}
pub struct Stack {
    pub layers: Vec<Layer>,
}
impl Stack {
    pub fn new(layers: Vec<Layer>) -> Stack {
        Stack { layers }
    }
    pub fn add(&mut self, layer: Layer) {
        self.layers.push(layer);
    }
    pub fn forward(&self, x: Vec<Tensor>) -> Vec<Tensor> {
        // one in 
        let mut output = x.clone();
        for l in &self.layers {
            output = l.forward(output);
        }
        output
    }
}