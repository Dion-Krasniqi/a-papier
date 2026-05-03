use crate::tensor::tensor::*;
use std::rc::Rc;
use std::cell::RefCell;

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