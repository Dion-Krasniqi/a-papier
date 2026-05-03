use crate::tensor::tensor::*;
use crate::tensor::operations::*;

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
    pub weights: Tensor,
    pub biases: Tensor,
}
impl LinearLayer {
    pub fn new(shape: Vec<usize>, block_size: usize) -> LinearLayer {
        // matmul(x, weights) = (block_size x emb_dim) @ (emb_dim x vocab_size) = block_size x vocab_size
        // add(matmul, bias) = (block_size x vocab_size) + (block_size x vocab_size)
        let weights = Tensor::rand(shape.clone());//vec![Tensor::rand(shape.clone());in_len];
        let biases = Tensor::rand(vec![block_size, shape.clone()[1]]);//vec![Tensor::rand(vec![block_size, shape.clone()[1]]);in_len];
        LinearLayer { weights, biases}
    }
    pub fn forward(&self, a: &Vec<Tensor>) -> Vec<Tensor> {
        a.iter().map(|x|add_forward(&matmul_forward(x, &self.weights), &self.biases)).collect()
    }
    pub fn backward(&self, out: &Vec<Tensor>, a: &Vec<Tensor>) {
            for i in 0..out.len() {
                let matmul_res = matmul_forward(&a[i], &self.weights);
                add_backward(&out[i], &matmul_res, &self.biases);
                matmul_backward(&matmul_res, &a[i], &self.weights);
            }
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.biases.clone()]
    }
}
pub struct LayerNorm {
    pub betta: Tensor,
    pub gamma: f32,
}
impl LayerNorm {
    pub fn new(shape: Vec<usize>, value: f32) -> LayerNorm {
        LayerNorm {
            betta: Tensor::zero(shape.clone()),
            gamma: value,
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