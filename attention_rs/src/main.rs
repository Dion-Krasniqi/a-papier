use std::ops::{Add, Mul, Div};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

mod tensor;
mod scalar;
use crate::scalar::neuron::definitions::*;
use crate::scalar::neuron::definitions::NONL as NONL;
use crate::scalar::value::definitions::*;
use crate::tensor::tensor::*;

fn main(){
    // since just character level for now
    let text = std::fs::read_to_string("src/tensor/data.txt").unwrap();
    let tokenizer = Tokenizer::new(&text);
    let tokens = tokenizer.encode(&text);
    let vocab_size = tokenizer.get_vocab_len();

    let emb_dim: usize = 10;
    let emb_w = Tensor::rand(vec![vocab_size, emb_dim]);
    let block_size: usize = 8; //dummy for now
    let x = &tokens[..block_size];
    let emb_x = embedding_forward(x, &emb_w); // x.len() * emb_w.cols = block_size * emb_dim
    let pe = positional_encoding(block_size, emb_dim);
    let pos_emb_x = add_forward(&emb_x, &pe);

    let head_dim = emb_dim;

    let W_q = Tensor::rand(vec![emb_dim, head_dim]);
    let W_k = Tensor::rand(vec![emb_dim, head_dim]);
    let W_v = Tensor::rand(vec![emb_dim, head_dim]);
    let Q = matmul_forward(&pos_emb_x, &W_q); // Q = pos_emb @ W_q = (block_size x emb_dim) @ (emb_dim x head_dim) = Q(block_size * head_dim)
    let K = matmul_forward(&pos_emb_x, &W_k);
    let V = matmul_forward(&pos_emb_x, &W_v);
    let dk: usize= K.shape().iter().product();

    let Q_Kt = matmul_forward(&Q, &transpose(&K)); // (block_size * head_dim) @ (head_dim * block_size)
    let scaling_factor = Tensor::tensor(1./((dk as f32).sqrt()), Q_Kt.shape());
    let softmaxed = softmax_forward(&matmul_forward(&Q_Kt, &scaling_factor));
    let attention = matmul_forward(&softmaxed, &V);
    println!("{:?}", attention.shape());
}