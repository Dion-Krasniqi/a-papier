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

    let attention_layer = AttentionHead::new(vec![emb_dim,head_dim], 1);
    let attention_head = attention_layer.forward(&pos_emb_x);
    // just one 
    let att_emb_x = add_forward(&pos_emb_x, &attention_head[0]);

    let betta = Tensor::zero(vec![block_size,emb_dim]);
    let norm_att_x = layernorm_forward(&att_emb_x, &betta, 0.0);

    // less stupid setup for now
    let ffn_layer = FeedForward::new(1, att_emb_x.shape());
    let ffn_out = ffn_layer.forward(&att_emb_x);
    let f_norm_x = layernorm_forward(&ffn_out[0], &betta, 0.0);

}