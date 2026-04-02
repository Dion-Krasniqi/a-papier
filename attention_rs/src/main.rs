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
    let emb = embedding_forward(&tokens,&emb_w);
    let block_size: usize = 128; //dummy for now
    let pe = positional_encoding(block_size, emb_dim);
}