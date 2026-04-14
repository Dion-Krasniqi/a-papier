use std::ops::{Add, Mul, Div};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

mod tensor;
// mod scalar;
// use crate::scalar::neuron::definitions::*;
// use crate::scalar::neuron::definitions::NONL as NONL;
// use crate::scalar::value::definitions::*;
use crate::tensor::tensor::*;
use crate::Layer::*;

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
    let head_dim = emb_dim;
    let emb_x = embedding_forward(x, &emb_w); // x.len() * emb_w.cols = block_size * emb_dim
    let pe = positional_encoding(block_size, emb_dim);
    let pos_emb_x = add_forward(&emb_x, &pe);

    let mut layer_stack = Stack::new(vec![
        ResidualMaskedAttention(MaskedAttentionHead::new(vec![emb_dim,head_dim])),
        Norm(LayerNorm::new(vec![block_size,head_dim])),
        ResidualAttention(AttentionHead::new(vec![emb_dim,head_dim])),
        Norm(LayerNorm::new(vec![block_size,head_dim])),
        ResidualFFN(FeedForward::new(vec![block_size,head_dim],x.len())),
        Norm(LayerNorm::new(vec![block_size,head_dim])),
        Linear(LinearLayer::new(vec![block_size,emb_dim])),
        Softmax,
        ]
    );
    layer_stack.forward(vec![pos_emb_x.clone()]);
    


}