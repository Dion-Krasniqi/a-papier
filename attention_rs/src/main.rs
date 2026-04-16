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
    let pos_emb_x = vec![add_forward(&emb_x, &pe)];
    
    let masked_head = MaskedAttentionHead::new(vec![emb_dim,head_dim]);
    let norm_layer = LayerNorm::new(vec![block_size,head_dim]);
    let attention_head = AttentionHead::new(vec![emb_dim,head_dim]);
    let mut ffn_layer = FeedForward::new(vec![block_size,head_dim], x.len());
    let linear_layer = LinearLayer::new(vec![block_size,emb_dim]);

    let masked_x = masked_head.forward(&pos_emb_x);
    let add_1 = add_forward_vec(&pos_emb_x, &masked_x);
    
    let add_norm1 = norm_layer.forward(&add_1);

    let attention_x = attention_head.forward(&add_norm1);
    let add_2 = add_forward_vec(&add_norm1, &attention_x);
    let add_norm2 = norm_layer.forward(&add_2);

    let ffn_x = ffn_layer.forward(&add_norm2);
    let add_3 = add_forward_vec(&add_norm2, &ffn_x);
    let add_norm3 = norm_layer.forward(&add_3);

    let linear_x = linear_layer.forward(&add_norm3);
    let softmax_x = softmax_forward(&linear_x);
    let loss = cross_entropy_forward(&softmax_x, &[3, 1, 8, 0, 1]); // dummy
    masked_x[0].print();
    loss.set_grad(1.0);
    cross_entropy_backward(&softmax_x, &[0, 1, 0, 0, 0]);
    softmax_backward(&softmax_x, &linear_x); 
    linear_layer.backward(&linear_x, &add_norm3);
    norm_layer.backward(&add_norm3, &add_3);
    add_backward_vec(&add_3, &add_norm2, &ffn_x);
    ffn_layer.backward(&ffn_x, &add_norm2);
    norm_layer.backward(&add_norm2, &add_2);
    add_backward_vec(&add_2, &add_norm1, &attention_x);
    attention_head.backward(&attention_x, &add_norm1);
    norm_layer.backward(&add_norm1, &add_1);
    add_backward_vec(&add_1, &pos_emb_x, &masked_x);
    masked_head.backward(&masked_x, &pos_emb_x);
    masked_x[0].print();
    // adjust params
    
    let n = (0.9 * text.len() as f32) as usize;
    let train_data = &tokens[..n];
    let val_data = &tokens[n..];


}