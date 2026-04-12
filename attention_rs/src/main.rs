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

    // let masked_attention_layer = MaskedAttentionHead::new(vec![emb_dim,head_dim], 1);
    // let masked_attention_head = masked_attention_layer.forward(&pos_emb_x);
    // // just one 
    // let m_att_emb_x = add_forward(&pos_emb_x, &masked_attention_head[0]);

    // let betta = Tensor::zero(vec![block_size,emb_dim]);
    // // att_emb_x * 1.0 + 0.0
    // let norm_m_att_x = layernorm_forward(&m_att_emb_x, &betta, 1.0);

    // let attention_layer = AttentionHead::new(vec![emb_dim, head_dim], 1);
    // let attention_head = attention_layer.forward(&norm_m_att_x);
    // let att_emb_x = add_forward(&norm_m_att_x, &attention_head[0]);
    // let norm_att_x = layernorm_forward(&att_emb_x, &betta, 1.0); // same as above

    // // less stupid setup for now
    // let ffn_layer = FeedForward::new( norm_att_x.shape());
    // let ffn_out = ffn_layer.forward(&norm_att_x);
    // let ffn_x = add_forward(&norm_att_x, &ffn_out[0]);
    // let norm_ffn_x = layernorm_forward(&ffn_x, &betta, 0.0);
    // let linear_layer = LinearLayer::new(norm_ffn_x.shape());
    // let linear_ffn_x = linear_layer.forward(&norm_ffn_x); 
    // let softmax_linear_x = softmax_forward(&linear_ffn_x[0]);

    let layer_stack = Stack::new(vec![
        MaskedAttention(MaskedAttentionHead::new(vec![emb_dim,head_dim])),
        // add
        Norm,
        Attention(AttentionHead::new(vec![emb_dim,head_dim])),
        // add
        Norm,
        FeedForwardLayer(FeedForward::new(vec![emb_dim,head_dim])),
        // add
        Norm,
        Linear(LinearLayer::new(vec![block_size, emb_dim])),
        Softmax,
        ]);
    


}