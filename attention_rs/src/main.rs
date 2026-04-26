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
    let n = (0.9 * text.len() as f32) as usize;
    let train_data = &tokens[..n];
    let val_data = &tokens[n..];
    let mut data = Vec::new();
    for i in 0..100 {
        data.push((&train_data[i..block_size+i],&train_data[i+1..block_size+i+1]));
    }
    let head_dim = emb_dim;
    let pe = positional_encoding(block_size, emb_dim);
    
    
    let masked_head = MaskedAttentionHead::new(vec![emb_dim,head_dim]);
    let norm_layer1 = LayerNorm::new(vec![block_size,head_dim]);
    let norm_layer2 = LayerNorm::new(vec![block_size,head_dim]);
    let mut ffn_layer = FeedForward::new(vec![block_size,head_dim], data.len());
    let linear_layer = LinearLayer::new(vec![emb_dim,vocab_size], block_size, data.len());
    let mut params: Vec<Tensor> = vec![emb_w.clone(), norm_layer1.betta.clone(),norm_layer2.betta.clone()];
    params.extend(masked_head.parameters());
    params.extend(ffn_layer.parameters());
    params.extend(linear_layer.parameters());
    let y : Vec<&[usize]> = data.iter().map(|(x)|x.1).collect();
    for _ in 0..100000{
        for p in &params {
            p.set_grad(0.0);
        }
        let emb_x: Vec<Tensor> = data.iter().map(|x| embedding_forward(&x.0, &emb_w)).collect(); // x.len() * emb_w.cols = block_size * emb_dim
        let pos_emb_x: Vec<Tensor> = emb_x.iter().map(|e|add_forward(&e,&pe)).collect();

        let masked_x = masked_head.forward(&pos_emb_x);
        let add_1 = add_forward_vec(&pos_emb_x, &masked_x);
        
        let add_norm1 = norm_layer1.forward(&add_1);

        let ffn_x = ffn_layer.forward(&add_norm1);
        let add_2 = add_forward_vec(&add_norm1, &ffn_x);
        let add_norm2 = norm_layer2.forward(&add_2);

        let linear_x = linear_layer.forward(&add_norm2);
        let softmax_x = softmax_forward(&linear_x);
        let loss = cross_entropy_forward(&softmax_x, &y);
        println!(
            "{:?}", loss.data()
        );
        loss.set_grad(1.0);
        cross_entropy_backward(&softmax_x, &y);
        softmax_backward(&softmax_x, &linear_x); 
        linear_layer.backward(&linear_x, &add_norm2);
        norm_layer2.backward(&add_norm2, &add_2);
        add_backward_vec(&add_2, &add_norm1, &ffn_x);
        ffn_layer.backward(&ffn_x, &add_norm1);
        norm_layer1.backward(&add_norm1, &add_1);
        add_backward_vec(&add_1, &pos_emb_x, &masked_x);
        masked_head.backward(&masked_x, &pos_emb_x);
        pos_emb_x.iter().zip(&emb_x).for_each(|(p,e)|add_backward(&p,&e,&pe));
        emb_x.iter().zip(&data).for_each(|(e, x)| embedding_backward(&e, &emb_w, &x.0));
        // adjust weights
        for p in &params {
            p.adjust_data(-0.001);
        }
    }
}