use crate::tensor::tensor::*;
use rand::random;


pub fn sample(probs: &[f32]) -> usize {
    let mut temp = 0.0f32;
    let mut idx = 0usize;
    for (e,p) in probs.iter().enumerate() {
        let mut r = random::<f32>();
        r -= p;
        if r <= 0.0 {
            return e
        }
    }
    probs.len() - 1
}
pub fn generator() {
    let text = std::fs::read_to_string("src/tensor/data.txt").unwrap();
    let tokenizer = Tokenizer::new(&text);
    let tokens = tokenizer.encode(&text);
    let vocab_size = tokenizer.get_vocab_len();

    let emb_dim: usize = 10;
    let emb_w = Tensor::rand(vec![vocab_size, emb_dim]);
    let block_size: usize = 64;
    let head_dim = emb_dim;
    let pe = positional_encoding(block_size, emb_dim);
    
    
    let masked_head = MaskedAttentionHead::new(vec![emb_dim,head_dim]);
    let norm_layer1 = LayerNorm::new(vec![block_size,head_dim]);
    let norm_layer2 = LayerNorm::new(vec![block_size,head_dim]);
    let mut ffn_layer = FeedForward::new(vec![block_size,head_dim], 1);
    let linear_layer = LinearLayer::new(vec![emb_dim,vocab_size], block_size, 1);
    let mut params: Vec<Tensor> = vec![emb_w.clone(), norm_layer1.betta.clone(),norm_layer2.betta.clone()];
    params.extend(masked_head.parameters());
    params.extend(ffn_layer.parameters());
    params.extend(linear_layer.parameters());

    load_model(&params,"model.bin");

    let mut input: Vec<usize> = tokens[5..5+block_size].to_vec();
    for i in 0..200 {
        let emb_x = embedding_forward(&input[input.len()-block_size..], &emb_w);
        let pos_emb_x: Vec<Tensor> = vec![add_forward(&emb_x, &pe)];

        let masked_x = masked_head.forward(&pos_emb_x);
        let add_1 = add_forward_vec(&pos_emb_x, &masked_x);
            
        let add_norm1 = norm_layer1.forward(&add_1);

        let ffn_x = ffn_layer.forward(&add_norm1);
        let add_2 = add_forward_vec(&add_norm1, &ffn_x);
        let add_norm2 = norm_layer2.forward(&add_2);

        let linear_x = linear_layer.forward(&add_norm2);
        let softmax_x = softmax_forward(&linear_x);
        let last_row = (block_size - 1)  * vocab_size;
        let probs = &softmax_x[0].data()[last_row..last_row+block_size];
        let next_token = sample(probs);
        input.push(next_token);
    }
    println!("{:?}", tokenizer.decode(&input));
}