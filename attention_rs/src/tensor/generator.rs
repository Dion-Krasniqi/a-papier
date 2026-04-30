use crate::tensor::tensor::*;
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

    let input = &tokens[5..5+block_size];
    println!("{:?}", tokenizer.decode(input));
    let emb_x = embedding_forward(&input, &emb_w);
    let pos_emb_x: Vec<Tensor> = vec![add_forward(&emb_x, &pe)];

    let masked_x = masked_head.forward(&pos_emb_x);
        let add_1 = add_forward_vec(&pos_emb_x, &masked_x);
        
        let add_norm1 = norm_layer1.forward(&add_1);

        let ffn_x = ffn_layer.forward(&add_norm1);
        let add_2 = add_forward_vec(&add_norm1, &ffn_x);
        let add_norm2 = norm_layer2.forward(&add_2);

        let linear_x = linear_layer.forward(&add_norm2);
        let softmax_x = softmax_forward(&linear_x);
        let mut temp = (0.0f32, 0);
        println!("{:?}", softmax_x[0].shape());
        for (e,s) in softmax_x[0].data().iter().enumerate() {
            if *s > temp.0 { 
                temp.0 = *s;
                temp.1 = e; 
            }
        }
        println!("{}", temp.1);
        // eeeh
        println!("{:?}",tokenizer.decode(&[(temp.1 as i16 - ((temp.1/65) as i16)*65) as usize]));
}