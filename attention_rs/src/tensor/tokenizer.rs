use std::collections::HashMap;
pub struct Tokenizer {
    vocab: HashMap<char, usize>,
    reverse: HashMap<usize, char>,
}
impl Tokenizer {
    pub fn new(text: &str) -> Tokenizer {
        let mut vocab = HashMap::new();
        let mut reverse = HashMap::new();
        let mut id = 0;
        for t in text.chars() {
            if !vocab.contains_key(&t) {
                vocab.insert(t, id);
                reverse.insert(id, t);
                id += 1;
            }
        }
        Tokenizer { vocab, reverse}
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|t| self.vocab[&t]).collect()
    }
    pub fn decode(&self, token: &[usize]) -> String {
        token.iter().map(|t| self.reverse[&t]).collect()
    }
    pub fn get_vocab_len(&self) -> usize {
        self.vocab.len()
    }
}