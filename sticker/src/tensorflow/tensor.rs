use std::cmp::min;

use tensorflow::Tensor;

use crate::SentVec;

pub struct TensorBuilder {
    sequence: usize,
    token: usize,
    sequence_lens: Tensor<i32>,
    tokens: Tensor<f32>,
    token_subword: Tensor<i32>,
    subword_lens: Tensor<i32>,
    subwords: Tensor<f32>,
}

impl TensorBuilder {
    pub fn new(
        batch_size: usize,
        subword_batch_size: usize,
        time_steps: usize,
        subword_timesteps: usize,
        token_embed_dims: usize,
        char_embed_dims: usize,
    ) -> Self {
        TensorBuilder {
            sequence: 0,
            token: 0,
            sequence_lens: Tensor::new(&[batch_size as u64]),
            tokens: Tensor::new(&[
                batch_size as u64,
                time_steps as u64,
                token_embed_dims as u64,
            ]),
            token_subword: Tensor::new(&[batch_size as u64, time_steps as u64]),
            subword_lens: Tensor::new(&[subword_batch_size as u64]),
            subwords: Tensor::new(&[
                subword_batch_size as u64,
                subword_timesteps as u64,
                char_embed_dims as u64,
            ]),
        }
    }

    pub fn add(&mut self, input: &SentVec) {
        assert!((self.sequence as u64) < self.tokens.dims()[0]);

        self.add_tokens(&input.tokens);
        self.add_subwords(&input.subwords);

        self.sequence += 1;
    }

    fn add_subwords(&mut self, subwords: &[impl AsRef<[f32]>]) {
        assert!(self.token as u64 + subwords.len() as u64 <= self.subwords.dims()[0]);

        let max_seq_len = self.tokens.dims()[1] as usize;
        let max_subword_len = self.subwords.dims()[1] as usize;
        let subword_embed_size = self.subwords.dims()[2] as usize;

        for (subword_idx, subword) in subwords.iter().enumerate() {
            let subword = subword.as_ref();

            // Mapping of token to its subword in the subword batch.
            let token_offset = self.sequence * max_seq_len + subword_idx;
            self.token_subword[token_offset] = self.token as i32;

            // Determine the number of timesteps.
            let timesteps = min(max_subword_len, subword.len() / subword_embed_size);
            self.subword_lens[self.token] = timesteps as i32;

            // Copy representations.
            let subword_offset = self.token * max_subword_len * subword_embed_size;
            let subword_seq = &mut self.subwords
                [subword_offset..subword_offset + (subword_embed_size * timesteps)];
            subword_seq.copy_from_slice(&subword);

            self.token += 1;
        }
    }

    fn add_tokens(&mut self, tokens: &[f32]) {
        let max_seq_len = self.tokens.dims()[1] as usize;
        let token_embed_size = self.tokens.dims()[2] as usize;

        // Number of time steps to copy.
        let timesteps = min(max_seq_len, tokens.len() / token_embed_size);
        self.sequence_lens[self.sequence] = timesteps as i32;

        let token_offset = self.sequence * max_seq_len * token_embed_size;
        let token_seq =
            &mut self.tokens[token_offset..token_offset + (token_embed_size * timesteps)];
        token_seq.copy_from_slice(&tokens[..token_embed_size * timesteps]);
    }

    pub fn seq_lens(&self) -> &Tensor<i32> {
        &self.sequence_lens
    }

    pub fn subwords(&self) -> &Tensor<f32> {
        &self.subwords
    }

    pub fn subword_seq_lens(&self) -> &Tensor<i32> {
        &self.subword_lens
    }

    pub fn token_subword(&self) -> &Tensor<i32> {
        &self.token_subword
    }

    pub fn tokens(&self) -> &Tensor<f32> {
        &self.tokens
    }
}
