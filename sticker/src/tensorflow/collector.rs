use conllx::Sentence;
use failure::{format_err, Error};
use tensorflow::Tensor;

use crate::{Collector, Numberer, SentVectorizer};

pub struct CollectedTensors {
    pub sequence_lens: Vec<Tensor<i32>>,
    pub subwords: Vec<Tensor<f32>>,
    pub subword_seq_lens: Vec<Tensor<i32>>,
    pub token_subword: Vec<Tensor<i32>>,
    pub tokens: Vec<Tensor<f32>>,
    pub labels: Vec<Tensor<i32>>,
}

pub struct TensorCollector {
    numberer: Numberer<String>,
    vectorizer: SentVectorizer,
    batch_size: usize,
    sequence_lens: Vec<Tensor<i32>>,
    subwords: Vec<Tensor<f32>>,
    subword_seq_lens: Vec<Tensor<i32>>,
    token_subword: Vec<Tensor<i32>>,
    tokens: Vec<Tensor<f32>>,
    labels: Vec<Tensor<i32>>,
    cur_labels: Vec<Vec<i32>>,
    cur_token_lens: Vec<usize>,
    cur_subwords: Vec<Vec<f32>>,
    cur_tokens: Vec<Vec<f32>>,
}

impl TensorCollector {
    pub fn new(batch_size: usize, numberer: Numberer<String>, vectorizer: SentVectorizer) -> Self {
        TensorCollector {
            batch_size,
            numberer,
            vectorizer,
            labels: Vec::new(),
            subwords: Vec::new(),
            subword_seq_lens: Vec::new(),
            token_subword: Vec::new(),
            tokens: Vec::new(),
            sequence_lens: Vec::new(),
            cur_labels: Vec::new(),
            cur_token_lens: Vec::new(),
            cur_subwords: Vec::new(),
            cur_tokens: Vec::new(),
        }
    }

    fn finalize_batch(&mut self) {
        if self.cur_labels.is_empty() {
            return;
        }

        let batch_size = self.cur_labels.len();
        let mut batch_seq_lens = Tensor::new(&[batch_size as u64]);
        self.cur_labels
            .iter()
            .enumerate()
            .for_each(|(idx, labels)| batch_seq_lens[idx] = labels.len() as i32);

        let max_seq_len = self.cur_labels.iter().map(Vec::len).max().unwrap_or(0);
        let token_dims = self.cur_tokens[0].len() / self.cur_labels[0].len();

        let mut batch_tokens =
            Tensor::new(&[batch_size as u64, max_seq_len as u64, token_dims as u64]);
        let mut batch_labels = Tensor::new(&[batch_size as u64, max_seq_len as u64]);
        let mut batch_token_subword = Tensor::new(&[batch_size as u64, max_seq_len as u64]);

        let mut n_tokens = 0;
        for i in 0..batch_size {
            let offset = i * max_seq_len;
            let token_offset = offset * token_dims;
            let seq_len = self.cur_labels[i].len();

            batch_tokens[token_offset..token_offset + token_dims * seq_len]
                .copy_from_slice(&self.cur_tokens[i]);
            batch_labels[offset..offset + seq_len].copy_from_slice(&self.cur_labels[i]);

            for token_idx in 0..seq_len {
                batch_token_subword[i * max_seq_len + token_idx] = n_tokens;
                n_tokens += 1;
            }
        }

        // Subword lens
        let subword_batch_size = self.cur_subwords.len() as u64;
        let mut batch_subword_seq_lens = Tensor::new(&[subword_batch_size]);
        self.cur_token_lens
            .iter()
            .enumerate()
            .for_each(|(idx, &len)| batch_subword_seq_lens[idx] = len as i32);

        // Subwords batch
        let char_dims = self.cur_subwords[0].len() / self.cur_token_lens[0];
        let max_subword_len = self.cur_token_lens.iter().max().cloned().unwrap_or(0);
        let mut batch_subwords =
            Tensor::new(&[subword_batch_size, max_subword_len as u64, char_dims as u64]);

        for (token_idx, (token, &token_len)) in self
            .cur_subwords
            .iter()
            .zip(&self.cur_token_lens)
            .enumerate()
        {
            let offset = token_idx * max_subword_len * char_dims;
            batch_subwords[offset..offset + token_len as usize * char_dims].copy_from_slice(token);
        }

        self.sequence_lens.push(batch_seq_lens);
        self.subwords.push(batch_subwords);
        self.subword_seq_lens.push(batch_subword_seq_lens);
        self.token_subword.push(batch_token_subword);
        self.tokens.push(batch_tokens);
        self.labels.push(batch_labels);

        self.cur_labels.clear();
        self.cur_token_lens.clear();
        self.cur_subwords.clear();
        self.cur_tokens.clear();
    }

    pub fn into_parts(mut self) -> CollectedTensors {
        self.finalize_batch();

        CollectedTensors {
            sequence_lens: self.sequence_lens,
            subwords: self.subwords,
            subword_seq_lens: self.subword_seq_lens,
            token_subword: self.token_subword,
            tokens: self.tokens,
            labels: self.labels,
        }
    }
}

impl Collector for TensorCollector {
    fn collect(&mut self, sentence: &Sentence) -> Result<(), Error> {
        if self.cur_labels.len() == self.batch_size {
            self.finalize_batch();
        }

        let input = self.vectorizer.realize(sentence)?;
        let mut labels = Vec::with_capacity(sentence.len());
        let mut token_lens = Vec::with_capacity(sentence.len());
        for token in sentence {
            let pos_tag = token
                .pos()
                .ok_or(format_err!("Token without a part-of-speech tag: {}", token))?;
            labels.push(self.numberer.add(pos_tag.to_owned()) as i32);
            token_lens.push(token.form().chars().count());
        }

        let (subwords, tokens) = input.into_inner();
        self.cur_subwords.extend_from_slice(&subwords);
        self.cur_tokens.push(tokens);
        self.cur_labels.push(labels);
        self.cur_token_lens.extend_from_slice(token_lens.as_slice());

        Ok(())
    }

    fn vectorizer(&self) -> &SentVectorizer {
        &self.vectorizer
    }
}
