use std::hash::Hash;
use std::mem;

use conllx::graph::Sentence;
use failure::Error;
use tch::{Device, Kind, Tensor};

use crate::{Collector, Numberer, SentVectorizer, SentenceEncoder};

//use super::tensor::{LabelTensor, TensorBuilder};

pub struct CollectedTensors {
    pub sequence_lens: Vec<Tensor>,
    pub inputs: Vec<Tensor>,
    pub labels: Vec<Tensor>,
}

pub struct TensorCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Eq + Hash,
{
    encoder: E,
    numberer: Numberer<E::Encoding>,
    vectorizer: SentVectorizer,
    batch_size: usize,
    sequence_lens: Vec<Tensor>,
    inputs: Vec<Tensor>,
    labels: Vec<Tensor>,
    cur_labels: Vec<Vec<i32>>,
    cur_inputs: Vec<Vec<f32>>,
}

impl<E> TensorCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Eq + Hash,
{
    pub fn new(
        batch_size: usize,
        encoder: E,
        numberer: Numberer<E::Encoding>,
        vectorizer: SentVectorizer,
    ) -> Self {
        TensorCollector {
            batch_size,
            encoder,
            numberer,
            vectorizer,
            labels: Vec::new(),
            inputs: Vec::new(),
            sequence_lens: Vec::new(),
            cur_labels: Vec::new(),
            cur_inputs: Vec::new(),
        }
    }

    fn finalize_batch(&mut self) {
        if self.cur_labels.is_empty() {
            return;
        }

        let batch_size = self.cur_labels.len() as i64;
        let max_seq_len = self.cur_labels.iter().map(Vec::len).max().unwrap_or(0) as i64;
        let input_dims = (self.cur_inputs[0].len() / self.cur_labels[0].len()) as i64;

        let batch_inputs = Tensor::zeros(
            &[batch_size, max_seq_len, input_dims],
            (Kind::Float, Device::Cpu),
        );
        let batch_labels = Tensor::zeros(&[batch_size, max_seq_len], (Kind::Float, Device::Cpu));

        let mut cur_inputs = Vec::new();
        let mut cur_labels = Vec::new();
        mem::swap(&mut cur_inputs, &mut self.cur_inputs);
        mem::swap(&mut cur_labels, &mut self.cur_labels);

        let mut seq_lens = Vec::new();
        for (batch_idx, (inputs, labels)) in cur_inputs.into_iter().zip(cur_labels).enumerate() {
            let mut input_tensor =
                batch_inputs
                    .narrow(0, batch_idx as i64, 1)
                    .narrow(1, 0, labels.len() as i64);
            input_tensor.copy_(&Tensor::of_slice(&inputs).reshape(&[
                1,
                labels.len() as i64,
                input_dims,
            ]));

            let mut label_tensor =
                batch_labels
                    .narrow(0, batch_idx as i64, 1)
                    .narrow(1, 0, labels.len() as i64);
            label_tensor.copy_(&Tensor::of_slice(&labels).reshape(&[1, labels.len() as i64]));

            seq_lens.push(labels.len() as i32);
        }

        let batch_seq_lens = Tensor::of_slice(&seq_lens);

        self.sequence_lens.push(batch_seq_lens);
        self.inputs.push(batch_inputs);
        self.labels.push(batch_labels);
    }

    pub fn into_parts(mut self) -> CollectedTensors {
        self.finalize_batch();

        CollectedTensors {
            sequence_lens: self.sequence_lens,
            inputs: self.inputs,
            labels: self.labels,
        }
    }
}

impl<E> Collector for TensorCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    fn collect(&mut self, sentence: &Sentence) -> Result<(), Error> {
        if self.cur_labels.len() == self.batch_size {
            self.finalize_batch();
        }

        let input = self.vectorizer.realize(sentence)?;
        let mut labels = Vec::with_capacity(sentence.len());

        for encoding in self.encoder.encode(&sentence)? {
            labels.push(self.numberer.add(encoding) as i32);
        }

        self.cur_inputs.push(input);
        self.cur_labels.push(labels);

        Ok(())
    }

    fn vectorizer(&self) -> &SentVectorizer {
        &self.vectorizer
    }
}
