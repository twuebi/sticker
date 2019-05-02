use std::cmp::min;

use tensorflow::Tensor;

mod labels {
    pub trait Labels {
        fn from_shape(batch_size: u64, time_steps: u64) -> Self;
    }
}

#[derive(Default)]
pub struct NoLabels;

impl labels::Labels for NoLabels {
    fn from_shape(_batch_size: u64, _time_steps: u64) -> Self {
        NoLabels
    }
}

pub struct LabelTensor(Tensor<i32>);

impl labels::Labels for LabelTensor {
    fn from_shape(batch_size: u64, time_steps: u64) -> Self {
        LabelTensor(Tensor::new(&[batch_size, time_steps]))
    }
}

pub struct TensorBuilder<L> {
    sequence: usize,
    sequence_lens: Tensor<i32>,
    inputs: Tensor<f32>,
    labels: L,
}

impl<L> TensorBuilder<L>
where
    L: labels::Labels,
{
    pub fn new(batch_size: usize, time_steps: usize, inputs_size: usize) -> Self {
        TensorBuilder {
            sequence: 0,
            sequence_lens: Tensor::new(&[batch_size as u64]),
            inputs: Tensor::new(&[batch_size as u64, time_steps as u64, inputs_size as u64]),
            labels: L::from_shape(batch_size as u64, time_steps as u64),
        }
    }
}

impl<L> TensorBuilder<L> {
    fn add_input(&mut self, input: &[f32]) {
        assert!((self.sequence as u64) < self.inputs.dims()[0]);

        let max_seq_len = self.inputs.dims()[1] as usize;
        let token_embed_size = self.inputs.dims()[2] as usize;

        // Number of time steps to copy.
        let timesteps = min(max_seq_len, input.len() / token_embed_size);
        self.sequence_lens[self.sequence] = timesteps as i32;

        let token_offset = self.sequence * max_seq_len * token_embed_size;
        let token_seq =
            &mut self.inputs[token_offset..token_offset + (token_embed_size * timesteps)];
        token_seq.copy_from_slice(&input[..token_embed_size * timesteps]);
    }

    pub fn seq_lens(&self) -> &Tensor<i32> {
        &self.sequence_lens
    }

    pub fn inputs(&self) -> &Tensor<f32> {
        &self.inputs
    }

    pub fn labels(&self) -> &L {
        &self.labels
    }
}

impl TensorBuilder<LabelTensor> {
    pub fn add_with_labels(&mut self, input: &[f32], labels: &[i32]) {
        self.add_input(input);

        let max_seq_len = self.inputs.dims()[1] as usize;
        let token_embed_size = self.inputs.dims()[2] as usize;

        // Number of time steps to copy
        let timesteps = min(max_seq_len, input.len() / token_embed_size);

        let label_offset = self.sequence * max_seq_len;
        let label_seq = &mut self.labels.0[label_offset..label_offset + timesteps];
        label_seq.copy_from_slice(labels);

        self.sequence += 1;
    }
}

impl TensorBuilder<NoLabels> {
    pub fn add(&mut self, input: &[f32]) {
        self.add_input(input);
        self.sequence += 1;
    }
}
