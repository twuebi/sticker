use std::hash::Hash;

use tch::nn::LSTM;
use tch::{Device, Tensor};

use super::rnn::RNNModel;
use crate::{Numberer, SentVectorizer};

pub struct Tagger<T>
where
    T: Eq + Hash,
{
    model: RNNModel<LSTM>,
    labels: Numberer<T>,
    vectorizer: SentVectorizer,
}

impl<T> Tagger<T>
where
    T: Clone + Eq + Hash,
{
    pub fn random_weights(
        model: RNNModel<LSTM>,
        labels: Numberer<T>,
        vectorizer: SentVectorizer,
    ) -> Self {
        Tagger {
            model,
            labels,
            vectorizer,
        }
    }

    pub fn train(&self, device: Device, inputs: &Tensor, labels: &Tensor) -> f64 {
        self.model
            .train_batch(device, inputs, labels, self.labels.len() as i64)
    }
}
