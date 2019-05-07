use failure::Error;
use tch::nn;
use tch::nn::{Module, ModuleT, OptimizerConfig};
use tch::{Device, Kind, Tensor};

pub struct RNNModel<R> {
    batch_norm: nn::BatchNorm,
    rnn: R,
    linear: nn::Linear,
    optimizer: nn::Optimizer<nn::Adam>,
}

impl RNNModel<nn::LSTM> {
    pub fn new_lstm(
        vs: &nn::VarStore,
        input_size: i64,
        hidden_size: i64,
        lstm_layers: i64,
        n_labels: i64,
        dropout: f64,
    ) -> Result<Self, Error> {
        // Bidirectional RNNs are not properly supported?
        // https://github.com/pytorch/pytorch/issues/17998
        let stacked_rnn_config = nn::RNNConfig {
            has_biases: true,
            num_layers: 1,
            bidirectional: false,
            dropout,
            train: false,
            batch_first: true,
        };

        let lstm = nn::lstm(&vs.root(), input_size, hidden_size, stacked_rnn_config);

        Self::new_from_rnn(vs, lstm, input_size, hidden_size, n_labels)
    }
}

impl RNNModel<nn::GRU> {
    pub fn new_gru(
        vs: &nn::VarStore,
        input_size: i64,
        hidden_size: i64,
        lstm_layers: i64,
        n_labels: i64,
        dropout: f64,
    ) -> Result<Self, Error> {
        let stacked_rnn_config = nn::RNNConfig {
            has_biases: true,
            num_layers: lstm_layers,
            bidirectional: true,
            dropout,
            train: false,
            batch_first: true,
        };

        let gru = nn::gru(&vs.root(), input_size, hidden_size, stacked_rnn_config);

        Self::new_from_rnn(vs, gru, input_size, hidden_size, n_labels)
    }
}

impl<R> RNNModel<R>
where
    R: nn::RNN,
{
    pub fn new_from_rnn(
        vs: &nn::VarStore,
        rnn: R,
        input_size: i64,
        hidden_size: i64,
        n_labels: i64,
    ) -> Result<Self, Error> {
        let batch_norm = nn::batch_norm1d(&vs.root(), hidden_size, Default::default());
        let linear = nn::linear(&vs.root(), hidden_size, n_labels, Default::default());
        let optimizer = nn::Adam::default().build(vs, 0.05)?;

        Ok(RNNModel {
            batch_norm,
            rnn,
            linear,
            optimizer,
        })
    }

    pub fn train_batch(
        &self,
        device: Device,
        inputs: &Tensor,
        labels: &Tensor,
        n_labels: i64,
    ) -> f64 {
        // Get shape.
        let (batch_size, seq_len) = labels.size2().unwrap();

        // How to handle padding?
        let (hidden, _) = self.rnn.seq(&inputs.to_device(device));
        let norm_hidden = self.batch_norm.forward_t(&hidden.transpose(1, 2), true);
        let logits = self.linear.forward(&norm_hidden.transpose(1, 2));
        let loss = logits
            .view(&[batch_size * seq_len, n_labels])
            .cross_entropy_for_logits(
                &labels
                    .totype(Kind::Float)
                    .to_device(device)
                    .view(&[batch_size, seq_len]),
            );
        self.optimizer.backward_step_clip(&loss, 0.5);

        loss.into()
    }
}
