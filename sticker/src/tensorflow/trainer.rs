use std::path::Path;

use failure::{Error, Fallible};
use ndarray::{Ix1, Ix2, Ix3};
use ndarray_tensorflow::NdTensor;
use tensorflow::{Session, SessionOptions, SessionRunArgs, Tensor};

use super::tagger::TaggerGraph;
use super::util::{prepare_path, status_to_error};

use crate::ModelPerformance;
use protobuf::Message;

use tf_proto::{Event, RunMetadata, RunOptions, RunOptions_TraceLevel, TaggedRunMetadata};
/// Trainer for a sequence labeling model.
pub struct TaggerTrainer {
    graph: TaggerGraph,
    session: Session,
    summaries: bool,
}

impl TaggerTrainer {
    /// Create a new trainer with loaded weights.
    ///
    /// This constructor will load the model parameters (such as weights) from
    /// the file specified in `parameters_path`.
    pub fn load_weights<P>(graph: TaggerGraph, parameters_path: P) -> Fallible<Self>
    where
        P: AsRef<Path>,
    {
        // Restore parameters.
        let path_tensor = prepare_path(parameters_path)?.into();
        let mut args = SessionRunArgs::new();
        args.add_feed(&graph.save_path_op, 0, &path_tensor);
        args.add_target(&graph.restore_op);
        let session = Self::new_session(&graph)?;
        session.run(&mut args).map_err(status_to_error)?;

        Ok(TaggerTrainer {
            graph,
            session,
            summaries: false,
        })
    }

    /// Create a new session with randomized weights.
    pub fn random_weights(graph: TaggerGraph) -> Result<Self, Error> {
        // Initialize parameters.
        let mut args = SessionRunArgs::new();
        args.add_target(&graph.init_op);
        let session = Self::new_session(&graph)?;
        session
            .run(&mut args)
            .expect("Cannot initialize parameters");

        Ok(TaggerTrainer {
            graph,
            session,
            summaries: false,
        })
    }

    fn new_session(graph: &TaggerGraph) -> Result<Session, Error> {
        let mut session_opts = SessionOptions::new();
        session_opts
            .set_config(&graph.model_config.to_protobuf()?)
            .map_err(status_to_error)?;

        Session::new(&session_opts, &graph.graph).map_err(status_to_error)
    }

    /// Save the model parameters.
    ///
    /// The model parameters are stored as the given path.
    pub fn save<P>(&self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        // Add leading directory component if absent.
        let path_tensor = prepare_path(path)?.into();

        // Call the save op.
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.save_path_op, 0, &path_tensor);
        args.add_target(&self.graph.save_op);
        self.session.run(&mut args).map_err(status_to_error)
    }

    pub fn init_logdir(&mut self, path: &str) -> Result<(), Error> {
        let path_tensor = path.to_string().into();
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.logdir_op, 0, &path_tensor);
        args.add_target(&self.graph.summary_init_op);
        self.session.run(&mut args).map_err(status_to_error)?;
        let mut args = SessionRunArgs::new();
        args.add_target(&self.graph.graph_write_op);
        self.summaries = true;
        self.session.run(&mut args).map_err(status_to_error)
    }

    /// Train on a batch of inputs and labels.
    pub fn train(
        &self,
        seq_lens: &NdTensor<i32, Ix1>,
        inputs: &NdTensor<f32, Ix3>,
        labels: &NdTensor<i32, Ix2>,
        learning_rate: f32,
    ) -> ModelPerformance {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = true;

        let mut lr = Tensor::new(&[]);
        lr[0] = learning_rate;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.is_training_op, 0, &is_training);
        args.add_feed(&self.graph.lr_op, 0, &lr);
        args.add_target(&self.graph.train_op);
        if self.summaries {
            args.add_target(&self.graph.train_summary_op);
        }
        self.validate_(seq_lens, inputs, labels, args, Mode::TRAIN)
    }

    /// Perform validation using a batch of inputs and labels.
    pub fn validate(
        &self,
        seq_lens: &NdTensor<i32, Ix1>,
        inputs: &NdTensor<f32, Ix3>,
        labels: &NdTensor<i32, Ix2>,
    ) -> ModelPerformance {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = false;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.is_training_op, 0, &is_training);
        if self.summaries {
            args.add_target(&self.graph.val_summary_op);
        }
        self.validate_(seq_lens, inputs, labels, args, Mode::VALIDATE)
    }

    fn validate_<'l>(
        &'l self,
        seq_lens: &'l NdTensor<i32, Ix1>,
        inputs: &'l NdTensor<f32, Ix3>,
        labels: &'l NdTensor<i32, Ix2>,
        mut args: SessionRunArgs<'l>,
        mode: Mode,
    ) -> ModelPerformance {
        // Add inputs.
        args.add_feed(&self.graph.inputs_op, 0, inputs.inner_ref());
        args.add_feed(&self.graph.seq_lens_op, 0, seq_lens.inner_ref());

        // Add gold labels.
        args.add_feed(&self.graph.labels_op, 0, labels.inner_ref());

        let accuracy_token = args.request_fetch(&self.graph.accuracy_op, 0);
        let loss_token = args.request_fetch(&self.graph.loss_op, 0);

        let mut ro = RunOptions::new();
        ro.set_trace_level(RunOptions_TraceLevel::FULL_TRACE);

        let mut ro_bytes = Vec::new();
        ro.write_to_vec(&mut ro_bytes)
            .expect("Failed serializing RunOptions proto!");
        let train_step_token = args.request_fetch(&self.graph.train_step_op, 0);
        let val_step_token = args.request_fetch(&self.graph.val_step_op, 0);

        let metadata = self
            .session
            .run_with_metadata(&mut args, &ro_bytes[..])
            .expect("Failed running graph");

        let mut rm = RunMetadata::new();
        rm.merge_from(&mut protobuf::CodedInputStream::from_bytes(&metadata))
            .expect("Retrieving metadata failed!");
        let mut trm = TaggedRunMetadata::new();
        trm.set_run_metadata(metadata);

        match mode {
            Mode::TRAIN => {
                let train_step: i64 = args
                    .fetch(train_step_token)
                    .expect("Unable to retrieve train step!")[0];
                trm.set_tag(format!("train_{}", train_step));
            }
            Mode::VALIDATE => {
                let val_step: i64 = args
                    .fetch(val_step_token)
                    .expect("Unable to retrieve val step!")[0];
                trm.set_tag(format!("val_{}", val_step));
            }
        }

        let mut event = Event::new();
        event.set_tagged_run_metadata(trm);

        let mp = ModelPerformance {
            loss: args.fetch(loss_token).expect("Unable to retrieve loss")[0],
            accuracy: args
                .fetch(accuracy_token)
                .expect("Unable to retrieve accuracy")[0],
        };

        let mut met = Vec::new();
        event
            .write_to_vec(&mut met)
            .expect("Serializing metadata failed!");
        unsafe {
            let mut input: Tensor<String> = Tensor::new(&[]);
            input[0] = String::from_utf8_unchecked(met);

            {
                let mut args = SessionRunArgs::new();

                args.add_feed(&self.graph.metadata_input_op, 0, &input);
                args.add_target(&self.graph.metadata_write_op);

                self.session.run(&mut args).expect("fail");
            }
        }
        mp
    }
}

pub enum Mode {
    TRAIN,
    VALIDATE,
}
