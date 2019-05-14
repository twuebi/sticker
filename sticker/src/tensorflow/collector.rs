use ndarray::{Ix1, Ix2, Ix3};
use ndarray_tensorflow::NdTensor;

pub struct CollectedTensors {
    pub sequence_lens: Vec<NdTensor<i32, Ix1>>,
    pub inputs: Vec<NdTensor<f32, Ix3>>,
    pub labels: Vec<NdTensor<i32, Ix2>>,
}
