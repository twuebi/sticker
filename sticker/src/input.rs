use std::borrow::Cow;

use conllx::Sentence;
use failure::Error;
use ndarray::Array1;

pub enum Embeddings {
    FinalFrontier {
        model: finalfrontier::Model,
        unknown: Array1<f32>,
    },
    Word2Vec {
        embeds: rust2vec::Embeddings,
        unknown: Array1<f32>,
    },
}

impl Embeddings {
    pub fn dims(&self) -> usize {
        match self {
            Embeddings::FinalFrontier { model, .. } => model.config().dims as usize,
            Embeddings::Word2Vec { embeds, .. } => embeds.embed_len(),
        }
    }

    pub fn embedding(&self, word: &str) -> Cow<[f32]> {
        let embed = match self {
            Embeddings::FinalFrontier { model, unknown } => Cow::Owned(
                model
                    .embedding(word)
                    .unwrap_or(unknown.clone())
                    .into_raw_vec(),
            ),
            Embeddings::Word2Vec { embeds, unknown } => Cow::Borrowed(
                embeds
                    .embedding(word)
                    .unwrap_or(unknown.view())
                    .into_slice()
                    .expect("Non-contiguous word embedding"),
            ),
        };

        embed
    }
}

impl From<finalfrontier::Model> for Embeddings {
    fn from(model: finalfrontier::Model) -> Self {
        let mut unknown = Array1::zeros(model.config().dims as usize);

        for (_, embed) in &model {
            unknown += &embed;
        }

        let l2norm = unknown.dot(&unknown).sqrt();
        if l2norm != 0f32 {
            unknown /= l2norm;
        }

        Embeddings::FinalFrontier { model, unknown }
    }
}

impl From<rust2vec::Embeddings> for Embeddings {
    fn from(embeds: rust2vec::Embeddings) -> Self {
        let mut unknown = Array1::zeros(embeds.embed_len());

        for (_, embed) in &embeds {
            unknown += &embed;
        }

        let l2norm = unknown.dot(&unknown).sqrt();
        if l2norm != 0f32 {
            unknown /= l2norm;
        }

        Embeddings::Word2Vec { embeds, unknown }
    }
}

/// Sentence represented as a vector.
///
/// This data type represents a sentence as vectors (`Vec`) of tokens and
/// part-of-speech indices. Such a vector is typically the input to a
/// sequence labeling graph.
pub struct SentVec {
    pub tokens: Vec<f32>,
    pub subwords: Vec<Vec<f32>>,
}

impl SentVec {
    /// Construct a new sentence vector.
    pub fn new() -> Self {
        SentVec {
            tokens: Vec::new(),
            subwords: Vec::new(),
        }
    }

    /// Construct a sentence vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        SentVec {
            tokens: Vec::with_capacity(capacity),
            subwords: Vec::with_capacity(capacity),
        }
    }

    /// Get the embedding representation of a sentence.
    ///
    /// The vector contains the concatenation of the embeddings of the
    /// tokens in the sentence.
    pub fn into_inner(self) -> (Vec<Vec<f32>>, Vec<f32>) {
        (self.subwords, self.tokens)
    }
}

/// Embeddings for annotation layers.
///
/// This data structure bundles embedding matrices for the input
/// annotation layers: tokens and part-of-speech.
pub struct LayerEmbeddings {
    token_embeddings: Embeddings,
    char_embeddings: Embeddings,
}

impl LayerEmbeddings {
    /// Construct `LayerEmbeddings` from the given embeddings.
    pub fn new(token_embeddings: Embeddings, char_embeddings: Embeddings) -> Self {
        LayerEmbeddings {
            token_embeddings: token_embeddings,
            char_embeddings: char_embeddings,
        }
    }

    /// Get the character embedding matrix.
    pub fn char_embeddings(&self) -> &Embeddings {
        &self.char_embeddings
    }

    /// Get the token embedding matrix.
    pub fn token_embeddings(&self) -> &Embeddings {
        &self.token_embeddings
    }
}

/// Vectorizer for sentences.
///
/// An `SentVectorizer` vectorizes sentences.
pub struct SentVectorizer {
    layer_embeddings: LayerEmbeddings,
}

impl SentVectorizer {
    /// Construct an input vectorizer.
    ///
    /// The vectorizer is constructed from the embedding matrices. The layer
    /// embeddings are used to find the indices into the embedding matrix for
    /// layer values.
    pub fn new(layer_embeddings: LayerEmbeddings) -> Self {
        SentVectorizer {
            layer_embeddings: layer_embeddings,
        }
    }

    /// Get the layer embeddings.
    pub fn layer_embeddings(&self) -> &LayerEmbeddings {
        &self.layer_embeddings
    }

    /// Vectorize a sentence.
    pub fn realize(&self, sentence: &Sentence) -> Result<SentVec, Error> {
        let mut input = SentVec::with_capacity(sentence.len());

        for token in sentence {
            let form = token.form();

            input
                .tokens
                .extend_from_slice(&self.layer_embeddings.token_embeddings.embedding(form));

            let mut char_embeds = Vec::new();
            for ch in form.chars() {
                char_embeds.extend_from_slice(
                    &self
                        .layer_embeddings
                        .char_embeddings
                        .embedding(&ch.to_string()),
                );
            }

            input.subwords.push(char_embeds);
        }

        Ok(input)
    }
}
