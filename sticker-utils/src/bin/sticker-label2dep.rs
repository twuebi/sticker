use std::fs::File;
use std::hash::Hash;
use std::io::BufWriter;

use clap::Arg;
use conllx::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, OrExit, Output};

use conllx::graph::Node;
use sticker::depparse::{RelativePOSEncoder, RelativePositionEncoder, RelativePOS, DependencyEncoding};
use sticker::tensorflow::{Tagger, TaggerGraph};
use sticker::EncodingProb;
use sticker::{CategoricalEncoder, LayerEncoder, Numberer, SentVectorizer, SentenceDecoder};
use sticker_utils::{
    sticker_app, CborRead, Config, EncoderType, LabelerType, SentProcessor, TaggerSpeed, TomlRead,
};

static CONFIG: &str = "CONFIG";
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

fn main() {
    let matches = sticker_app("sticker-tag")
        .arg(Arg::with_name(INPUT).help("Input data").index(2))
        .arg(Arg::with_name(OUTPUT).help("Output data").index(3))
        .get_matches();

    let configs: String = matches.value_of(CONFIG).unwrap().into();
    let input = matches.value_of(INPUT).map(ToOwned::to_owned);
    let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);

    let config_file =
        File::open(&configs).or_exit(format!("Cannot open configuration file '{}'", &configs), 1);
    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(configs)
        .or_exit("Cannot relativize paths in configuration", 1);

    let input = Input::from(input);
    let reader = Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

    let output = Output::from(output);
    let writer = Writer::new(BufWriter::new(
        output.write().or_exit("Cannot open output for writing", 1),
    ));

    match config.labeler.labeler_type {
        LabelerType::Sequence(ref layer) => {
            process_with_decoder(&config, LayerEncoder::new(layer.clone()), reader, writer)
        }
        LabelerType::Parser(EncoderType::RelativePOS) => {
            process_with_decoder(&config, RelativePOSEncoder, reader, writer)
        }
        LabelerType::Parser(EncoderType::RelativePosition) => {
            process_with_decoder(&config, RelativePositionEncoder, reader, writer)
        }
    };
}

fn process_with_decoder<D, R, W>(config: &Config, decoder: D, read: R, mut write: W)
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
    R: ReadSentence,
    W: WriteSentence,
{
    let labels  = config.labeler.load_labels().or_exit(
        format!("Cannot load label file '{}'", config.labeler.labels),
        1,
    );

    let categorical_decoder = CategoricalEncoder::new(decoder, labels);

    for sentence in read.sentences() {
        let mut sentence = sentence.or_exit("Cannot parse sentence", 1);
        let labels : Vec<Vec<EncodingProb<_>>> = sentence
            .iter()
            .filter(|&t |t.is_token())
            .map(|&t| {
                let enc = &t.token()
                    .unwrap()
                    .features()
                    .unwrap()
                    .as_map()
                    .get("deplabel")
                    .cloned()
                    .unwrap()
                    .unwrap().split("/");
                let rel = enc.next().unwrap();
                let pos = enc.next().unwrap();
                let dist = enc.next();
                let rel_pos = RelativePOS::new(pos, dist.unwrap().parse().unwrap());
                let denc : DependencyEncoding<RelativePOS> = DependencyEncoding {
                    head : rel_pos,
                    label : rel.to_string()
                };
                vec!(EncodingProb::new(&labels.number(&denc).unwrap(),1.0,))
            })
            .collect::<Vec<Vec<_>>>();
        decoder.decode(&labels, &mut sentence);
        write.write_sentence(&sentence);
    }
}
