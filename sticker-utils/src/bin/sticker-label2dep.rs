use std::io::BufWriter;

use clap::{App, AppSettings, Arg};

use conllx::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, OrExit, Output};

use conllx::graph::Node;
use sticker::depparse::{
    DependencyEncoding, RelativePOS, RelativePOSEncoder,
};
use sticker::EncodingProb;
use sticker::{Numberer, SentenceDecoder};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];
//static CONFIG: &str = "CONFIG";
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

fn main() {
    let matches = App::new("sticker-label2dep").settings(DEFAULT_CLAP_SETTINGS)
        .arg(Arg::with_name(INPUT).help("Input data").index(2))
        .arg(Arg::with_name(OUTPUT).help("Output data").index(3))
        .get_matches();

//    let configs: String = matches.value_of(CONFIG).unwrap().into();
    let input = matches.value_of(INPUT).map(ToOwned::to_owned);
    let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);

//    let config_file =
//        File::open(&configs).or_exit(format!("Cannot open configuration file '{}'", &configs), 1);
//    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
//    config
//        .relativize_paths(configs)
//        .or_exit("Cannot relativize paths in configuration", 1);

    let input = Input::from(input);
    let reader = Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

    let output = Output::from(output);
    let writer = Writer::new(BufWriter::new(
        output.write().or_exit("Cannot open output for writing", 1),
    ));
    process_with_decoder(RelativePOSEncoder, reader, writer);
}

fn process_with_decoder<R, W>(decoder:  RelativePOSEncoder, read: R, mut write: W)
where
    R: ReadSentence,
    W: WriteSentence,
{
//    let labels = config.labeler.load_labels().or_exit(
//        format!("Cannot load label file '{}'", config.labeler.labels),
//        1,
//    );

//    let categorical_decoder : CategoricalEncoder<_,_> = CategoricalEncoder::new(decoder, labels.clone());

    for sentence in read.sentences() {
        let mut sentence = sentence.or_exit("Cannot parse sentence", 1);
        let labels = sentence
            .iter().by_ref()
            .filter(|t| t.is_token())
            .map(|t| {
                let token = &t.token().unwrap();
                let feat_map = token
                    .features()
                    .unwrap()
                    .as_map();
                let entry = feat_map
                    .get("deplabel").cloned()
                    .unwrap()
                    .unwrap();
                let mut cans = entry.split(";");
                 
                let mut v : Vec<EncodingProb<sticker::depparse::DependencyEncoding<RelativePOS>>> = cans.into_iter().map(|c|  {
                    eprintln!("{:?}",c);
                    let mut pieces = c.split(",");
                    let prob = pieces.next().unwrap().parse().unwrap();
                    let mut enc = pieces.next().unwrap().split("/");

                    let rel = enc.next().unwrap();
                    let pos = enc.next().unwrap();
                    let dist = enc.next();
                    let rel_pos = RelativePOS::new(pos, dist.unwrap().parse().unwrap());

                    let e = EncodingProb::new_from_owned(DependencyEncoding::new(
                        rel_pos,
                        rel.to_string(),
                    ),prob);

                    e
                }).collect::<Vec<_>>();
                v.sort_by(|a, b| a.prob().partial_cmp(&b.prob()).unwrap());
                v.reverse();
                v
            })
            .collect::<Vec<Vec<_>>>();

        decoder.decode(&labels, &mut sentence);
        write.write_sentence(&sentence);
    }
}
