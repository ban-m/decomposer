extern crate bio_utils;
extern crate clap;
extern crate poa_hmm;
extern crate rand;
extern crate rand_distr;
extern crate rand_xoshiro;
use bio_utils::fasta;
use clap::{App, Arg};
use poa_hmm::gen_sample::*;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoroshiro128Plus;
use std::fs::File;
use std::io::BufWriter;
fn command() -> App<'static, 'static> {
    App::new("test dataset")
        .version("0.1")
        .author("Bansho Masutani")
        .about("Softwares to create mock short tandem repeat.")
        .arg(
            Arg::with_name("unit_length")
                .default_value("6")
                .short("u")
                .long("unit_length")
                .value_name("UNIT LENGTH")
                .help("length of unit.")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("repeat_num")
                .short("r")
                .default_value("500")
                .long("repeat")
                .value_name("REPEAT")
                .help("Repeat number of STR.")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("flanking_length")
                .default_value("1000")
                .short("f")
                .long("flanking_length")
                .value_name("LEN")
                .help("length of flanking region")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("mean_length")
                .default_value("3000")
                .short("m")
                .long("mean_length")
                .value_name("LEN")
                .help("mean length of reads")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("sd_length")
                .default_value("500")
                .short("s")
                .long("sd_length")
                .value_name("SD")
                .help("standard deviation of the length of reads")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("cluster_num")
                .default_value("2")
                .short("c")
                .long("cluster_num")
                .value_name("CLUSTER_NUM")
                .help("Number of cluster.")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("divergence")
                .short("d")
                .default_value("0.01")
                .long("div_rate")
                .value_name("DIVERGENCE")
                .help("Divergence rate between each haplotype(%).")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("num")
                .short("n")
                .long("num")
                .default_value("100")
                .value_name("NUM")
                .help("Generate number")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("seed")
                .long("seed")
                .default_value("24")
                .value_name("SEED")
                .help("seed")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("outdir")
                .short("o")
                .long("outdir")
                .value_name("OUTPUT DIRECTORY")
                .help("output directory")
                .required(true)
                .takes_value(true),
        )
}

fn parse<T: std::str::FromStr>(matches: &clap::ArgMatches, arg: &str) -> Option<T> {
    matches.value_of(arg).and_then(|e| e.parse::<T>().ok())
}

fn create_dataset(matches: &clap::ArgMatches) -> Option<()> {
    let unitlen = parse::<usize>(matches, "unit_length")?;
    let repnum = parse::<usize>(matches, "repeat_num")?;
    let flanking = parse::<usize>(matches, "flanking_length")?;
    let sd_length = parse::<f64>(matches, "sd_length")?;
    let mean_len = parse::<f64>(matches, "mean_length")?;
    let cluster_num = parse::<usize>(matches, "cluster_num")?;
    let div = parse::<f64>(matches, "divergence")?;
    let num = parse::<usize>(matches, "num")?;
    let seed = parse::<u64>(matches, "seed")?;
    let templates = generate_template(unitlen, repnum, flanking, cluster_num, div, seed);
    let mut rng: Xoroshiro128Plus = SeedableRng::seed_from_u64(seed);
    let len_distr = Normal::new(mean_len, sd_length).unwrap();
    let outdir = matches.value_of("outdir").unwrap();
    std::fs::create_dir_all(outdir).unwrap();
    let mut wtr = File::create(format!("{}/templates.fa", outdir))
        .map(BufWriter::new)
        .map(fasta::Writer::new)
        .unwrap();
    for template in templates.iter() {
        wtr.write_record(template).unwrap();
    }
    let mut wtr = File::create(format!("{}/reads.fa", outdir))
        .map(BufWriter::new)
        .map(fasta::Writer::new)
        .unwrap();
    for idx in 0..num {
        let template = templates.choose(&mut rng).unwrap();
        let max_len = template.seq().len();
        let length = (len_distr.sample(&mut rng).floor() as usize).min(max_len - 1);
        let start = rng.gen_range(0, max_len - length);
        let seq = &template.seq()[start..start + length];
        let seq = introduce_randomness(seq, &mut rng, &PROFILE);
        let desc = Some(template.id().to_string());
        let record = fasta::Record::with_data(&format!("{}", idx), &desc, &seq);
        wtr.write_record(&record).unwrap();
    }
    Some(())
}

fn generate_template(
    unitlen: usize,
    repnum: usize,
    flanking: usize,
    cluster_num: usize,
    div: f64,
    seed: u64,
) -> Vec<fasta::Record> {
    let mut rng: Xoroshiro128Plus = SeedableRng::seed_from_u64(seed * 11);
    let unit = generate_seq(&mut rng, unitlen);
    let flanking_1 = generate_seq(&mut rng, flanking);
    let flanking_2 = generate_seq(&mut rng, flanking);
    let core_repeat: Vec<_> = std::iter::repeat(unit)
        .take(repnum)
        .flat_map(|e| e)
        .collect();
    let template = vec![flanking_1, core_repeat, flanking_2].concat();
    let p = Profile {
        sub: div / 3.,
        ins: div / 3.,
        del: div / 3.,
    };
    (0..cluster_num)
        .map(|_| introduce_randomness(&template, &mut rng, &p))
        .enumerate()
        .map(|(id, seq)| fasta::Record::with_data(&format!("{}", id), &None, &seq))
        .collect()
}

fn main() {
    let app = command();
    let matches = app.get_matches();
    create_dataset(&matches).unwrap()
}
