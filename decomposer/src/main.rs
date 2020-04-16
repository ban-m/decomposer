extern crate bio_utils;
extern crate clap;
#[macro_use]
extern crate log;
extern crate decomposer;
extern crate env_logger;
use clap::{App, Arg, SubCommand};
use decomposer::*;
fn subcommand_decompose() -> App<'static, 'static> {
    SubCommand::with_name("decompose")
        .version("0.1")
        .about("To Decompose long reads")
        .arg(
            Arg::with_name("reads")
                .required(true)
                .short("r")
                .long("reads")
                .value_name("READS")
                .help("Raw long reads<FASTA>")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("cluster_num")
                .short("c")
                .long("cluster_num")
                .required(false)
                .value_name("CLUSTER_NUM")
                .help("Minimum cluster number.")
                .default_value(&"2")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("threads")
                .short("t")
                .long("threads")
                .required(false)
                .value_name("THREADS")
                .help("Number of Threads")
                .default_value(&"1")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .multiple(true)
                .help("Output debug to the standard error."),
        )
        .arg(
            Arg::with_name("limit")
                .short("l")
                .long("limit")
                .required(false)
                .value_name("LIMIT")
                .help("Maximum Execution time(sec)")
                .default_value(&"7200")
                .takes_value(true),
        )
}

fn decompose(matches: &clap::ArgMatches) -> std::io::Result<()> {
    let level = match matches.occurrences_of("verbose") {
        0 => "warn",
        1 => "info",
        2 => "debug",
        3 | _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level)).init();
    debug!("MMMM started. Debug mode.");
    let reads = matches
        .value_of("reads")
        .map(|file| match bio_utils::fasta::parse_into_vec(file) {
            Ok(res) => res,
            Err(why) => panic!("{}:{}", why, file),
        })
        .unwrap();
    let cluster_num: usize = matches
        .value_of("cluster_num")
        .and_then(|num| num.parse().ok())
        .unwrap();
    let threads: usize = matches
        .value_of("threads")
        .and_then(|num| num.parse().ok())
        .unwrap();
    let limit: u64 = matches
        .value_of("limit")
        .and_then(|num| num.parse().ok())
        .unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let config = &poa_hmm::DEFAULT_CONFIG;
    let results = clustering(&reads, cluster_num, limit, config, &DEFAULT_ALN);
    assert_eq!(results.len(), reads.len());
    for (read, pred) in reads
        .iter()
        .zip(results)
        .filter_map(|(r, p)| p.map(|p| (r, p)))
    {
        let desc = match read.desc() {
            Some(res) => res,
            None => "",
        };
        println!("{}\t{}\t{}", pred, read.id(), desc);
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let matches = App::new("MMMM")
        .version("0.1")
        .author("Bansho Masutani")
        .about("Softwares to Decompose long reads.")
        .subcommand(subcommand_decompose())
        .get_matches();
    match matches.subcommand() {
        ("decompose", Some(sub_m)) => decompose(sub_m),
        _ => Ok(()),
    }
}
