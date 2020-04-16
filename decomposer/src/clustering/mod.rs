use bio_utils::fasta::Record;
use poa_hmm::*;
use rand::distributions::Standard;
use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
const BETA_INCREASE: f64 = 0.9;
const SMALL_WEIGHT: f64 = 0.000_000_000_0001;
const GIBBS_PRIOR: f64 = 0.00;
const STABLE_LIMIT: u32 = 8;
const IS_STABLE: u32 = 5;

pub struct AlnParam<F>
where
    F: Fn(u8, u8) -> i32,
{
    ins: i32,
    del: i32,
    score: F,
}

#[allow(dead_code)]
fn score2(x: u8, y: u8) -> i32 {
    if x == y {
        3
    } else {
        -4
    }
}

#[allow(dead_code)]
fn score(x: u8, y: u8) -> i32 {
    if x == y {
        2
    } else {
        -4
    }
}

pub const DEFAULT_ALN: AlnParam<fn(u8, u8) -> i32> = AlnParam {
    ins: -3,
    del: -3,
    score,
};

fn get_models<F, R>(
    data: &[Record],
    assignments: &[u8],
    sampled: &[bool],
    cluster_num: usize,
    rng: &mut R,
    param: (i32, i32, &F),
    pick: f64,
) -> Vec<POA>
where
    R: Rng,
    F: Fn(u8, u8) -> i32 + std::marker::Sync,
{
    let mut chunks: Vec<_> = vec![vec![]; cluster_num];
    let choises: Vec<u8> = (0..cluster_num).map(|e| e as u8).collect();
    for ((read, &asn), &b) in data.iter().zip(assignments.iter()).zip(sampled) {
        if !b {
            if let Ok(&chosen) =
                choises.choose_weighted(rng, |&k| if k == asn { 1. + pick } else { pick })
            {
                chunks[chosen as usize].push(read.seq());
            }
        }
    }
    let seeds: Vec<_> = rng.sample_iter(Standard).take(cluster_num).collect();
    chunks
        .par_iter()
        .zip(seeds.into_par_iter())
        .map(|(cluster, seed)| {
            POA::default().update(cluster, &vec![1.; cluster.len()], param, seed)
        })
        .collect()
}

fn logsumexp(xs: &[f64]) -> f64 {
    let max = xs.iter().max_by(|x, y| x.partial_cmp(&y).unwrap()).unwrap();
    let sum = xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    assert!(sum >= 0., "{:?}->{}", xs, sum);
    max + sum
}

fn update_assignments<R: Rng>(
    models: &[POA],
    assignments: &mut [u8],
    data: &[Record],
    sampled: &[bool],
    rng: &mut R,
    cluster_num: usize,
    config: &Config,
    beta: f64,
) -> u32 {
    let choises: Vec<_> = (0..cluster_num).map(|e| e as u8).collect();
    let fractions: Vec<f64> = (0..cluster_num)
        .map(|cl| assignments.iter().filter(|&&e| e == cl as u8).count())
        .map(|count| count as f64 / data.len() as f64 + SMALL_WEIGHT)
        .collect();
    let seeds: Vec<u64> = rng.sample_iter(Standard).take(data.len()).collect();
    assignments
        .iter_mut()
        .zip(data.iter())
        .zip(seeds.into_iter())
        .zip(sampled.iter())
        .filter(|&(_, &b)| b)
        .map(|(((asn, read), s), _)| {
            let likelihoods: Vec<_> = models
                .par_iter()
                .zip(fractions.par_iter())
                .map(|(m, &q)| m.forward(read.seq(), &config) * beta + q.ln())
                .collect();
            let total = logsumexp(&likelihoods);
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(s);
            let chosen = *choises
                .choose_weighted(&mut rng, |k| {
                    (likelihoods[*k as usize] - total).exp() + SMALL_WEIGHT
                })
                .unwrap();
            let is_the_same = if *asn == chosen { 0 } else { 1 };
            *asn = chosen;
            is_the_same
        })
        .sum::<u32>()
}

pub fn clustering<F>(
    data: &[Record],
    cluster_num: usize,
    limit: u64,
    config: &poa_hmm::Config,
    aln: &AlnParam<F>,
) -> Vec<Option<u8>>
where
    F: Fn(u8, u8) -> i32 + std::marker::Sync,
{
    if cluster_num <= 1 || data.len() <= 2 {
        return vec![None; data.len()];
    }
    let lim = limit / 3;
    let times = vec![0.005, 0.05, 0.10];
    let sum = data
        .iter()
        .flat_map(|e| e.id().bytes())
        .map(|e| e as u64)
        .sum::<u64>()
        + 1;
    let mut assignments = vec![];
    for (seed, pick_prob) in times.into_iter().enumerate() {
        let params = (lim, pick_prob, seed as u64 * sum);
        let (res, end) = gibbs_sampling_inner(data, cluster_num, params, config, aln);
        assignments = res;
        if end {
            break;
        }
    }
    assignments
}

fn print_lk_gibbs<F>(
    cluster_num: usize,
    asns: &[u8],
    data: &[Record],
    id: u64,
    name: &str,
    param: (i32, i32, &F),
    config: &Config,
) where
    F: Fn(u8, u8) -> i32 + std::marker::Sync,
{
    let falses = vec![false; data.len()];
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(id);
    let rng = &mut rng;
    let models = get_models(data, asns, &falses, cluster_num, rng, param, 0.);
    let fractions: Vec<f64> = (0..cluster_num)
        .map(|cl| asns.iter().filter(|&&e| e == cl as u8).count())
        .map(|count| count as f64 / data.len() as f64 + SMALL_WEIGHT)
        .collect();
    for (idx, read) in data.iter().enumerate() {
        let lks = calc_probs(&models, read, config, &fractions);
        let lks: Vec<_> = lks.into_iter().map(|l| format!("{}", l)).collect();
        trace!("FEATURE\t{}\t{}\t{}\t{}", name, id, idx, lks.join("\t"));
    }
}

fn gibbs_sampling_inner<F>(
    data: &[Record],
    cluster_num: usize,
    (limit, pick_prob, seed): (u64, f64, u64),
    config: &poa_hmm::Config,
    aln: &AlnParam<F>,
) -> (Vec<Option<u8>>, bool)
where
    F: Fn(u8, u8) -> i32 + std::marker::Sync,
{
    let param = (aln.ins, aln.del, &aln.score);
    let id: u64 = thread_rng().gen::<u64>() % 100_000;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
    let rng = &mut rng;
    let mut assignments: Vec<_> = (0..data.len())
        .map(|_| rng.gen_range(0, cluster_num) as u8)
        .collect();
    let mut beta = (cluster_num as f64).powi(2) / 2.;
    let mut count = 0;
    let mut predictions = std::collections::VecDeque::new();
    let asn = &mut assignments;
    let start = std::time::Instant::now();
    if log_enabled!(log::Level::Trace) {
        print_lk_gibbs(cluster_num, asn, &data, id, "B", param, config);
    }
    while count < STABLE_LIMIT {
        beta *= BETA_INCREASE;
        let changed_num = (0..pick_prob.recip().ceil() as usize / 2)
            .map(|_| {
                let s: Vec<bool> = (0..data.len()).map(|_| rng.gen_bool(pick_prob)).collect();
                let ms = get_models(&data, asn, &s, cluster_num, rng, param, GIBBS_PRIOR);
                update_assignments(&ms, asn, &data, &s, rng, cluster_num, config, beta)
            })
            .sum::<u32>();
        debug!("CHANGENUM\t{}", changed_num);
        if changed_num <= (data.len() as u32 / 50).max(2) {
            count += 1;
        } else {
            count = 0;
        }
        predictions.push_back(asn.clone());
        if predictions.len() as u32 > STABLE_LIMIT {
            predictions.pop_front();
        }
        report_gibbs(asn, count, id, cluster_num, pick_prob);
        if (std::time::Instant::now() - start).as_secs() > limit && count < STABLE_LIMIT / 2 {
            info!("Break by timelimit:{:?}", std::time::Instant::now() - start);
            let result = predictions_into_assignments(predictions, cluster_num, data.len());
            return (result, false);
        }
    }
    if log_enabled!(log::Level::Trace) {
        print_lk_gibbs(cluster_num, asn, &data, id, "A", param, config);
    }
    let result = predictions_into_assignments(predictions, cluster_num, data.len());
    (result, true)
}

fn predictions_into_assignments(
    predictions: std::collections::VecDeque<Vec<u8>>,
    cluster_num: usize,
    data: usize,
) -> Vec<Option<u8>> {
    let maximum_a_posterior = |xs: Vec<u8>| {
        let mut counts: Vec<u32> = vec![0; cluster_num];
        for x in xs {
            counts[x as usize] += 1;
        }
        let (cluster, count): (usize, u32) = counts
            .into_iter()
            .enumerate()
            .max_by_key(|e| e.1)
            .unwrap_or((0, 0));
        if count > IS_STABLE {
            Some(cluster as u8)
        } else {
            None
        }
    };
    predictions
        .into_iter()
        .fold(vec![vec![]; data], |mut acc, xs| {
            assert_eq!(acc.len(), xs.len());
            for (y, x) in acc.iter_mut().zip(xs) {
                y.push(x)
            }
            acc
        })
        .into_iter()
        .map(maximum_a_posterior)
        .collect()
}

fn report_gibbs(asn: &[u8], lp: u32, id: u64, cl: usize, beta: f64) {
    let line: Vec<_> = (0..cl)
        .map(|c| asn.iter().filter(|&&a| a == c as u8).count())
        .map(|e| format!("{}", e))
        .collect();
    info!("Summary\t{}\t{}\t{:.3}\t{}", id, lp, beta, line.join("\t"));
}

fn calc_probs(models: &[POA], read: &Record, c: &Config, fractions: &[f64]) -> Vec<f64> {
    let likelihoods: Vec<_> = models
        .par_iter()
        .zip(fractions.par_iter())
        .map(|(m, f)| m.forward(read.seq(), c) + f.ln())
        .collect();
    let t = logsumexp(&likelihoods);
    likelihoods.iter().map(|l| (l - t).exp()).collect()
}

pub fn predict<F>(
    data: &[Record],
    labels: &[Option<u8>],
    cluster_num: usize,
    c: &Config,
    input: &[Record],
    seed: u64,
    aln: &AlnParam<F>,
) -> Vec<Option<u8>>
where
    F: Fn(u8, u8) -> i32 + std::marker::Sync,
{
    let predictions: std::collections::VecDeque<_> = (0..STABLE_LIMIT)
        .map(|st| {
            let seed = seed + st as u64;
            predict_inner(data, labels, cluster_num, c, input, seed, aln)
        })
        .collect();
    predictions_into_assignments(predictions, cluster_num, input.len())
}

fn predict_inner<F>(
    data: &[Record],
    labels: &[Option<u8>],
    cluster_num: usize,
    c: &Config,
    input: &[Record],
    seed: u64,
    aln: &AlnParam<F>,
) -> Vec<u8>
where
    F: Fn(u8, u8) -> i32 + std::marker::Sync,
{
    let (data, labels): (Vec<Record>, Vec<_>) = data
        .into_iter()
        .zip(labels)
        .filter_map(|(d, l)| l.map(|l| (d.clone(), l)))
        .unzip();
    let param = (aln.ins, aln.del, &aln.score);
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
    let rng = &mut rng;
    let falses = vec![false; data.len()];
    let models = get_models(&data, &labels, &falses, cluster_num, rng, param, 0.);
    let choises: Vec<_> = (0..cluster_num).map(|e| e as u8).collect();
    let fractions: Vec<f64> = (0..cluster_num)
        .map(|cl| labels.iter().filter(|&&e| e == cl as u8).count())
        .map(|count| count as f64 / data.len() as f64 + SMALL_WEIGHT)
        .collect();
    input
        .iter()
        .filter_map(|read| {
            let probs = calc_probs(&models, read, c, &fractions);
            choises.choose_weighted(rng, |&k| probs[k as usize]).ok()
        })
        .copied()
        .collect()
}
