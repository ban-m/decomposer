extern crate bio_utils;
extern crate poa_hmm;
extern crate rand;
extern crate rand_xoshiro;
extern crate rayon;
#[macro_use]
extern crate log;
pub mod clustering;
pub use clustering::clustering;
pub use clustering::predict;
pub use clustering::AlnParam;
pub use clustering::DEFAULT_ALN;

