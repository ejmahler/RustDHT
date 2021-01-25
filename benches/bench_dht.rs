#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate test;

use std::sync::Arc;

use test::Bencher;

use rustdht::{Dht, scalar::MixedRadix};
use rustfft::{FftNum, Length};

struct Noop {
    len: usize,
}
impl<T: FftNum> Dht<T> for Noop {
    fn process_with_scratch(&self, _buffer: &mut [T], _scratch: &mut [T]) {}
    fn process_outofplace_with_scratch(&self, _input: &mut [T], _output: &mut [T], _scratch: &mut [T]) {}
    fn get_inplace_scratch_len(&self) -> usize { self.len }
    fn get_outofplace_scratch_len(&self) -> usize { 0 }
}
impl Length for Noop {
    fn len(&self) -> usize { self.len }
}

fn bench_mixed_radix_noop(b: &mut Bencher, width: usize, height: usize) {

    let width_dht = Arc::new(Noop { len: width });
    let height_dht = Arc::new(Noop { len: height });

    let dht : Arc<Dht<_>> = Arc::new(MixedRadix::new(width_dht, height_dht));

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| {dht.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn mixed_radix_noop_0002_3(b: &mut Bencher) { bench_mixed_radix_noop(b,  2, 3); }
#[bench] fn mixed_radix_noop_0003_4(b: &mut Bencher) { bench_mixed_radix_noop(b,  3, 4); }
#[bench] fn mixed_radix_noop_0004_5(b: &mut Bencher) { bench_mixed_radix_noop(b,  4, 5); }
#[bench] fn mixed_radix_noop_0007_32(b: &mut Bencher) { bench_mixed_radix_noop(b, 7, 32); }
#[bench] fn mixed_radix_noop_0032_27(b: &mut Bencher) { bench_mixed_radix_noop(b,  32, 27); }
#[bench] fn mixed_radix_noop_0256_243(b: &mut Bencher) { bench_mixed_radix_noop(b,  256, 243); }
#[bench] fn mixed_radix_noop_2048_0003(b: &mut Bencher) { bench_mixed_radix_noop(b,  2048, 3); }
#[bench] fn mixed_radix_noop_0003_2048(b: &mut Bencher) { bench_mixed_radix_noop(b,  3, 2048); }
#[bench] fn mixed_radix_noop_2048_2048(b: &mut Bencher) { bench_mixed_radix_noop(b,  2048, 2048); }
#[bench] fn mixed_radix_noop_2048_2187(b: &mut Bencher) { bench_mixed_radix_noop(b,  2048, 2187); }
#[bench] fn mixed_radix_noop_2187_2187(b: &mut Bencher) { bench_mixed_radix_noop(b,  2187, 2187); }