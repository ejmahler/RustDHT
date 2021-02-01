#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate test;

use std::sync::Arc;

use test::Bencher;

use rustdht::{Dht, scalar::{Butterfly16, Butterfly2, Butterfly4, Butterfly8, MixedRadix, MixedRadix4xn}};
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

#[bench] fn mixed_radix_generic4x_noop_0003(b: &mut Bencher) { bench_mixed_radix_noop(b, 4, 3); }
#[bench] fn mixed_radix_generic4x_noop_0004(b: &mut Bencher) { bench_mixed_radix_noop(b, 4, 4); }
#[bench] fn mixed_radix_generic4x_noop_0005(b: &mut Bencher) { bench_mixed_radix_noop(b, 4, 5); }
#[bench] fn mixed_radix_generic4x_noop_0027(b: &mut Bencher) { bench_mixed_radix_noop(b, 4, 27); }
#[bench] fn mixed_radix_generic4x_noop_0032(b: &mut Bencher) { bench_mixed_radix_noop(b, 4, 32); }
#[bench] fn mixed_radix_generic4x_noop_0243(b: &mut Bencher) { bench_mixed_radix_noop(b, 4, 243); }
#[bench] fn mixed_radix_generic4x_noop_2048(b: &mut Bencher) { bench_mixed_radix_noop(b, 4, 2048); }
#[bench] fn mixed_radix_generic4x_noop_2187(b: &mut Bencher) { bench_mixed_radix_noop(b, 4, 2187); }


fn bench_4xn_noop(b: &mut Bencher, height: usize) {

    let height_dht = Arc::new(Noop { len: height });

    let dht : Arc<Dht<_>> = Arc::new(MixedRadix4xn::new(height_dht));

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| {dht.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn bench_4xn_noop_4x_0003(b: &mut Bencher) { bench_4xn_noop(b, 3); }
#[bench] fn bench_4xn_noop_4x_0004(b: &mut Bencher) { bench_4xn_noop(b, 4); }
#[bench] fn bench_4xn_noop_4x_0005(b: &mut Bencher) { bench_4xn_noop(b, 5); }
#[bench] fn bench_4xn_noop_4x_0027(b: &mut Bencher) { bench_4xn_noop(b, 27); }
#[bench] fn bench_4xn_noop_4x_0032(b: &mut Bencher) { bench_4xn_noop(b, 32); }
#[bench] fn bench_4xn_noop_4x_0243(b: &mut Bencher) { bench_4xn_noop(b, 243); }
#[bench] fn bench_4xn_noop_4x_2048(b: &mut Bencher) { bench_4xn_noop(b, 2048); }
#[bench] fn bench_4xn_noop_4x_2187(b: &mut Bencher) { bench_4xn_noop(b, 2187); }


fn make_4xn_radix4(len: usize) -> Arc<dyn Dht<f32>> {
    match len {
        0|1 => panic!(),
        2 => Arc::new(Butterfly2::new()),
        4 => Arc::new(Butterfly4::new()),
        8 => Arc::new(Butterfly8::new()),
        16 => Arc::new(Butterfly16::new()),
        _ => Arc::new(MixedRadix4xn::new(make_4xn_radix4(len/4))),
    }
}

fn bench_4xn_radix4(b: &mut Bencher, len: usize) {

    let dht = make_4xn_radix4(len);

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| {dht.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn bench_4xn_radix4_00016(b: &mut Bencher) { bench_4xn_radix4(b, 16); }
#[bench] fn bench_4xn_radix4_00032(b: &mut Bencher) { bench_4xn_radix4(b, 32); }
#[bench] fn bench_4xn_radix4_00064(b: &mut Bencher) { bench_4xn_radix4(b, 64); }
#[bench] fn bench_4xn_radix4_00128(b: &mut Bencher) { bench_4xn_radix4(b, 128); }
#[bench] fn bench_4xn_radix4_00256(b: &mut Bencher) { bench_4xn_radix4(b, 256); }
#[bench] fn bench_4xn_radix4_00512(b: &mut Bencher) { bench_4xn_radix4(b, 512); }
#[bench] fn bench_4xn_radix4_01024(b: &mut Bencher) { bench_4xn_radix4(b, 1024); }
#[bench] fn bench_4xn_radix4_02048(b: &mut Bencher) { bench_4xn_radix4(b, 2048); }
#[bench] fn bench_4xn_radix4_16384(b: &mut Bencher) { bench_4xn_radix4(b, 16384); }
#[bench] fn bench_4xn_radix4_32768(b: &mut Bencher) { bench_4xn_radix4(b, 32768); }
#[bench] fn bench_4xn_radix4_65536(b: &mut Bencher) { bench_4xn_radix4(b, 65536); }