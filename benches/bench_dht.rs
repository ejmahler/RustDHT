#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate test;

use std::sync::Arc;

use test::Bencher;

use rustdht::{Dht, scalar::{Butterfly1, Butterfly16, Butterfly2, Butterfly3, Butterfly4, Butterfly5, Butterfly6, Butterfly8, Butterfly9, MixedRadix, MixedRadix3xn, MixedRadix4xn, MixedRadix5xn, MixedRadix6xn, RadersAlgorithm, Radix4, SplitRadix}};
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

#[bench] fn bench_4xn_radix4_0000016(b: &mut Bencher) { bench_4xn_radix4(b, 16); }
#[bench] fn bench_4xn_radix4_0000032(b: &mut Bencher) { bench_4xn_radix4(b, 32); }
#[bench] fn bench_4xn_radix4_0000064(b: &mut Bencher) { bench_4xn_radix4(b, 64); }
#[bench] fn bench_4xn_radix4_0000128(b: &mut Bencher) { bench_4xn_radix4(b, 128); }
#[bench] fn bench_4xn_radix4_0000256(b: &mut Bencher) { bench_4xn_radix4(b, 256); }
#[bench] fn bench_4xn_radix4_0000512(b: &mut Bencher) { bench_4xn_radix4(b, 512); }
#[bench] fn bench_4xn_radix4_0001024(b: &mut Bencher) { bench_4xn_radix4(b, 1024); }
#[bench] fn bench_4xn_radix4_0002048(b: &mut Bencher) { bench_4xn_radix4(b, 2048); }
#[bench] fn bench_4xn_radix4_0016384(b: &mut Bencher) { bench_4xn_radix4(b, 16384); }
#[bench] fn bench_4xn_radix4_0032768(b: &mut Bencher) { bench_4xn_radix4(b, 32768); }
#[bench] fn bench_4xn_radix4_0065536(b: &mut Bencher) { bench_4xn_radix4(b, 65536); }
#[bench] fn bench_4xn_radix4_1048576(b: &mut Bencher) { bench_4xn_radix4(b, 1048576); }

fn bench_direct_radix4(b: &mut Bencher, len: usize) {

    let dht = Arc::new(Radix4::new(len)) as Arc<dyn Dht<f32>>;

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| {dht.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn bench_direct_radix4_0000016(b: &mut Bencher) { bench_direct_radix4(b, 16); }
#[bench] fn bench_direct_radix4_0000032(b: &mut Bencher) { bench_direct_radix4(b, 32); }
#[bench] fn bench_direct_radix4_0000064(b: &mut Bencher) { bench_direct_radix4(b, 64); }
#[bench] fn bench_direct_radix4_0000128(b: &mut Bencher) { bench_direct_radix4(b, 128); }
#[bench] fn bench_direct_radix4_0000256(b: &mut Bencher) { bench_direct_radix4(b, 256); }
#[bench] fn bench_direct_radix4_0000512(b: &mut Bencher) { bench_direct_radix4(b, 512); }
#[bench] fn bench_direct_radix4_0001024(b: &mut Bencher) { bench_direct_radix4(b, 1024); }
#[bench] fn bench_direct_radix4_0002048(b: &mut Bencher) { bench_direct_radix4(b, 2048); }
#[bench] fn bench_direct_radix4_0016384(b: &mut Bencher) { bench_direct_radix4(b, 16384); }
#[bench] fn bench_direct_radix4_0032768(b: &mut Bencher) { bench_direct_radix4(b, 32768); }
#[bench] fn bench_direct_radix4_0065536(b: &mut Bencher) { bench_direct_radix4(b, 65536); }
#[bench] fn bench_direct_radix4_1048576(b: &mut Bencher) { bench_direct_radix4(b, 1048576); }

fn make_splitradix(len: usize) -> Arc<dyn Dht<f32>> {
    let mut transforms = vec![
        Arc::new(Butterfly1::new()) as Arc<dyn Dht<f32>>,
        Arc::new(Butterfly2::new()),
        Arc::new(Butterfly4::new()),
        Arc::new(Butterfly8::new()),
        Arc::new(Butterfly16::new()),
    ];

    let index = len.trailing_zeros() as usize;
    while transforms.len() <= index {
        let quarter = Arc::clone(&transforms[transforms.len() - 2]);
        let half = Arc::clone(&transforms[transforms.len() - 1]);

        transforms.push(Arc::new(SplitRadix::new(quarter, half)));
    }

    Arc::clone(transforms.last().unwrap())
}


fn bench_splitradix(b: &mut Bencher, len: usize) {

    let dht = make_splitradix(len);
    assert_eq!(dht.len(), len);

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| { dht.process_with_scratch(&mut buffer, &mut scratch); });
}

#[bench] fn bench_splitradix_00016(b: &mut Bencher) { bench_splitradix(b, 16); }
#[bench] fn bench_splitradix_00032(b: &mut Bencher) { bench_splitradix(b, 32); }
#[bench] fn bench_splitradix_00064(b: &mut Bencher) { bench_splitradix(b, 64); }
#[bench] fn bench_splitradix_00128(b: &mut Bencher) { bench_splitradix(b, 128); }
#[bench] fn bench_splitradix_00256(b: &mut Bencher) { bench_splitradix(b, 256); }
#[bench] fn bench_splitradix_00512(b: &mut Bencher) { bench_splitradix(b, 512); }
#[bench] fn bench_splitradix_01024(b: &mut Bencher) { bench_splitradix(b, 1024); }
#[bench] fn bench_splitradix_02048(b: &mut Bencher) { bench_splitradix(b, 2048); }
#[bench] fn bench_splitradix_16384(b: &mut Bencher) { bench_splitradix(b, 16384); }
#[bench] fn bench_splitradix_32768(b: &mut Bencher) { bench_splitradix(b, 32768); }
#[bench] fn bench_splitradix_65536(b: &mut Bencher) { bench_splitradix(b, 65536); }

fn make_3xn_radix3(len: usize) -> Arc<dyn Dht<f32>> {
    match len {
        0|1 => panic!(),
        3 => Arc::new(Butterfly3::new()),
        9 => Arc::new(Butterfly9::new()),
        _ => Arc::new(MixedRadix3xn::new(make_3xn_radix3(len/3))),
    }
}

fn bench_3xn_radix3(b: &mut Bencher, len: usize) {

    let dht = make_3xn_radix3(len);
    assert_eq!(dht.len(), len);

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| {dht.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn bench_3xn_radix3_00003(b: &mut Bencher) { bench_3xn_radix3(b, 3); }
#[bench] fn bench_3xn_radix3_00009(b: &mut Bencher) { bench_3xn_radix3(b, 9); }
#[bench] fn bench_3xn_radix3_00027(b: &mut Bencher) { bench_3xn_radix3(b, 27); }
#[bench] fn bench_3xn_radix3_00081(b: &mut Bencher) { bench_3xn_radix3(b, 81); }
#[bench] fn bench_3xn_radix3_00243(b: &mut Bencher) { bench_3xn_radix3(b, 243); }
#[bench] fn bench_3xn_radix3_02187(b: &mut Bencher) { bench_3xn_radix3(b, 2187); }
#[bench] fn bench_3xn_radix3_06561(b: &mut Bencher) { bench_3xn_radix3(b, 6561); }
#[bench] fn bench_3xn_radix3_19683(b: &mut Bencher) { bench_3xn_radix3(b, 19683); }
#[bench] fn bench_3xn_radix3_59049(b: &mut Bencher) { bench_3xn_radix3(b, 59049); }

fn make_5xn_radix5(len: usize) -> Arc<dyn Dht<f32>> {
    match len {
        0|1 => panic!(),
        5 => Arc::new(Butterfly5::new()),
        _ => Arc::new(MixedRadix5xn::new(make_5xn_radix5(len/5))),
    }
}

fn bench_5xn_radix5(b: &mut Bencher, len: usize) {

    let dht = make_5xn_radix5(len);
    assert_eq!(dht.len(), len);

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| {dht.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn bench_5xn_radix5_00005(b: &mut Bencher) { bench_5xn_radix5(b, 5); }
#[bench] fn bench_5xn_radix5_00025(b: &mut Bencher) { bench_5xn_radix5(b, 25); }
#[bench] fn bench_5xn_radix5_00125(b: &mut Bencher) { bench_5xn_radix5(b, 125); }
#[bench] fn bench_5xn_radix5_00625(b: &mut Bencher) { bench_5xn_radix5(b, 625); }
#[bench] fn bench_5xn_radix5_03125(b: &mut Bencher) { bench_5xn_radix5(b, 3125); }
#[bench] fn bench_5xn_radix5_15625(b: &mut Bencher) { bench_5xn_radix5(b, 15625); }
#[bench] fn bench_5xn_radix5_78125(b: &mut Bencher) { bench_5xn_radix5(b, 78125); }

fn make_6xn_radix6(len: usize) -> Arc<dyn Dht<f32>> {
    match len {
        0|1 => panic!(),
        6 => Arc::new(Butterfly6::new()),
        _ => Arc::new(MixedRadix6xn::new(make_6xn_radix6(len/6))),
    }
}

fn bench_6xn_radix6(b: &mut Bencher, len: usize) {

    let dht = make_6xn_radix6(len);
    assert_eq!(dht.len(), len);

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| {dht.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn bench_6xn_radix6_00006(b: &mut Bencher) { bench_6xn_radix6(b, 6); }
#[bench] fn bench_6xn_radix6_00036(b: &mut Bencher) { bench_6xn_radix6(b, 36); }
#[bench] fn bench_6xn_radix6_00216(b: &mut Bencher) { bench_6xn_radix6(b, 216); }
#[bench] fn bench_6xn_radix6_01296(b: &mut Bencher) { bench_6xn_radix6(b, 1296); }
#[bench] fn bench_6xn_radix6_07776(b: &mut Bencher) { bench_6xn_radix6(b, 7776); }
#[bench] fn bench_6xn_radix6_46656(b: &mut Bencher) { bench_6xn_radix6(b, 46656); }

fn bench_raders_noop(b: &mut Bencher, len: usize) {

    let inner_dht = Arc::new(Noop { len: len - 1 });

    let dht : Arc<Dht<_>> = Arc::new(RadersAlgorithm::new(inner_dht));

    let mut buffer = vec![0_f32; dht.len()];
    let mut scratch = vec![0_f32; dht.get_inplace_scratch_len()];
    b.iter(|| {dht.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn raders_noop_00031(b: &mut Bencher) { bench_raders_noop(b, 31); }
#[bench] fn raders_noop_00097(b: &mut Bencher) { bench_raders_noop(b, 97); }
#[bench] fn raders_noop_00257(b: &mut Bencher) { bench_raders_noop(b, 257); }
#[bench] fn raders_noop_01031(b: &mut Bencher) { bench_raders_noop(b, 1031); }
#[bench] fn raders_noop_04099(b: &mut Bencher) { bench_raders_noop(b, 4099); }
#[bench] fn raders_noop_16411(b: &mut Bencher) { bench_raders_noop(b, 16411); }
#[bench] fn raders_noop_65537(b: &mut Bencher) { bench_raders_noop(b, 65537); }

fn bench_raders_power2(b: &mut Bencher, len: usize) {

    let inner_dht = Arc::new(Radix4::new(len - 1));

    let fft : Arc<Dht<f32>> = Arc::new(RadersAlgorithm::new(inner_dht));

    let mut buffer = vec![0_f32; fft.len()];
    let mut scratch = vec![0_f32; fft.get_inplace_scratch_len()];
    b.iter(|| {fft.process_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn raders_power2_00017(b: &mut Bencher) { bench_raders_power2(b, 17); }
#[bench] fn raders_power2_00257(b: &mut Bencher) { bench_raders_power2(b, 257); }
#[bench] fn raders_power2_65537(b: &mut Bencher) { bench_raders_power2(b, 65537); }

#[bench] fn bench_butterfly3(b: &mut Bencher) { 
    let dht = Arc::new(Butterfly3::<f32>::new()) as Arc<dyn Dht<f32>>;

    let mut buffer = vec![0_f32; dht.len() * 100];
    b.iter(|| { dht.process_with_scratch(&mut buffer, &mut []); });
}
#[bench] fn bench_butterfly6(b: &mut Bencher) { 
    let dht = Arc::new(Butterfly6::<f32>::new()) as Arc<dyn Dht<f32>>;

    let mut buffer = vec![0_f32; dht.len() * 100];
    b.iter(|| { dht.process_with_scratch(&mut buffer, &mut []); });
}