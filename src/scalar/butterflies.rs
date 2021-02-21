use num_complex::Complex;
use rustfft::FftNum;

use crate::array_utils;
use crate::array_utils::{RawSlice, RawSliceMut};
use crate::{dht_error_inplace, dht_error_outofplace};
use crate::twiddles;
use crate::{Dht, Length};

#[allow(unused)]
macro_rules! boilerplate_fft_butterfly {
    ($struct_name:ident, $len:expr) => {
        impl<T: FftNum> $struct_name<T> {
            #[inline(always)]
            pub(crate) unsafe fn perform_dht_butterfly(&self, buffer: &mut [T]) {
                self.perform_dht_contiguous(RawSlice::new(buffer), RawSliceMut::new(buffer));
            }
            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn perform_dht_array(&self, buffer: &mut [T; $len]) {
                unsafe { self.perform_dht_contiguous(RawSlice::new(buffer), RawSliceMut::new(buffer)) };
            }
        }
        impl<T: FftNum> Dht<T> for $struct_name<T> {
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [T],
                output: &mut [T],
                _scratch: &mut [T],
            ) {
                if input.len() < self.len() || output.len() != input.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    dht_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                    return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
                }

                let result = array_utils::iter_chunks_zipped(
                    input,
                    output,
                    self.len(),
                    |in_chunk, out_chunk| {
                        unsafe {
                            self.perform_dht_contiguous(
                                RawSlice::new(in_chunk),
                                RawSliceMut::new(out_chunk),
                            )
                        };
                    },
                );

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    dht_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                }
            }
            fn process_with_scratch(&self, buffer: &mut [T], _scratch: &mut [T]) {
                if buffer.len() < self.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    dht_error_inplace(self.len(), buffer.len(), 0, 0);
                    return; // Unreachable, because fft_error_inplace asserts, but it helps codegen to put it here
                }

                let result = array_utils::iter_chunks(buffer, self.len(), |chunk| unsafe {
                    self.perform_dht_butterfly(chunk)
                });

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    dht_error_inplace(self.len(), buffer.len(), 0, 0);
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                0
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len
            }
        }
    };
}

pub struct Butterfly1<T> {
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FftNum> Butterfly1<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData, }
    }
}
impl<T: FftNum> Dht<T> for Butterfly1<T> {
    fn process_outofplace_with_scratch(
        &self,
        input: &mut [T],
        output: &mut [T],
        _scratch: &mut [T],
    ) {
        output.copy_from_slice(&input);
    }

    fn process_with_scratch(&self, _buffer: &mut [T], _scratch: &mut [T]) {}
    fn get_inplace_scratch_len(&self) -> usize { 0 }
    fn get_outofplace_scratch_len(&self) -> usize { 0 }
}
impl<T> Length for Butterfly1<T> {
    fn len(&self) -> usize { 1 }
}

pub struct Butterfly2<T> {
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_butterfly!(Butterfly2, 2);
impl<T: FftNum> Butterfly2<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData, }
    }
    #[inline(always)]
    pub(crate) fn perform_dht_strided(left: &mut T, right: &mut T) {
        let temp = *left + *right;

        *right = *left - *right;
        *left = temp;
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_dht_indexed(buffer: &mut [T], index0: usize, index1: usize) {
        let input0 = *buffer.get_unchecked(index0);
        let input1 = *buffer.get_unchecked(index1);

        *buffer.get_unchecked_mut(index0) = input0 + input1;
        *buffer.get_unchecked_mut(index1) = input0 - input1;
    }
    #[inline(always)]
    unsafe fn perform_dht_contiguous(
        &self,
        input: RawSlice<T>,
        output: RawSliceMut<T>,
    ) {
        let value0 = input.load(0);
        let value1 = input.load(1);
        output.store(value0 + value1, 0);
        output.store(value0 - value1, 1);
    }
}


pub struct Butterfly3<T> {
    twiddle: Complex<T>,
}
boilerplate_fft_butterfly!(Butterfly3, 3);
impl<T: FftNum> Butterfly3<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { twiddle: twiddles::compute_dft_twiddle_inverse(1, 3), }
    }
    #[inline(always)]
    unsafe fn perform_dht_contiguous(
        &self,
        input: RawSlice<T>,
        output: RawSliceMut<T>,
    ) {
        let input0 = input.load(0);
        let mut input1 = input.load(1);
        let mut input2 = input.load(2);

        Butterfly2::perform_dht_strided(&mut input1, &mut input2);

        let output0 = input0 + input1;
        input1 = input1 * self.twiddle.re + input0;
        input2 = input2 * self.twiddle.im;

        let output1 = input1 + input2;
        let output2 = input1 - input2;

        output.store(output0, 0);
        output.store(output1, 1);
        output.store(output2, 2);
    }
}


pub struct Butterfly4<T> {
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_butterfly!(Butterfly4, 4);
impl<T: FftNum> Butterfly4<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData, }
    }
    #[inline(always)]
    unsafe fn perform_dht_contiguous(
        &self,
        input: RawSlice<T>,
        output: RawSliceMut<T>,
    ) {
        // Algorithm: Six-Step DHT with width=2 and height=2
        let mut tmp0 = [
            input.load(0),
            input.load(2),
        ];
        let mut tmp1 = [
            input.load(1),
            input.load(3),
        ];

        let butterfly2 = Butterfly2::new();
        butterfly2.perform_dht_butterfly(&mut tmp0);
        butterfly2.perform_dht_butterfly(&mut tmp1);

        Butterfly2::perform_dht_strided(&mut tmp0[0], &mut tmp1[0]);
        Butterfly2::perform_dht_strided(&mut tmp0[1], &mut tmp1[1]);

        output.store(tmp0[0], 0);
        output.store(tmp0[1], 1);
        output.store(tmp1[0], 2);
        output.store(tmp1[1], 3);
    }
}

pub struct Butterfly5<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}
boilerplate_fft_butterfly!(Butterfly5, 5);
impl<T: FftNum> Butterfly5<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            twiddle1: twiddles::compute_dft_twiddle_inverse::<T>(1, 5),
            twiddle2: twiddles::compute_dft_twiddle_inverse::<T>(2, 5),
        }
    }
    #[inline(always)]
    unsafe fn perform_dht_contiguous(
        &self,
        input: RawSlice<T>,
        output: RawSliceMut<T>,
    ) {
        // Algorithm from https://arxiv.org/pdf/1502.01038.pdf "A factorization scheme for some discrete hartley transform matrices" by Olibeira, Cintra, Campello de Souza
        let input0 = input.load(0);
        let input1 = input.load(1);
        let input2 = input.load(2);
        let input3 = input.load(3);
        let input4 = input.load(4);

        let a = [
            input1 + input4,
            input1 - input4,
            input2 + input3,
            input2 - input3,
        ];

        let a01 = a[0] * self.twiddle1.re;
        let a02 = a[0] * self.twiddle2.re;
        let a11 = a[1] * self.twiddle1.im;
        let a12 = a[1] * self.twiddle2.im;
        let a21 = a[2] * self.twiddle1.re;
        let a22 = a[2] * self.twiddle2.re;
        let a31 = a[3] * self.twiddle1.im;
        let a32 = a[3] * self.twiddle2.im;

        let a01a22 = a01 + a22;
        let a02a21 = a02 + a21;
        let a11a32 = a11 + a32;
        let a12a31 = a12 - a31;

        let out0: T = input0 + a[0] + a[2];
        let out1: T = input0 + a01a22 + a11a32;
        let out2: T = input0 + a02a21 + a12a31;
        let out3: T = input0 + a02a21 - a12a31;
        let out4: T = input0 + a01a22 - a11a32;

        output.store(out0, 0);
        output.store(out1, 1);
        output.store(out2, 2);
        output.store(out3, 3);
        output.store(out4, 4);
    }
}

pub struct Butterfly6<T> {
    butterfly3: Butterfly3<T>,
}
boilerplate_fft_butterfly!(Butterfly6, 6);
impl<T: FftNum> Butterfly6<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            butterfly3: Butterfly3::new(),
        }
    }
    #[inline(always)]
    unsafe fn perform_dht_contiguous(
        &self,
        input: RawSlice<T>,
        output: RawSliceMut<T>,
    ) {
        // algorithm: MixedRadix2xn with an inner DHT of Butterfly3
        let mut chunk0 = [
            input.load(0),
            input.load(2),
            input.load(4),
        ];
        let mut chunk1 = [
            input.load(1),
            input.load(3),
            input.load(5),
        ];

        self.butterfly3.perform_dht_array(&mut chunk0);
        self.butterfly3.perform_dht_array(&mut chunk1);

        let input_top_fwd = chunk1[1];
        let input_top_rev = chunk1[2];

        chunk1[1] = self.butterfly3.twiddle.im * input_top_rev - self.butterfly3.twiddle.re * input_top_fwd;
        chunk1[2] =-self.butterfly3.twiddle.re * input_top_rev - self.butterfly3.twiddle.im * input_top_fwd;

        Butterfly2::perform_dht_strided(&mut chunk0[0], &mut chunk1[0]);
        Butterfly2::perform_dht_strided(&mut chunk0[1], &mut chunk1[1]);
        Butterfly2::perform_dht_strided(&mut chunk0[2], &mut chunk1[2]);

        output.store(chunk0[0],0);
        output.store(chunk0[1],1);
        output.store(chunk1[2],2);
        output.store(chunk1[0],3);
        output.store(chunk1[1],4);
        output.store(chunk0[2],5);
    }
}

pub struct Butterfly8<T> {
    twiddle: T,
}
boilerplate_fft_butterfly!(Butterfly8, 8);
impl<T: FftNum> Butterfly8<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { twiddle: twiddles::compute_dht_twiddle(1, 8), }
    }
    #[inline(always)]
    unsafe fn perform_dht_contiguous(
        &self,
        input: RawSlice<T>,
        output: RawSliceMut<T>,
    ) {
        // Algorithm from https://www.researchgate.net/publication/261048572_Matrix_expansions_for_computing_the_discrete_hartley_transform_for_blocklength_N_0_mod_4
        // "Matrix expansions for computing the Discrete Hartley Transform for blocklength N%4==0" by Oliveira and Campello de Souza
        // This is basically a radix 2 decomposition, with a few twiddle factors merged, etc
        let mut chunk0 = [
            input.load(0),
            input.load(4),
            input.load(2),
            input.load(6),
        ];
        let mut chunk1 = [
            input.load(1),
            input.load(5),
            input.load(3),
            input.load(7),
        ];

        let butterfly2 = Butterfly2::new();
        butterfly2.perform_dht_butterfly(&mut chunk0[0..2]);
        butterfly2.perform_dht_butterfly(&mut chunk0[2..4]);
        butterfly2.perform_dht_butterfly(&mut chunk1[0..2]);
        butterfly2.perform_dht_butterfly(&mut chunk1[2..4]);

        let (split0, split1) = chunk0.split_at_mut(2);
        let (split2, split3) = chunk1.split_at_mut(2);

        Butterfly2::perform_dht_strided(&mut split0[0], &mut split1[0]);
        Butterfly2::perform_dht_strided(&mut split0[1], &mut split1[1]);
        Butterfly2::perform_dht_strided(&mut split2[0], &mut split3[0]);

        chunk1[1] = chunk1[1] * self.twiddle;
        chunk1[3] = chunk1[3] * self.twiddle;

        for i in 0..4 {
            Butterfly2::perform_dht_strided(&mut chunk0[i], &mut chunk1[i]);
        }

        output.store(chunk0[0], 0);
        output.store(chunk0[1], 1);
        output.store(chunk0[2], 2);
        output.store(chunk0[3], 3);
        output.store(chunk1[0], 4);
        output.store(chunk1[1], 5);
        output.store(chunk1[2], 6);
        output.store(chunk1[3], 7);
    }
}


pub struct Butterfly9<T> {
    butterfly3_twiddle: T,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle4: Complex<T>,
}
boilerplate_fft_butterfly!(Butterfly9, 9);
impl<T: FftNum> Butterfly9<T> {
    #[inline(always)]
    pub fn new() -> Self {
        let butterfly_twiddle =  twiddles::compute_dft_twiddle_forward::<T>(1, 3);
        Self {
            butterfly3_twiddle: butterfly_twiddle.re - butterfly_twiddle.im,
            twiddle1: twiddles::compute_dft_twiddle_inverse(1, 9),
            twiddle2: twiddles::compute_dft_twiddle_inverse(2, 9),
            twiddle4: twiddles::compute_dft_twiddle_inverse(4, 9),
        }
    }
    #[inline(always)]
    unsafe fn perform_dht_contiguous(
        &self,
        input: RawSlice<T>,
        output: RawSliceMut<T>,
    ) {
        // Algorithm: https://www.researchgate.net/publication/3315015_Fast_Radix-39_Discrete_Hartley_Transform
        // "Fast Radix-3/9 Discrete Hartley Transform" by Lun, Siu
        // The formatting of the code is a garbage fire - I just took it dorectly out of the paper, line for line.
        // It works and it's fast, it just needs to be clenaed up.
        let t1: T = (input.load(3) - input.load(6)) * self.butterfly3_twiddle;
        let t2: T = input.load(0) - input.load(6);
        let t3: T = input.load(0) - input.load(3);
        let w1: T = t2 + t1;
        let w2: T = t3 - t1;
        //
        let x0: T = input.load(0) + input.load(3) + input.load(6);
        let x3: T = input.load(1) + input.load(4) + input.load(7);
        let x6: T = input.load(2) + input.load(5) + input.load(8);
        let t1: T = (x3 - x6) * self.butterfly3_twiddle;
        let t2: T = x0 + x3 + x6;
        let t3: T = x0 - x6 + t1;
        let x6: T = x0 - x3 - t1;
        let x3: T = t3;
        let x0: T = t2;
        //
        let t1: T = input.load(7) - input.load(4);
        let t2: T = input.load(8) - input.load(5);
        let t3: T = input.load(1) - input.load(4);
        let t4: T = input.load(2) - input.load(5);
        let v1: T = t1 + t4;
        let t4: T = t1 - t4;
        let t1: T = v1;
        let v2: T = t2 + t3;
        let t3: T = t2 - t3;
        let t2: T = v2;
        let v1: T = (t1 - t2) * self.twiddle4.re;
        let v2: T = t1 * self.twiddle2.re;
        let v3: T = t2 * self.twiddle1.re;
        let t1: T = v1 - v3;
        let t2: T = -v1 - v2;
        let v1: T = (t3 + t4) * self.twiddle4.im;
        let v2: T = t3 * self.twiddle1.im;
        let v3: T = t4 * self.twiddle2.im;
        let t3: T = -v1 - v2;
        let t4: T = v3 - v1;
        let x2: T = t1 + t3;
        let x4: T = t2 + t4;
        let x7: T = t1 - t3;
        let x5: T = t2 - t4;
        let x1: T = -x7 - x4 + w1;
        let x8: T = -x5 - x2 + w2;
        let x2 = x2 + w2;
        let x4 = x4 + w1;
        let x7 = x7 + w1;
        let x5 = x5 + w2;

        output.store(x0, 0);
        output.store(x1, 1);
        output.store(x2, 2);
        output.store(x3, 3);
        output.store(x4, 4);
        output.store(x5, 5);
        output.store(x6, 6);
        output.store(x7, 7);
        output.store(x8, 8);
    }
}


pub struct Butterfly16<T> {
    butterfly8: Butterfly8<T>,
    twiddle: Complex<T>,
}
boilerplate_fft_butterfly!(Butterfly16, 16);
impl<T: FftNum> Butterfly16<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { 
            butterfly8: Butterfly8::new(),
            twiddle: twiddles::compute_dft_twiddle_inverse(1, 16),
        }
    }
    #[inline(always)]
    unsafe fn perform_dht_contiguous(
        &self,
        input: RawSlice<T>,
        output: RawSliceMut<T>,
    ) {
        // Algorithm from https://pdfs.semanticscholar.org/4ff6/43d640ff796e44669a780bd9a6de9f18118e.pdf
        // "Split-radix fast Hartley transform" by Pei, Wu
        let mut chunk0 = [
            input.load(0),
            input.load(1),
            input.load(2),
            input.load(3),
            input.load(4),
            input.load(5),
            input.load(6),
            input.load(7),
        ];
        let mut chunk1 = [
            input.load(8),
            input.load(9),
            input.load(10),
            input.load(11),
            input.load(12),
            input.load(13),
            input.load(14),
            input.load(15),
        ];

        // first step is a set of butterfly 2's between the first half and second half
        for i in 0..8 {
            Butterfly2::perform_dht_strided(&mut chunk0[i], &mut chunk1[i]);
        }
        
        // second step is to do a butterfly 8 on the first half, and write the results out to the even-indexed outputs
        // Writing these out now frees up registers, to minimize the amount stack space we need
        self.butterfly8.perform_dht_butterfly(&mut chunk0);
        for i in 0..8 {
            output.store(chunk0[i], i*2);
        }

        // third step: apply some twiddle factors to the second half
        Butterfly2::perform_dht_indexed(&mut chunk1, 0, 4);
        Butterfly2::perform_dht_indexed(&mut chunk1, 1, 3);
        Butterfly2::perform_dht_indexed(&mut chunk1, 7, 5);

        let mut post_twiddle = [
            chunk1[0],
            chunk1[1] * self.twiddle.re + chunk1[5] * self.twiddle.im,
            chunk1[2] * self.butterfly8.twiddle,
            chunk1[1] * self.twiddle.im - chunk1[5] * self.twiddle.re,
            chunk1[4],
            chunk1[3] * self.twiddle.im + chunk1[7] * self.twiddle.re,
            chunk1[6] * self.butterfly8.twiddle,
            chunk1[3] * self.twiddle.re - chunk1[7] * self.twiddle.im,
        ];

        // Step 4: Inner DHTs of size N/4
        let (post_twiddle1, post_twiddle3) = post_twiddle.split_at_mut(4);
        Butterfly4::new().perform_dht_butterfly(post_twiddle1);
        Butterfly4::new().perform_dht_butterfly(post_twiddle3);

        for i in 0..4 {
            output.store(post_twiddle1[i], i*4 + 1);
            output.store(post_twiddle3[i], i*4 + 3);
        }
    }
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_dht_algorithm;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly32 = $struct_name::new();
                check_dht_algorithm::<f32>(&butterfly32, $size);

                let butterfly64 = $struct_name::new();
                check_dht_algorithm::<f64>(&butterfly64, $size);
            }
        };
    }
    test_butterfly_func!(test_butterfly2, Butterfly2, 2);
    test_butterfly_func!(test_butterfly3, Butterfly3, 3);
    test_butterfly_func!(test_butterfly4, Butterfly4, 4);
    test_butterfly_func!(test_butterfly5, Butterfly5, 5);
    test_butterfly_func!(test_butterfly6, Butterfly6, 6);
    test_butterfly_func!(test_butterfly8, Butterfly8, 8);
    test_butterfly_func!(test_butterfly9, Butterfly9, 9);
    test_butterfly_func!(test_butterfly16, Butterfly16, 16);
}
