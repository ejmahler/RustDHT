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
    unsafe fn perform_dht_strided(left: &mut T, right: &mut T) {
        let temp = *left + *right;

        *right = *left - *right;
        *left = temp;
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
    twiddle: T,
}
boilerplate_fft_butterfly!(Butterfly3, 3);
impl<T: FftNum> Butterfly3<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { twiddle: twiddles::compute_dht_twiddle(1, 3), }
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

        let mut value1 = input1;
        let mut value2 = input2;

        Butterfly2::perform_dht_strided(&mut value1, &mut value2);
        value2 = value2 * self.twiddle;

        let output0 = input0 + value1;
        let output1 = input0 + value2 - input2;
        let output2 = input0 - value2 - input1;

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
        // Algorithm from https://arxiv.org/pdf/1502.01038.pdf "Matrix expansions for computing the Discrete Hartley Transform for blocklength N%4==0" by Oliveira, and Campello de Souza
        // This is basically a radix 2 decomposition.
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
    test_butterfly_func!(test_butterfly8, Butterfly8, 8);
}
