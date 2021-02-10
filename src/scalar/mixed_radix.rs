use std::{cmp::max, sync::Arc};

use num_complex::Complex;
use rustfft::{FftNum, Length};

use crate::{Dht, array_utils, dht_error_inplace, dht_error_outofplace, twiddles};

/// Implementation of the Mixed-Radix DHT algorithm
///
/// This algorithm factors a size n DHT into n1 * n2, computes several inner DHTs of size n1 and n2, then combines the
/// results to get the final answer
///
/// ~~~
/// // Computes a forward DHT of size 120, using the Mixed-Radix Algorithm
/// use rustdht::{Dht, scalar::{MixedRadix, DhtNaive}};
/// use std::sync::Arc;
///
/// // we need to find an n1 and n2 such that n1 * n2 == 120
/// // n1 = 12 and n2 = 10 satisfies this
/// let inner_fft_n1 = Arc::new(DhtNaive::new(10));
/// let inner_fft_n2 = Arc::new(DhtNaive::new(12));
///
/// // the mixed radix DHT length will be inner_fft_n1.len() * inner_fft_n2.len() = 120
/// let dht = MixedRadix::new(inner_fft_n1, inner_fft_n2);
///
/// let mut buffer = vec![0.0f32; 120];
/// dht.process(&mut buffer);
/// ~~~
pub struct MixedRadix<T> {
    twiddles: Box<[Complex<T>]>,

    width_size_fft: Arc<dyn Dht<T>>,
    width: usize,

    height_size_fft: Arc<dyn Dht<T>>,
    height: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    len: usize,
}

impl<T: FftNum> MixedRadix<T> {
    /// Creates a DHT instance which will process inputs/outputs of size `width_dht.len() * height_dht.len()`
    pub fn new(width_dht: Arc<dyn Dht<T>>, height_dht: Arc<dyn Dht<T>>) -> Self {
        let width = width_dht.len();
        let height = height_dht.len();

        let len = width * height;

        let row_limit = width / 2 + 1;
        let column_limit = height / 2 + 1;

        let mut twiddles = Vec::with_capacity(row_limit * column_limit);
        let root2 = T::from_f32(0.5f32.sqrt()).unwrap();
        for row in 1..row_limit {
            for column in 1..column_limit {
                // Our twiddle will be a one-eigth turn around the circle from what the basic "row * column" twiddle would have been, times sqrt(0.5)
                // The one-eigth turn times the sqrt comes fro mthe fact that we factored a
                // (twiddle.re - twiddle.im) and (twiddle.re + twiddle.im) out of every single twiddle multiplication in `apply_twiddles`, and that's equivalent to an eigth turn
                let twiddle = twiddles::compute_dft_twiddle_inverse::<T>(row * column * 8 + len, len * 8) * root2;

                twiddles.push(twiddle);
            }
        }

        // Collect some data about what kind of scratch space our inner DHTs need
        let height_inplace_scratch = height_dht.get_inplace_scratch_len();
        let width_inplace_scratch = width_dht.get_inplace_scratch_len();
        let width_outofplace_scratch = width_dht.get_outofplace_scratch_len();

        // Computing the scratch we'll require is a somewhat confusing process.
        // When we compute an out-of-place DHT, both of our inner DHTs are in-place
        // When we compute an inplace DHT, our inner width DHT will be inplace, and our height DHT will be out-of-place
        // For the out-of-place DHT, one of 2 things can happen regarding scratch:
        //      - If the required scratch of both DHTs is <= self.len(), then we can use the input or output buffer as scratch, and so we need 0 extra scratch
        //      - If either of the inner DHTs require more, then we'll have to request an entire scratch buffer for the inner DHTs,
        //          whose size is the max of the two inner DHTs' required scratch
        let max_inner_inplace_scratch = max(height_inplace_scratch, width_inplace_scratch);
        let outofplace_scratch_len = if max_inner_inplace_scratch > len {
            max_inner_inplace_scratch
        } else {
            0
        };

        // For the in-place DHT, again the best case is that we can just bounce data around between internal buffers, and the only inplace scratch we need is self.len()
        // If our width fft's OOP DHT requires any scratch, then we can tack that on the end of our own scratch, and use split_at_mut to separate our own from our internal DHT's
        // Likewise, if our height inplace DHT requires more inplace scracth than self.len(), we can tack that on to the end of our own inplace scratch.
        // Thus, the total inplace scratch is our own length plus the max of what the two inner DHTs will need
        let inplace_scratch_len = len
            + max(
                if height_inplace_scratch > len {
                    height_inplace_scratch
                } else {
                    0
                },
                width_outofplace_scratch,
            );

        Self {
            twiddles: twiddles.into_boxed_slice(),

            width_size_fft: width_dht,
            width: width,

            height_size_fft: height_dht,
            height: height,

            inplace_scratch_len,
            outofplace_scratch_len,
            len,
        }
    }

    #[inline(never)]
    fn apply_twiddles(&self, buffer: &mut [T]) {
        // Instead of just multiplying a single input vlaue with a single complex number like we do in the DFT,
        // we need to combine 4 numbers, determined by mirroring the input number across the horizontal and vertical axes of the array

        // So we're going to iterate over the first half of the rows, and the reversed second half of the rows simultaneously
        // then, for each pair of rows, we'll loop over the first half of the elements, and the reversed second half of the elements simultaneously
        // that will give us 4 mirrored elements to process in-place for each iteration of the loop

        // Note that there's a special case here: If width or height are even, we'll get into situations where some of the 4 elements are the same
        // IE, if the height is even, we'll get into a situation in our inner loop where y == y_bottom
        // And, if the width is even, we'll get into a situation in our outer loop where x == x_rev
        // In these situations, we *could* split out the loop and handle these special cases to optimize out the redundant floating point ops
        // but that adds significant complexity, and benchmarking shows that the added complexity actually hurts performance more than the reduced ops helps

        let row_limit = (self.width + 1) / 2;
        let column_limit = self.height / 2 + 1;
        let twiddle_stride = column_limit - 1;

        if self.width < 2 || self.height < 2 {
            return;
        }

        let mut twiddles_iter = self.twiddles.chunks_exact(twiddle_stride);
        for (row, twiddle_chunk) in (1..row_limit).zip(twiddles_iter.by_ref()) {
            let row_rev = self.width - row;
            
            for (column,  twiddle) in (1..column_limit).zip(twiddle_chunk.iter()) {
                let column_rev = self.height - column;

                let input_top_fwd = buffer[self.height * row + column];
                let input_top_rev = buffer[self.height * row + column_rev];
                let input_bot_fwd = buffer[self.height * row_rev + column];
                let input_bot_rev = buffer[self.height * row_rev + column_rev];

                let out_top_fwd = input_top_fwd + input_bot_rev;
                let out_top_rev = input_bot_fwd + input_top_rev;
                let out_bot_fwd = input_bot_fwd - input_top_rev;
                let out_bot_rev = input_top_fwd - input_bot_rev;

                buffer[self.height * row + column]          = twiddle.re * out_top_fwd - twiddle.im * out_bot_fwd;
                buffer[self.height * row + column_rev]      = twiddle.re * out_top_rev - twiddle.im * out_bot_rev;
                buffer[self.height * row_rev + column]      = twiddle.re * out_bot_fwd + twiddle.im * out_top_fwd;
                buffer[self.height * row_rev + column_rev]  = twiddle.re * out_bot_rev + twiddle.im * out_top_rev;
            }
        }
        if self.width % 2 == 0 {
            let row = self.width / 2;
            let main_twiddle_chunk = twiddles_iter.next().unwrap();

            for (column, twiddle) in (1..column_limit).zip(main_twiddle_chunk.iter()) {
                let column_rev = self.height - column;

                let input_top_fwd = buffer[self.height * row + column];
                let input_top_rev = buffer[self.height * row + column_rev];

                let out_top_fwd = input_top_fwd - input_top_rev;
                let out_bot_fwd = input_top_fwd + input_top_rev;

                buffer[self.height * row + column]      = twiddle.re * out_top_fwd + twiddle.im * out_bot_fwd;
                buffer[self.height * row + column_rev]  = twiddle.re * out_bot_fwd - twiddle.im * out_top_fwd;
            }
        }
    }

    fn perform_dht_inplace(&self, buffer: &mut [T], scratch: &mut [T]) {
        // SIX STEP DHT:
        let (scratch, inner_scratch) = scratch.split_at_mut(self.len());

        // STEP 1: transpose
        transpose::transpose(buffer, scratch, self.width, self.height);

        // STEP 2: perform DHTs of size `height`
        let height_scratch = if inner_scratch.len() > buffer.len() {
            &mut inner_scratch[..]
        } else {
            &mut buffer[..]
        };

        let reversal_row_begin = if self.width % 2 == 0 { self.width / 2 + 1 } else { (self.width + 1) / 2 };
        let second_half = &mut scratch[reversal_row_begin * self.height..];
        for chunk in second_half.chunks_exact_mut(self.height) {
            chunk.reverse();
        }

        self.height_size_fft
            .process_with_scratch(scratch, height_scratch);

        // STEP 3: Apply twiddle factors
        self.apply_twiddles(scratch);

        // STEP 4: transpose again
        transpose::transpose(scratch, buffer, self.height, self.width);

        // STEP 5: perform DHTs of size `width`
        self.width_size_fft
            .process_outofplace_with_scratch(buffer, scratch, inner_scratch);

        // reverse the second half of the width DHT output, to finalize our twiddle factors
        let reversal_row_begin = (self.height + 1) / 2;
        let second_half = &mut scratch[reversal_row_begin * self.width..];
        for chunk in second_half.chunks_exact_mut(self.width) {
            chunk.reverse();
        }

        // STEP 6: transpose again
        transpose::transpose(scratch, buffer, self.width, self.height);
    }

    fn perform_dht_out_of_place(
        &self,
        input: &mut [T],
        output: &mut [T],
        scratch: &mut [T],
    ) {
        // SIX STEP DHT:

        // STEP 1: transpose
        transpose::transpose(input, output, self.width, self.height);

        // STEP 2: perform DHTs of size `height`
        let height_scratch = if scratch.len() > input.len() {
            &mut scratch[..]
        } else {
            &mut input[..]
        };

        let reversal_row_begin = if self.width % 2 == 0 { self.width / 2 + 1 } else { (self.width + 1) / 2 };
        let second_half = &mut output[reversal_row_begin * self.height..];
        for chunk in second_half.chunks_exact_mut(self.height) {
            chunk.reverse();
        }

        self.height_size_fft
            .process_with_scratch(output, height_scratch);

        // STEP 3: Apply twiddle factors
        self.apply_twiddles(output);

        // STEP 4: transpose again
        transpose::transpose(output, input, self.height, self.width);

        // STEP 5: perform DHTs of size `width`
        let width_scratch = if scratch.len() > output.len() {
            &mut scratch[..]
        } else {
            &mut output[..]
        };
        self.width_size_fft
            .process_with_scratch(input, width_scratch);

        // reverse the second half of the width DHT output, to finalize our twiddle factors
        let reversal_row_begin = (self.height + 1) / 2;
        let second_half = &mut input[reversal_row_begin * self.width..];
        for chunk in second_half.chunks_exact_mut(self.width) {
            chunk.reverse();
        }

        // STEP 6: transpose again
        transpose::transpose(input, output, self.width, self.height);
    }
}
boilerplate_dht!(MixedRadix);


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_dht_algorithm;
    use crate::{scalar::DhtNaive, test_utils::BigScratchAlgorithm};
    use num_traits::Zero;
    use std::sync::Arc;

    #[test]
    fn test_mixed_radix_correct() {
        for width in 1..9 {
            for height in 1..9 {
                test_mixed_radix_with_lengths(width, height);
            }
        }
    }

    fn test_mixed_radix_with_lengths(width: usize, height: usize) {
        let width_dht = Arc::new(DhtNaive::new(width)) as Arc<dyn Dht<f32>>;
        let height_dht = Arc::new(DhtNaive::new(height)) as Arc<dyn Dht<f32>>;

        let dht = MixedRadix::new(width_dht, height_dht);

        check_dht_algorithm(&dht, width * height);
    }

    // Verify that the mixed radix algorithm correctly provides scratch space to inner FFTs
    #[test]
    fn test_mixed_radix_inner_scratch() {
        let scratch_lengths = [1, 5, 25];

        let mut inner_dhts = Vec::new();

        for &len in &scratch_lengths {
            for &inplace_scratch in &scratch_lengths {
                for &outofplace_scratch in &scratch_lengths {
                    inner_dhts.push(Arc::new(BigScratchAlgorithm {
                        len,
                        inplace_scratch,
                        outofplace_scratch,
                    }) as Arc<dyn Dht<f32>>);
                }
            }
        }

        for width_dht in inner_dhts.iter() {
            for height_dht in inner_dhts.iter() {
                let dht = MixedRadix::new(Arc::clone(width_dht), Arc::clone(height_dht));

                let mut inplace_buffer = vec![Zero::zero(); dht.len()];
                let mut inplace_scratch = vec![Zero::zero(); dht.get_inplace_scratch_len()];

                dht.process_with_scratch(&mut inplace_buffer, &mut inplace_scratch);

                let mut outofplace_input = vec![Zero::zero(); dht.len()];
                let mut outofplace_output = vec![Zero::zero(); dht.len()];
                let mut outofplace_scratch =
                    vec![Zero::zero(); dht.get_outofplace_scratch_len()];
                dht.process_outofplace_with_scratch(
                    &mut outofplace_input,
                    &mut outofplace_output,
                    &mut outofplace_scratch,
                );
            }
        }
    }
}