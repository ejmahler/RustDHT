use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;
use rustfft::{FftNum, Length};

use crate::{Dht, array_utils, dht_error_inplace, dht_error_outofplace, twiddles};

use super::Butterfly3;

pub struct MixedRadix3xn<T> {
    twiddles: Box<[Complex<T>]>,
    butterfly3_twiddle: Complex<T>,

    height_size_fft: Arc<dyn Dht<T>>,
    height: usize,

    butterfly3: Butterfly3<T>,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    len: usize,
}

impl<T: FftNum> MixedRadix3xn<T> {
    /// Creates a DHT instance which will process inputs/outputs of size `height_dht.len() * 3`
    pub fn new(height_dht: Arc<dyn Dht<T>>) -> Self {
        let width = 3;
        let height = height_dht.len();

        let len = width * height;

        let twiddle_limit = height / 2 + 1;

        let mut twiddles = Vec::with_capacity(twiddle_limit * 2);
        let half = T::from_f64(0.5).unwrap();
        for x in 1..twiddle_limit {
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(x * 1, len) * half);
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(x * 2, len) * half);
        }

        // Collect some data about what kind of scratch space our inner DHTs need
        let height_inplace_scratch = height_dht.get_inplace_scratch_len();

        // Computing the scratch we'll require is a somewhat confusing process.
        // When we compute an out-of-place DHT, both of our inner DHTs are in-place
        // When we compute an inplace DHT, our inner width DHT will be inplace, and our height DHT will be out-of-place
        // For the out-of-place DHT, one of 2 things can happen regarding scratch:
        //      - If the required scratch of both DHTs is <= self.len(), then we can use the input or output buffer as scratch, and so we need 0 extra scratch
        //      - If either of the inner DHTs require more, then we'll have to request an entire scratch buffer for the inner DHTs,
        //          whose size is the max of the two inner DHTs' required scratch
        let max_inner_inplace_scratch = height_inplace_scratch;
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
            + 
                if height_inplace_scratch > len {
                    height_inplace_scratch
                } else {
                    0
                }
            ;
        Self {
            twiddles: twiddles.into_boxed_slice(),
            butterfly3_twiddle: twiddles::compute_dft_twiddle_inverse::<T>(1, 3),

            height_size_fft: height_dht,
            height,

            butterfly3: Butterfly3::new(),

            inplace_scratch_len,
            outofplace_scratch_len,
            len,
        }
    }

    #[inline(never)]
    fn apply_twiddles(&self, output: &mut [T]) {
        // For the rest of this function, we're going to be dealing with the buffer in 3 chunks simultaneously, so go ahead and make that split now
        let split_buffer = {
            let mut buffer_chunks = output.chunks_exact_mut(self.height);

            [
                buffer_chunks.next().unwrap(),
                buffer_chunks.next().unwrap(),
                buffer_chunks.next().unwrap(),
            ]
        };

        // first column is simpler than the rest - no twiddles required, and -column == column == 0, so we don't have to deal with forward/reverse stuff
        {
            let mut col0 = [
                split_buffer[0][0],
                split_buffer[1][0],
                split_buffer[2][0],
            ];

            unsafe { 
                self.butterfly3.perform_dht_butterfly(&mut col0);
            }

            split_buffer[0][0] = col0[0];
            split_buffer[1][0] = col0[1];
            split_buffer[2][0] = col0[2];
        }

        

        
        for (column, twiddle_chunk) in (1..self.height/2 + 1).zip(self.twiddles.chunks_exact(2)) {
             // we need -k % height, but k is unsigned, so do it without actual negatives
            let column_rev = self.height - column;

            let mut tmp_rev = [Zero::zero(); 3];
            tmp_rev[0] = split_buffer[0][column_rev];

            let fwd_top_twiddle = twiddle_chunk[0];
            let fwd_bot_twiddle = twiddle_chunk[1];

            let input_fwd = [
                split_buffer[0][column],
                split_buffer[1][column],
                split_buffer[2][column],
            ];
            let input_rev = [
                split_buffer[0][column_rev],
                split_buffer[1][column_rev],
                split_buffer[2][column_rev],
            ];

            let sum = [
                input_fwd[1] + input_rev[1],
                input_fwd[2] + input_rev[2]
            ];
            let diff = [
                input_fwd[1] - input_rev[1],
                input_fwd[2] - input_rev[2]
            ];

            let a = fwd_top_twiddle.re * sum[0] - fwd_top_twiddle.im * diff[0];
            let b = fwd_bot_twiddle.re * diff[1] + fwd_bot_twiddle.im * sum[1];
            let c = fwd_bot_twiddle.re * sum[1] - fwd_bot_twiddle.im * diff[1];
            let d = fwd_top_twiddle.re * diff[0] + fwd_top_twiddle.im * sum[0];
            let e = a - b;
            let f = c - d;

            let mut tmp_fwd = [
                input_fwd[0],
                a + b,
                c + d,
            ];

            let mut tmp_rev = [
                input_rev[0],
                self.butterfly3_twiddle.re * e - self.butterfly3_twiddle.im * f,
                self.butterfly3_twiddle.re * f + self.butterfly3_twiddle.im * e,
            ];

            self.butterfly3.perform_dht_array(&mut tmp_fwd);
            self.butterfly3.perform_dht_array(&mut tmp_rev);

            split_buffer[0][column] = tmp_fwd[0];
            split_buffer[1][column] = tmp_fwd[1];
            split_buffer[2][column] = tmp_fwd[2];

            split_buffer[0][column_rev] = tmp_rev[0];
            split_buffer[1][column_rev] = tmp_rev[1];
            split_buffer[2][column_rev] = tmp_rev[2];
        }
    }

    fn perform_dht_inplace(&self, buffer: &mut [T], _scratch: &mut [T]) {
        let mut scratch = vec![Zero::zero(); buffer.len()];

        // Step 1: Transpose the width x height array to height x width
        transpose::transpose(buffer, &mut scratch, 3, self.height);

        // Step 2: Compute DHTs of size `height` down the rows of our transposed array
        self.height_size_fft.process_outofplace_with_scratch(&mut scratch, buffer, &mut []);

        // Step 3: Apply twiddle factors
        self.apply_twiddles(buffer);
    }

    fn perform_dht_out_of_place(
        &self,
        input: &mut [T],
        output: &mut [T],
        _scratch: &mut [T],
    ) {
        // Step 1: Transpose the width x height array to height x width
        transpose::transpose(input, output, 3, self.height);

        // Step 2: Compute DHTs of size `height` down the rows of our transposed array
        self.height_size_fft.process_with_scratch(output, input);

        // Step 3: Apply twiddle factors
        self.apply_twiddles(output);
    }
}
boilerplate_dht!(MixedRadix3xn);


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_dht_algorithm;
    use crate::scalar::DhtNaive;
    use std::sync::Arc;

    #[test]
    fn test_mixed_radix_3xn_correct() {
        for height in 1..7 {
            test_mixed_radix_3xn_with_lengths(height);
        }
    }

    fn test_mixed_radix_3xn_with_lengths(height: usize) {
        let height_dht = Arc::new(DhtNaive::new(height)) as Arc<dyn Dht<f32>>;

        let dht = MixedRadix3xn::new(height_dht);

        check_dht_algorithm(&dht, height * 3);
    }
}