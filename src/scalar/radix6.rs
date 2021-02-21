use std::sync::Arc;

use num_complex::Complex;
use rustfft::{FftNum, Length};

use crate::{Dht, array_utils, dht_error_inplace, dht_error_outofplace, twiddles};

use super::Butterfly6;

pub struct MixedRadix6xn<T> {
    twiddles: Box<[Complex<T>]>,

    height_size_fft: Arc<dyn Dht<T>>,
    height: usize,

    butterfly6: Butterfly6<T>,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    len: usize,
}
boilerplate_dht!(MixedRadix6xn);
impl<T: FftNum> MixedRadix6xn<T> {
    /// Creates a DHT instance which will process inputs/outputs of size `height_dht.len() * 3`
    pub fn new(height_dht: Arc<dyn Dht<T>>) -> Self {
        let width = 6;
        let height = height_dht.len();

        let len = width * height;

        let column_limit = height / 2 + 1;

        let mut twiddles = Vec::with_capacity(3 * (column_limit - 1));
        let root2 = T::from_f32(0.5f32.sqrt()).unwrap();
        for column in 1..column_limit {
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(1 * column * 8 + len, len * 8) * root2);
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(2 * column * 8 + len, len * 8) * root2);
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(3 * column * 8 + len, len * 8) * root2);
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

            height_size_fft: height_dht,
            height,

            butterfly6: Butterfly6::new(),

            inplace_scratch_len,
            outofplace_scratch_len,
            len,
        }
    }

    #[inline(never)]
    fn apply_twiddles(&self, output: &mut [T]) {
        let column_limit = self.height / 2 + 1;

        // For the rest of this function, we're going to be dealing with the buffer in 3 chunks simultaneously, so go ahead and make that split now
        let split_buffer = {
            let mut buffer_chunks = output.chunks_exact_mut(self.height);

            [
                buffer_chunks.next().unwrap(),
                buffer_chunks.next().unwrap(),
                buffer_chunks.next().unwrap(),
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
                split_buffer[3][0],
                split_buffer[4][0],
                split_buffer[5][0],
            ];

            self.butterfly6.perform_dht_array(&mut col0);

            split_buffer[0][0] = col0[0];
            split_buffer[1][0] = col0[1];
            split_buffer[2][0] = col0[2];
            split_buffer[3][0] = col0[3];
            split_buffer[4][0] = col0[4];
            split_buffer[5][0] = col0[5];
        }

        for (column, twiddle_chunk) in (1..column_limit).zip(self.twiddles.chunks_exact(3)) {
            let column_rev = self.height - column;

            let input0_fwd = split_buffer[0][column];
            let input0_rev = split_buffer[0][column_rev];
            let input1_fwd = split_buffer[1][column];
            let input1_rev = split_buffer[1][column_rev];
            let input5_fwd = split_buffer[5][column];
            let input5_rev = split_buffer[5][column_rev];
            let input2_fwd = split_buffer[2][column];
            let input2_rev = split_buffer[2][column_rev];
            let input4_fwd = split_buffer[4][column];
            let input4_rev = split_buffer[4][column_rev];
            let input3_fwd = split_buffer[3][column];
            let input3_rev = split_buffer[3][column_rev];

            let out1_fwd = input1_fwd + input5_rev;
            let out1_rev = input5_fwd + input1_rev;
            let out5_fwd = input5_fwd - input1_rev;
            let out5_rev = input1_fwd - input5_rev;
            let out2_fwd = input2_fwd + input4_rev;
            let out2_rev = input4_fwd + input2_rev;
            let out4_fwd = input4_fwd - input2_rev;
            let out4_rev = input2_fwd - input4_rev;
            let out3_fwd = input3_fwd - input3_rev;
            let out3_rev = input3_fwd + input3_rev;

            let mut tmp_fwd = [
                input0_fwd,
                twiddle_chunk[0].re * out1_fwd - twiddle_chunk[0].im * out5_fwd,
                twiddle_chunk[1].re * out2_fwd - twiddle_chunk[1].im * out4_fwd,
                twiddle_chunk[2].re * out3_fwd + twiddle_chunk[2].im * out3_rev,
                twiddle_chunk[1].re * out4_fwd + twiddle_chunk[1].im * out2_fwd,
                twiddle_chunk[0].re * out5_fwd + twiddle_chunk[0].im * out1_fwd,
            ];

            let mut tmp_rev = [
                input0_rev,
                twiddle_chunk[0].re * out1_rev - twiddle_chunk[0].im * out5_rev,
                twiddle_chunk[1].re * out2_rev - twiddle_chunk[1].im * out4_rev,
                twiddle_chunk[2].re * out3_rev - twiddle_chunk[2].im * out3_fwd,
                twiddle_chunk[1].re * out4_rev + twiddle_chunk[1].im * out2_rev,
                twiddle_chunk[0].re * out5_rev + twiddle_chunk[0].im * out1_rev,
            ];

            self.butterfly6.perform_dht_array(&mut tmp_fwd);
            self.butterfly6.perform_dht_array(&mut tmp_rev);

            split_buffer[0][column]      = tmp_fwd[0];
            split_buffer[1][column]      = tmp_fwd[1];
            split_buffer[2][column]      = tmp_fwd[2];
            split_buffer[3][column]      = tmp_fwd[3];
            split_buffer[4][column]      = tmp_fwd[4];
            split_buffer[5][column]      = tmp_fwd[5];

            split_buffer[0][column_rev]  = tmp_rev[5];
            split_buffer[1][column_rev]  = tmp_rev[4];
            split_buffer[2][column_rev]  = tmp_rev[3];
            split_buffer[3][column_rev]  = tmp_rev[2];
            split_buffer[4][column_rev]  = tmp_rev[1];
            split_buffer[5][column_rev]  = tmp_rev[0];
        }
    }

    
    fn perform_dht_inplace(&self, buffer: &mut [T], scratch: &mut [T]) {
        // Step 1: Transpose the width x height array to height x width
        array_utils::transpose_half_rev_out(buffer, scratch, 6, self.height);

        // Step 2: Compute DHTs of size `height` down the rows of our transposed array
        self.height_size_fft.process_outofplace_with_scratch(scratch, buffer, &mut []);

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
        array_utils::transpose_half_rev_out(input, output, 6, self.height);

        // Step 2: Compute DHTs of size `height` down the rows of our transposed array
        self.height_size_fft.process_with_scratch(output, input);

        // Step 3: Apply twiddle factors
        self.apply_twiddles(output);
    }
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_dht_algorithm;
    use crate::scalar::DhtNaive;
    use std::sync::Arc;

    #[test]
    fn test_mixed_radix_6xn_correct() {
        for height in 1..7 {
            test_mixed_radix_6xn_with_lengths(height);
        }
    }

    fn test_mixed_radix_6xn_with_lengths(height: usize) {
        let height_dht = Arc::new(DhtNaive::new(height)) as Arc<dyn Dht<f32>>;

        let dht = MixedRadix6xn::new(height_dht);

        check_dht_algorithm(&dht, height * 6);
    }
}
