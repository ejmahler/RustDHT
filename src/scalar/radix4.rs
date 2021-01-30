use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;
use rustfft::{FftNum, Length};

use crate::{Dht, array_utils, dht_error_inplace, dht_error_outofplace, twiddles};

use super::{Butterfly2, Butterfly4};

pub struct MixedRadix4xn<T> {
    twiddles: Box<[Complex<T>]>,
    height_size_fft: Arc<dyn Dht<T>>,
    height: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    len: usize,
}

impl<T: FftNum> MixedRadix4xn<T> {
    /// Creates a DHT instance which will process inputs/outputs of size `height_dht.len() * 4`
    pub fn new(height_dht: Arc<dyn Dht<T>>) -> Self {
        let width = 4;
        let height = height_dht.len();

        let len = width * height;

        let twiddle_limit = height / 2 + 1;

        let mut twiddles = Vec::with_capacity(twiddle_limit * 3);
        for x in 1..twiddle_limit {
            for y in 1..4 {
                twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(x * y, len));
            }
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

            inplace_scratch_len,
            outofplace_scratch_len,
            len,
        }
    }

    #[inline(never)]
    fn apply_twiddles(&self, output: &mut [T]) {
        let butterfly4 = Butterfly4::new();

        // For the rest of this function, we're going to be dealing with the buffer in 4 chunks simultaneously, so go ahead and make that split now
        let split_buffer = {
            let mut buffer_chunks = output.chunks_exact_mut(self.height);

            [
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
            ];

            unsafe { 
                butterfly4.perform_dht_butterfly(&mut col0);
            }

            for i in 0..4 {
                split_buffer[i][0] = col0[i];
            }
        }

        
        for (column, twiddle_chunk) in (1..self.height/2 + 1).zip(self.twiddles.chunks_exact(3)) {
            // load elements from each of our 4 rows, both from the front and the back of the array
            let input_fwd = [
                split_buffer[0][column],
                split_buffer[1][column],
                split_buffer[2][column],
                split_buffer[3][column]
            ];
            let input_rev = [
                split_buffer[0][self.height - column],
                split_buffer[1][self.height - column],
                split_buffer[2][self.height - column],
                split_buffer[3][self.height - column]
            ];

            // Apply our twiddle factors
            let mut out0_fwd = input_fwd[0];
            let mut out1_fwd = input_fwd[1] * twiddle_chunk[0].re + input_rev[1] * twiddle_chunk[0].im;
            let mut out2_fwd = input_fwd[2] * twiddle_chunk[1].re + input_rev[2] * twiddle_chunk[1].im;
            let mut out3_fwd = input_fwd[3] * twiddle_chunk[2].im - input_rev[3] * twiddle_chunk[2].re;

            let mut out0_rev = input_rev[0];
            let mut out1_rev = input_fwd[3] * twiddle_chunk[2].re + input_rev[3] * twiddle_chunk[2].im;
            let mut out2_rev = input_fwd[2] * twiddle_chunk[1].im - input_rev[2] * twiddle_chunk[1].re;
            let mut out3_rev = input_rev[1] * twiddle_chunk[0].re - input_fwd[1] * twiddle_chunk[0].im;
            
            // do a giant pile of butterfy 2's
            // most of these come from the 2 size-4 DHTs (one for fwd, one for rev) we would have done after the twiddle factrs were applied.
            // In the process of removing redundant operations, we unfortunately need to unravel the butterfly 4's, leaving us with this mess.
            Butterfly2::perform_dht_strided(&mut out1_fwd, &mut out1_rev);
            Butterfly2::perform_dht_strided(&mut out3_fwd, &mut out3_rev);
            Butterfly2::perform_dht_strided(&mut out0_fwd, &mut out2_fwd);
            Butterfly2::perform_dht_strided(&mut out0_rev, &mut out2_rev);
            Butterfly2::perform_dht_strided(&mut out0_fwd, &mut out1_fwd);
            Butterfly2::perform_dht_strided(&mut out0_rev, &mut out1_rev);
            Butterfly2::perform_dht_strided(&mut out2_fwd, &mut out3_fwd);
            Butterfly2::perform_dht_strided(&mut out2_rev, &mut out3_rev);
            
            // The last step in the butterfly 4 would have been to do a 2x2 transpose, so we have to do that here
            let post_dht_fwd = [out0_fwd, out2_fwd, out1_fwd, out3_fwd];
            let post_dht_rev = [out0_rev, out2_rev, out1_rev, out3_rev];

            for i in 0..4 {
                split_buffer[i][column] = post_dht_fwd[i];
                split_buffer[i][self.height - column] = post_dht_rev[i];
            }
        }
    }

    fn perform_dht_inplace(&self, buffer: &mut [T], _scratch: &mut [T]) {
        let mut scratch = vec![Zero::zero(); buffer.len()];

        // Step 1: Transpose the width x height array to height x width
        transpose::transpose(buffer, &mut scratch, 4, self.height);

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
        transpose::transpose(input, output, 4, self.height);

        // Step 2: Compute DHTs of size `height` down the rows of our transposed array
        self.height_size_fft.process_with_scratch(output, input);

        // Step 3: Apply twiddle factors
        self.apply_twiddles(output);
    }
}
boilerplate_dht!(MixedRadix4xn);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_dht_algorithm;
    use crate::scalar::DhtNaive;
    use std::sync::Arc;

    #[test]
    fn test_mixed_radix_4xn_correct() {
        for height in 1..7 {
                test_mixed_radix_4xn_with_lengths(height);
        }
    }

    fn test_mixed_radix_4xn_with_lengths(height: usize) {
        let height_dht = Arc::new(DhtNaive::new(height)) as Arc<dyn Dht<f32>>;

        let dht = MixedRadix4xn::new(height_dht);

        check_dht_algorithm(&dht, height * 4);
    }
}