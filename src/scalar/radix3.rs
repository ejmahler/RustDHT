use std::sync::Arc;

use num_complex::Complex;
use rustfft::{FftNum, Length};

use crate::{Dht, array_utils, dht_error_inplace, dht_error_outofplace, twiddles};

use super::Butterfly3;

pub struct MixedRadix3xn<T> {
    twiddles: Box<[Complex<T>]>,

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

        let root2 = T::from_f32(0.5f32.sqrt()).unwrap();
        let mut twiddles = Vec::with_capacity((twiddle_limit - 1) * 2);
        for column in 1..twiddle_limit {
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(column * 1 * 8 + len, len * 8) * root2);
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
        
        // Step 3: Apply twiddle factors
        for (column, twiddle) in (1..self.height/2+1).zip(self.twiddles.iter()) {
            let input1fwd = split_buffer[1][column];
            let input1rev = split_buffer[1][self.height - column];
            let input2fwd = split_buffer[2][column];
            let input2rev = split_buffer[2][self.height - column];
            
            let out1fwd = input1fwd + input2rev;
            let out1rev = input2fwd + input1rev;
            let out2fwd = input2fwd - input1rev;
            let out2rev = input1fwd - input2rev;
            
            let mut tmp_fwd = [
                split_buffer[0][column],
                twiddle.re * out1fwd - twiddle.im * out2fwd,
                twiddle.re * out2fwd + twiddle.im * out1fwd,
            ];
            let mut tmp_rev = [
                split_buffer[0][self.height - column],
                twiddle.re * out1rev - twiddle.im * out2rev,
                twiddle.re * out2rev + twiddle.im * out1rev,
            ];
            
            self.butterfly3.perform_dht_array(&mut tmp_fwd);
            
            split_buffer[0][column] = tmp_fwd[0];
            split_buffer[1][column] = tmp_fwd[1];
            split_buffer[2][column] = tmp_fwd[2];
            
            self.butterfly3.perform_dht_array(&mut tmp_rev);

            split_buffer[0][self.height - column] = tmp_rev[2];
            split_buffer[1][self.height - column] = tmp_rev[1];
            split_buffer[2][self.height - column] = tmp_rev[0];
        }
    }

    fn perform_dht_inplace(&self, buffer: &mut [T], scratch: &mut [T]) {
        // Step 1: Transpose the width x height array to height x width
        array_utils::transpose_half_rev_out(buffer, scratch, 3, self.height);

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
        array_utils::transpose_half_rev_out(input, output, 3, self.height);

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