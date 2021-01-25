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
    twiddles_width: Box<[Complex<T>]>,
    twiddles_height: Box<[Complex<T>]>,

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

        let width_limit = width / 2 + 1;
        let height_limit = height / 2 + 1;

        let mut twiddles = Vec::with_capacity(width_limit * height_limit);
        for x in 0..width_limit {
            for y in 0..height_limit {
                twiddles.push(twiddles::compute_dft_twiddle_inverse(x * y, len));
            }
        }

        let twiddles_width = (0..width_limit).map(|i| twiddles::compute_dft_twiddle_inverse(i, width)).collect::<Box<[_]>>();
        let twiddles_height = (0..height_limit).map(|k| twiddles::compute_dft_twiddle_inverse(k, height)).collect::<Box<[_]>>();

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
            twiddles_width,
            twiddles_height,

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
        let width_limit = (self.width + 1) / 2;
        let height_limit = (self.height + 1) / 2;
        let twiddle_stride = self.twiddles_height.len();

        for i in 1..width_limit {
            let i_bot = self.width - i;

            let twiddle_i = self.twiddles_width[i];
            
            for k in 1..height_limit {
                let k_rev = self.height - k;

                let twiddle_ik = self.twiddles[i * twiddle_stride + k];
                let twiddle_k = self.twiddles_height[k];

                let twiddle_top_fwd = twiddle_ik;
                let twiddle_top_rev = twiddle_i * twiddle_ik.conj();
                let twiddle_bot_fwd = twiddle_k * twiddle_ik.conj();
                let twiddle_bot_rev = twiddle_ik * twiddle_i.conj() * twiddle_k.conj();

                // Instead of just multiplying a single input vlaue with a single complex number like we do in the DFT,
                // we need to combine 4 numbers, determined by mirroring the input number across the horizontal and vertical axes of the array
                let input_top_fwd = buffer[i*self.height + k];
                let input_bot_fwd = buffer[i_bot*self.height + k];
                let input_top_rev = buffer[i*self.height + k_rev];
                let input_bot_rev = buffer[i_bot*self.height + k_rev];
    
                // Since we're overwriting data that our mirrored input values will need whenthey compute their own twiddles,
                // we currently can't apply twiddles inplace. An obvious optimization here is to compute all 4 values at once and write them all out at once.
                // That would cut down on the number of flops by 75%, and would let us do this inplace
                buffer[i*self.height + k] = T::from_f32(0.5).unwrap() * (
                    input_top_fwd      * twiddle_top_fwd.re
                    - input_top_fwd    * twiddle_top_fwd.im
                    + input_top_rev    * twiddle_top_fwd.re
                    + input_top_rev    * twiddle_top_fwd.im
                    + input_bot_fwd    * twiddle_bot_fwd.re
                    + input_bot_fwd    * twiddle_bot_fwd.im
                    - input_bot_rev    * twiddle_bot_fwd.re
                    + input_bot_rev    * twiddle_bot_fwd.im
                );
                    
                buffer[i*self.height + k_rev] = T::from_f32(0.5).unwrap() * (
                    input_top_rev     * twiddle_top_rev.re
                    - input_top_rev   * twiddle_top_rev.im
                    + input_top_fwd   * twiddle_top_rev.re
                    + input_top_fwd   * twiddle_top_rev.im
                    + input_bot_rev   * twiddle_bot_rev.re
                    + input_bot_rev   * twiddle_bot_rev.im
                    - input_bot_fwd   * twiddle_bot_rev.re
                    + input_bot_fwd   * twiddle_bot_rev.im
                );

                buffer[i_bot*self.height + k] = T::from_f32(0.5).unwrap() * (
                    input_bot_fwd     * twiddle_bot_fwd.re
                    - input_bot_fwd   * twiddle_bot_fwd.im
                    + input_bot_rev   * twiddle_bot_fwd.re
                    + input_bot_rev   * twiddle_bot_fwd.im
                    + input_top_fwd   * twiddle_top_fwd.re
                    + input_top_fwd   * twiddle_top_fwd.im
                    - input_top_rev   * twiddle_top_fwd.re
                    + input_top_rev   * twiddle_top_fwd.im
                );
                    
                buffer[i_bot*self.height + k_rev] = T::from_f32(0.5).unwrap() * (
                    input_bot_rev     * twiddle_bot_rev.re
                    - input_bot_rev   * twiddle_bot_rev.im
                    + input_bot_fwd   * twiddle_bot_rev.re
                    + input_bot_fwd   * twiddle_bot_rev.im
                    + input_top_rev   * twiddle_top_rev.re
                    + input_top_rev   * twiddle_top_rev.im
                    - input_top_fwd   * twiddle_top_rev.re
                    + input_top_fwd   * twiddle_top_rev.im
                );
            }
            
            if self.height % 2 == 0 {
                let k = self.height / 2;

                let twiddle_ik = self.twiddles[i * twiddle_stride + k];
                let twiddle_k = self.twiddles_height[k];

                let twiddle_top_fwd = twiddle_ik;
                let twiddle_bot_fwd = twiddle_k * twiddle_ik.conj();

                // Instead of just multiplying a single input vlaue with a single complex number like we do in the DFT,
                // we need to combine 4 numbers, determined by mirroring the input number across the horizontal and vertical axes of the array
                let input_top_fwd = buffer[i*self.height + k];
                let input_bot_fwd = buffer[i_bot*self.height + k];
    
                // Since we're overwriting data that our mirrored input values will need whenthey compute their own twiddles,
                // we currently can't apply twiddles inplace. An obvious optimization here is to compute all 4 values at once and write them all out at once.
                // That would cut down on the number of flops by 75%, and would let us do this inplace
                buffer[i*self.height + k] = T::from_f32(0.5).unwrap() * (
                    input_top_fwd      * twiddle_top_fwd.re
                    + input_top_fwd    * twiddle_top_fwd.re
                    + input_bot_fwd    * twiddle_bot_fwd.im
                    + input_bot_fwd    * twiddle_bot_fwd.im
                );

                buffer[i_bot*self.height + k] = T::from_f32(0.5).unwrap() * (
                    input_bot_fwd     * twiddle_bot_fwd.re
                    + input_bot_fwd   * twiddle_bot_fwd.re
                    + input_top_fwd   * twiddle_top_fwd.im
                    + input_top_fwd   * twiddle_top_fwd.im
                );
            }
        }

        if self.width % 2 == 0 {
            let i = self.width / 2;

            let twiddle_i = self.twiddles_width[i];

            for k in 1..height_limit {
                let k_rev = self.height - k;

                let twiddle_ik = self.twiddles[i * twiddle_stride + k];

                let twiddle_top_fwd = twiddle_ik;
                let twiddle_top_rev = twiddle_i * twiddle_ik.conj();

                // Instead of just multiplying a single input vlaue with a single complex number like we do in the DFT,
                // we need to combine 4 numbers, determined by mirroring the input number across the horizontal and vertical axes of the array
                let input_top_fwd = buffer[i*self.height + k];
                let input_top_rev = buffer[i*self.height + k_rev];
    
                // Since we're overwriting data that our mirrored input values will need whenthey compute their own twiddles,
                // we currently can't apply twiddles inplace. An obvious optimization here is to compute all 4 values at once and write them all out at once.
                // That would cut down on the number of flops by 75%, and would let us do this inplace
                buffer[i*self.height + k] = T::from_f32(0.5).unwrap() * (
                    input_top_fwd      * twiddle_top_fwd.re
                    + input_top_rev    * twiddle_top_fwd.im
                    + input_top_fwd    * twiddle_top_fwd.re
                    + input_top_rev    * twiddle_top_fwd.im
                );
                    
                buffer[i*self.height + k_rev] = T::from_f32(0.5).unwrap() * (
                    input_top_rev     * twiddle_top_rev.re
                    + input_top_fwd   * twiddle_top_rev.im
                    + input_top_rev   * twiddle_top_rev.re
                    + input_top_fwd   * twiddle_top_rev.im
                );
            }

            if self.height % 2 == 0 {
                let k = self.height / 2;

                let twiddle_ik = self.twiddles[i * twiddle_stride + k];

                // Instead of just multiplying a single input vlaue with a single complex number like we do in the DFT,
                // we need to combine 4 numbers, determined by mirroring the input number across the horizontal and vertical axes of the array
                let input_top_fwd = buffer[i*self.height + k];
    
                // Since we're overwriting data that our mirrored input values will need whenthey compute their own twiddles,
                // we currently can't apply twiddles inplace. An obvious optimization here is to compute all 4 values at once and write them all out at once.
                // That would cut down on the number of flops by 75%, and would let us do this inplace
                buffer[i*self.height + k] = T::from_f32(0.5).unwrap() * (
                    input_top_fwd      * twiddle_ik.re
                    + input_top_fwd    * twiddle_ik.im
                    + input_top_fwd    * twiddle_ik.re
                    + input_top_fwd    * twiddle_ik.im
                );
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
        self.height_size_fft
            .process_with_scratch(scratch, height_scratch);

        // STEP 3: Apply twiddle factors
        self.apply_twiddles(scratch);

        // STEP 4: transpose again
        transpose::transpose(scratch, buffer, self.height, self.width);

        // STEP 5: perform DHTs of size `width`
        self.width_size_fft
            .process_outofplace_with_scratch(buffer, scratch, inner_scratch);

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
    fn test_mixed_radix() {
        for width in 1..7 {
            for height in 1..7 {
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