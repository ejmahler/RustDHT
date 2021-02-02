use std::sync::Arc;

use num_complex::Complex;
use rustfft::{FftNum, Length};

use crate::{Dht, array_utils, dht_error_inplace, dht_error_outofplace, twiddles};

pub struct SplitRadix<T> {
    twiddles: Box<[Complex<T>]>,

    quarter_dht: Arc<dyn Dht<T>>,
    half_dht: Arc<dyn Dht<T>>,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    len: usize,
}
boilerplate_dht!(SplitRadix);
impl<T: FftNum> SplitRadix<T> {
    /// Creates a DHT instance which will process inputs/outputs of size `height_dht.len() * 4`
    pub fn new(quarter_dht: Arc<dyn Dht<T>>, half_dht: Arc<dyn Dht<T>>) -> Self {
        let quarter_len = quarter_dht.len();
        let half_len = half_dht.len();

        assert_eq!(half_len, quarter_len * 2);
        let len = quarter_len * 4;

        let twiddle_limit = len / 8 + 1;

        let mut twiddles = Vec::with_capacity(twiddle_limit * 2);
        for x in 0..twiddle_limit {
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(x * 1, len));
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(x * 3, len));
        }

        Self {
            twiddles: twiddles.into_boxed_slice(),

            quarter_dht,
            half_dht,

            inplace_scratch_len: half_len,
            outofplace_scratch_len: 0,
            len,
        }
    }

    fn apply_twiddles(&self, buffer: &mut [T]) {
        let half_len = self.len() / 2;
        let quarter_len = self.len() / 4;
        let eigth_len = self.len() / 8;

        let (buffer_evens, buffer_odds) = buffer.split_at_mut(half_len);
        let (buffer_half0, buffer_half1) = buffer_evens.split_at_mut(quarter_len);
        let (buffer_quarter1, buffer_quarter3) = buffer_odds.split_at_mut(quarter_len);

        // i == 0 needs to be handled slightly differently, both because we can save some twiddle factor work, and because our indexing algorithm breaks down
        {
            let input_q0_fwd = buffer_half0[0];
            let input_q1_fwd = buffer_half1[0];
            let input_q2_fwd = buffer_quarter1[0];
            let input_q3_fwd = buffer_quarter3[0];

            let diff0_fwd = input_q0_fwd - input_q2_fwd;
            let diff0_rev = input_q1_fwd - input_q3_fwd;

            buffer_half0[0] = input_q0_fwd + input_q2_fwd;
            buffer_half1[0] = input_q1_fwd + input_q3_fwd;
            buffer_quarter1[0] = diff0_fwd + diff0_rev;
            buffer_quarter3[0] = diff0_fwd - diff0_rev;
        }

        for i in 1..eigth_len + 1 {
            let twiddle1_fwd = unsafe { *self.twiddles.get_unchecked(2*i) };
            let twiddle3_fwd = unsafe { *self.twiddles.get_unchecked(2*i + 1) };

            let input_fwd = unsafe { [
                *buffer_half0.get_unchecked(i),
                *buffer_half1.get_unchecked(i),
                *buffer_quarter1.get_unchecked(i),
                *buffer_quarter3.get_unchecked(i),
            ] };

            let input_rev = unsafe { [
                *buffer_half0.get_unchecked(quarter_len - i),
                *buffer_half1.get_unchecked(quarter_len - i),
                *buffer_quarter1.get_unchecked(quarter_len - i),
                *buffer_quarter3.get_unchecked(quarter_len - i),
            ] };

            let diff0_fwd = input_fwd[0] - input_fwd[2];
            let diff0_rev = input_rev[0] - input_rev[2];

            let diff1_fwd = input_fwd[1] - input_fwd[3];
            let diff1_rev = input_rev[1] - input_rev[3];

            let diff0_sum = diff0_fwd + diff0_rev;
            let diff0_dif = diff0_fwd - diff0_rev;
            
            let diff1_sum = diff1_fwd + diff1_rev;
            let diff1_dif = diff1_fwd - diff1_rev;

            unsafe { *buffer_half0.get_unchecked_mut(i) = input_fwd[0] + input_fwd[2] };
            unsafe { *buffer_half1.get_unchecked_mut(i) = input_fwd[1] + input_fwd[3] };
            unsafe { *buffer_half0.get_unchecked_mut(quarter_len - i) = input_rev[0] + input_rev[2] };
            unsafe { *buffer_half1.get_unchecked_mut(quarter_len - i) = input_rev[1] + input_rev[3] };

            unsafe { *buffer_quarter1.get_unchecked_mut(i) = diff0_sum * twiddle1_fwd.re - diff1_dif * twiddle1_fwd.im };
            unsafe { *buffer_quarter3.get_unchecked_mut(i) = diff0_dif * twiddle3_fwd.re + diff1_sum * twiddle3_fwd.im };
            unsafe { *buffer_quarter1.get_unchecked_mut(quarter_len - i) = diff0_sum * twiddle1_fwd.im + diff1_dif * twiddle1_fwd.re };
            unsafe { *buffer_quarter3.get_unchecked_mut(quarter_len - i) = diff0_dif * twiddle3_fwd.im - diff1_sum * twiddle3_fwd.re };
        }
    }

    fn perform_dht_inplace(&self, buffer: &mut [T], scratch: &mut [T]) {
        self.apply_twiddles(buffer);
        
        let half_len = self.len() / 2;
        let quarter_len = self.len() / 4;

        let (buffer_evens, buffer_odds) = buffer.split_at_mut(half_len);

        self.half_dht.process_with_scratch(buffer_evens, scratch);
        self.quarter_dht.process_outofplace_with_scratch(buffer_odds, scratch, &mut []);
        
        let (scratch_quarter1, scratch_quarter3) = scratch.split_at_mut(quarter_len);

        for i in (0..quarter_len).rev() {
            unsafe { *buffer.get_unchecked_mut(4*i + 0) = *buffer.get_unchecked(i*2) };
            unsafe { *buffer.get_unchecked_mut(4*i + 2) = *buffer.get_unchecked(i*2 + 1) };
            unsafe { *buffer.get_unchecked_mut(4*i + 1) = *scratch_quarter1.get_unchecked(i) };
            unsafe { *buffer.get_unchecked_mut(4*i + 3) = *scratch_quarter3.get_unchecked(i) };
        }
    }

    fn perform_dht_out_of_place(
        &self,
        input: &mut [T],
        output: &mut [T],
        _scratch: &mut [T],
    ) {
        self.apply_twiddles(input);
        
        let half_len = self.len() / 2;
        let quarter_len = self.len() / 4;

        let (buffer_evens, buffer_odds) = input.split_at_mut(half_len);
        
        self.half_dht.process_with_scratch(buffer_evens, output);
        self.quarter_dht.process_with_scratch(buffer_odds, output);

        let (buffer_quarter1, buffer_quarter3) = buffer_odds.split_at_mut(quarter_len);

        for (i, output_chunk) in output.chunks_exact_mut(4).enumerate() {
            unsafe { *output_chunk.get_unchecked_mut(0) = *buffer_evens.get_unchecked(i*2) };
            unsafe { *output_chunk.get_unchecked_mut(1) = *buffer_quarter1.get_unchecked(i) };
            unsafe { *output_chunk.get_unchecked_mut(2) = *buffer_evens.get_unchecked(i*2 + 1) };
            unsafe { *output_chunk.get_unchecked_mut(3) = *buffer_quarter3.get_unchecked(i) };
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_dht_algorithm;
    use crate::scalar::DhtNaive;
    use std::sync::Arc;

    #[test]
    fn test_split_radix_correct() {
        for quarter_len in 1..30 {
            test_split_radix_with_quarter_len(quarter_len);
        }
    }

    fn test_split_radix_with_quarter_len(quarter_len: usize) {
        let quarter_dht = Arc::new(DhtNaive::new(quarter_len)) as Arc<dyn Dht<f32>>;
        let half_dht = Arc::new(DhtNaive::new(quarter_len * 2)) as Arc<dyn Dht<f32>>;

        let dht = SplitRadix::new(quarter_dht, half_dht);

        check_dht_algorithm(&dht, quarter_len * 4);
    }
}
