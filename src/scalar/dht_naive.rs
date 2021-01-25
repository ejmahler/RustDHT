use num_traits::Zero;
use rustfft::{FftNum, Length};

use crate::{Dht, twiddles, array_utils, dht_error_inplace, dht_error_outofplace};



/// Naive O(n^2 ) Discrete Hartley Transform implementation
///
/// This implementation is primarily used to test other DHT algorithms.
///
/// ~~~
/// // Computes a naive DHT of size 123
/// use rustdht::{Dht, scalar::DhtNaive};
///
/// let mut buffer = vec![0.0f32; 123];
///
/// let dht = DhtNaive::new(123);
/// dht.process(&mut buffer);
/// ~~~
pub struct DhtNaive<T> {
    twiddles: Vec<T>,
}

impl<T: FftNum> DhtNaive<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute DhtNaive
    pub fn new(len: usize) -> Self {
        let twiddles = (0..len)
            .map(|i| twiddles::compute_dht_twiddle(i, len))
            .collect();
        Self {
            twiddles,
        }
    }

    fn perform_dht_out_of_place(
        &self,
        input: &[T],
        output: &mut [T],
        _scratch: &mut [T],
    ) {
        for k in 0..output.len() {
            let mut output_value = Zero::zero();
            let mut twiddle_index = 0;

            for input_cell in input {
                let twiddle = self.twiddles[twiddle_index];
                output_value = output_value + twiddle * *input_cell;

                twiddle_index += k;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }

            output[k] = output_value;
        }
    }
}
boilerplate_dht_oop!(DhtNaive, |this: &DhtNaive<_>| this.twiddles.len());

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::{compare_real_vectors, random_real_signal};
    use num_traits::Zero;
    use std::f32;

    fn dht(signal: &[f32], spectrum: &mut [f32]) {
        for (k, spec_bin) in spectrum.iter_mut().enumerate() {
            let mut sum = Zero::zero();
            for (i, &x) in signal.iter().enumerate() {
                let angle = 2f32 * (i * k) as f32 * f32::consts::PI / signal.len() as f32;
                let twiddle = angle.cos() + angle.sin();

                sum = sum + twiddle * x;
            }
            *spec_bin = sum;
        }
    }

    #[test]
    fn test_matches_dht() {
        let n = 4;

        for len in 1..20 {
            let dht_instance = DhtNaive::new(len);
            assert_eq!(
                dht_instance.len(),
                len,
                "DhtNaive instance reported incorrect length"
            );

            let input = random_real_signal(len * n);
            let mut expected_output = input.clone();

            // Compute the control data using our simplified DhtNaive definition
            for (input_chunk, output_chunk) in
                input.chunks(len).zip(expected_output.chunks_mut(len))
            {
                dht(input_chunk, output_chunk);
            }

            // test process()
            {
                let mut inplace_buffer = input.clone();

                dht_instance.process(&mut inplace_buffer);

                assert!(
                    compare_real_vectors(&expected_output, &inplace_buffer),
                    "process() failed, length = {}",
                    len
                );
            }

            // test process_with_scratch()
            {
                let mut inplace_with_scratch_buffer = input.clone();
                let mut inplace_scratch =
                    vec![Zero::zero(); dht_instance.get_inplace_scratch_len()];

                dht_instance
                    .process_with_scratch(&mut inplace_with_scratch_buffer, &mut inplace_scratch);

                assert!(
                    compare_real_vectors(&expected_output, &inplace_with_scratch_buffer),
                    "process_inplace() failed, length = {}",
                    len
                );

                // one more thing: make sure that the DhtNaive algorithm even works with dirty scratch space
                for item in inplace_scratch.iter_mut() {
                    *item = 100.0;
                }
                inplace_with_scratch_buffer.copy_from_slice(&input);

                dht_instance
                    .process_with_scratch(&mut inplace_with_scratch_buffer, &mut inplace_scratch);

                assert!(
                    compare_real_vectors(&expected_output, &inplace_with_scratch_buffer),
                    "process_with_scratch() failed the 'dirty scratch' test for len = {}",
                    len
                );
            }

            // test process_outofplace_with_scratch
            {
                let mut outofplace_input = input.clone();
                let mut outofplace_output = expected_output.clone();

                dht_instance.process_outofplace_with_scratch(
                    &mut outofplace_input,
                    &mut outofplace_output,
                    &mut [],
                );

                assert!(
                    compare_real_vectors(&expected_output, &outofplace_output),
                    "process_outofplace_with_scratch() failed, length = {}",
                    len
                );
            }
        }

        //verify that it doesn't crash or infinite loop if we have a length of 0
        let zero_dht = DhtNaive::new(0);
        let mut zero_input: Vec<f32> = Vec::new();
        let mut zero_output: Vec<f32> = Vec::new();
        let mut zero_scratch: Vec<f32> = Vec::new();

        zero_dht.process(&mut zero_input);
        zero_dht.process_with_scratch(&mut zero_input, &mut zero_scratch);
        zero_dht.process_outofplace_with_scratch(
            &mut zero_input,
            &mut zero_output,
            &mut zero_scratch,
        );
    }
}