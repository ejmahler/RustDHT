use std::sync::Arc;

use num_traits::Zero;
use num_integer::Integer;

use primal_check::miller_rabin;
use rustfft::{FftNum, Length};
use strength_reduce::StrengthReducedUsize;

use crate::{Dht, array_utils, dht_error_inplace, dht_error_outofplace, math_utils, twiddles};

pub struct RadersAlgorithm<T> {
    inner_dht: Arc<dyn Dht<T>>,
    twiddles: Box<[T]>,

    primitive_root: usize,
    primitive_root_inverse: usize,

    reduced_len: StrengthReducedUsize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    len: usize,
}
boilerplate_dht!(RadersAlgorithm);
impl<T: FftNum> RadersAlgorithm<T> {
    /// Creates a DHT instance which will process inputs/outputs of size `inner_dht.len() + 1`
    pub fn new(inner_dht: Arc<dyn Dht<T>>) -> Self {
        let inner_dht_len = inner_dht.len();
        let len = inner_dht_len + 1;
        assert!(miller_rabin(len as u64), "For raders algorithm, inner_dht.len() + 1 must be prime. Expected prime number, got {} + 1 = {}", inner_dht_len, len);

        let reduced_len = StrengthReducedUsize::new(len);

        // compute the primitive root and its inverse for this size
        let primitive_root = math_utils::primitive_root(len as u64).unwrap() as usize;

        // compute the multiplicative inverse of primative_root mod len and vice versa.
        // i64::extended_gcd will compute both the inverse of left mod right, and the inverse of right mod left, but we're only goingto use one of them
        // the primtive root inverse might be negative, if o make it positive by wrapping
        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(len as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + len as i64
        } as usize;

        // precompute the coefficients to use inside the process method.
        let normalization_scale = T::from_f64(1f64 / inner_dht_len as f64).unwrap();
        let mut inner_dht_input = vec![Zero::zero(); inner_dht_len];
        let mut twiddle_input = 1;
        for input_cell in &mut inner_dht_input {
            let twiddle = twiddles::compute_dht_twiddle::<T>(twiddle_input, len);
            *input_cell = twiddle * normalization_scale;

            twiddle_input = (twiddle_input * primitive_root_inverse) % reduced_len;
        }

        //precompute a FFT of our reordered twiddle factors
        inner_dht.process(&mut inner_dht_input);
        

        let required_inner_scratch = inner_dht.get_inplace_scratch_len();
        let extra_inner_scratch = if required_inner_scratch <= inner_dht_len {
            0
        } else {
            required_inner_scratch
        };

        Self {
            inner_dht,
            twiddles: inner_dht_input.into_boxed_slice(),

            primitive_root,
            primitive_root_inverse,

            reduced_len,

            len,
            inplace_scratch_len: inner_dht_len + extra_inner_scratch,
            outofplace_scratch_len: extra_inner_scratch,
        }
    }

    fn perform_dht_inplace(&self, buffer: &mut [T], scratch: &mut [T]) {
        // The first output element is just the sum of all the input elements, and we need to store off the first input value
        let (buffer_first, buffer) = buffer.split_first_mut().unwrap();
        let buffer_first_val = *buffer_first;

        let (inner_dht_buffer, extra_scratch) = scratch.split_at_mut(self.len() - 1);

        // copy the buffer into the scratch, reordering as we go. also compute a sum of all elements
        let mut input_index = 1;
        for scratch_element in inner_dht_buffer.iter_mut() {
            input_index = (input_index * self.primitive_root) % self.reduced_len;

            let buffer_element = buffer[input_index - 1];
            *scratch_element = buffer_element;
        }

        // perform the first of two inner FFTs
        let inner_scratch = if extra_scratch.len() > 0 {
            extra_scratch
        } else {
            &mut buffer[..]
        };
        self.inner_dht.process_with_scratch(inner_dht_buffer, inner_scratch);

        // inner_dht_buffer[0] now contains the sum of elements 1..len. We need the sum of all elements, so all we have to do is add the first input
        *buffer_first = *buffer_first + inner_dht_buffer[0];

        // Multiply the result of our inner DHT with our precomputed twiddle factors.
        inner_dht_buffer[0] = inner_dht_buffer[0] * self.twiddles[0];
        dbg!(&self.twiddles);
        for i in 1..inner_dht_buffer.len() / 2 {
            let i_rev = inner_dht_buffer.len() - i;

            let twiddle_fwd = self.twiddles[i];
            let twiddle_rev = self.twiddles[i_rev];

            let input_fwd = inner_dht_buffer[i];
            let input_rev = inner_dht_buffer[i_rev];

            println!();
            dbg!(i, i_rev);
            dbg!(twiddle_fwd, twiddle_rev);
            dbg!(input_fwd, input_rev);

            let output_fwd = twiddle_fwd * input_fwd - twiddle_rev * input_rev;
            let output_rev = twiddle_fwd * input_rev + twiddle_rev * input_fwd;

            inner_dht_buffer[i] = output_fwd;
            inner_dht_buffer[i_rev] = output_rev;
        }
        // If our length is even, we have to process the middle-most element separately
        if inner_dht_buffer.len() % 2 == 0 {
            let mid_index = inner_dht_buffer.len() / 2;
            inner_dht_buffer[mid_index] = inner_dht_buffer[mid_index] * self.twiddles[mid_index];
        }

        // We need to add the first input value to all output values. We can accomplish this by adding it to the DC input of our inner dht.
        inner_dht_buffer[0] = inner_dht_buffer[0] + buffer_first_val;

        // execute the second FFT
        self.inner_dht.process_with_scratch(inner_dht_buffer, inner_scratch);

        // copy the final values into the output, reordering as we go
        let mut output_index = 1;
        for inner_element in inner_dht_buffer {
            output_index = (output_index * self.primitive_root_inverse) % self.reduced_len;
            buffer[output_index - 1] = *inner_element;
        }
    }

    fn perform_dht_out_of_place(
        &self,
        input: &mut [T],
        output: &mut [T],
        scratch: &mut [T],
    ) {
        // The first output element is just the sum of all the input elements, and we need to store off the first input value
        let (output_first, output) = output.split_first_mut().unwrap();
        let (input_first, input) = input.split_first_mut().unwrap();

        // copy the inout into the output, reordering as we go. also compute a sum of all elements
        let mut input_index = 1;
        for output_element in output.iter_mut() {
            input_index = (input_index * self.primitive_root) % self.reduced_len;

            let input_element = input[input_index - 1];
            *output_element = input_element;
        }

        // perform the first of two inner DHTs
        let inner_scratch = if scratch.len() > 0 {
            &mut scratch[..]
        } else {
            &mut input[..]
        };
        self.inner_dht.process_with_scratch(output, inner_scratch);

        // output[0] now contains the sum of elements 1..len. We need the sum of all elements, so all we have to do is add the first input
        *output_first = *input_first + output[0];

        // Multiply the result of our inner DHT with our precomputed twiddle factors. Also copy to the input buffer.
        input[0] = output[0] * self.twiddles[0];
    
        for i in 1..output.len() / 2 {
            let i_rev = output.len() - i;

            let twiddle_fwd = self.twiddles[i];
            let twiddle_rev = self.twiddles[i_rev];

            let input_fwd = output[i];
            let input_rev = output[i_rev];

            let output_fwd = twiddle_fwd * input_fwd - twiddle_rev * input_rev;
            let output_rev = twiddle_fwd * input_rev + twiddle_rev * input_fwd;

            input[i] = output_fwd;
            input[i_rev] = output_rev;
        }
        // If our length is even, we have to process the middle-most element separately
        if output.len() % 2 == 0 {
            let mid_index = output.len() / 2;
            input[mid_index] = output[mid_index] * self.twiddles[mid_index];
        }
            
        // We need to add the first input value to all output values. We can accomplish this by adding it to the DC input of our inner dht.
        input[0] = input[0] + *input_first;

        // execute the second DHT
        let inner_scratch = if scratch.len() > 0 {
            scratch
        } else {
            &mut output[..]
        };
        self.inner_dht.process_with_scratch(input, inner_scratch);

        // copy the final values into the output, reordering as we go
        let mut output_index = 1;
        for input_element in input {
            output_index = (output_index * self.primitive_root_inverse) % self.reduced_len;
            output[output_index - 1] = *input_element;
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::scalar::DhtNaive;
    use crate::test_utils::check_dht_algorithm;

    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_raders_correct() {
        for len in 3..100 {
            if miller_rabin(len as u64) {
                test_raders_with_length(len);
            }
        }
    }

    fn test_raders_with_length(len: usize) {
        let inner_fft = Arc::new(DhtNaive::new(len - 1));
        let fft = RadersAlgorithm::new(inner_fft);

        check_dht_algorithm::<f32>(&fft, len);
    }
}