use std::sync::Arc;

use num_complex::Complex;
use rustfft::{FftNum, Length};

use crate::{Dht, array_utils, dht_error_inplace, dht_error_outofplace, twiddles};

use super::{Butterfly1, Butterfly16, Butterfly2, Butterfly4, Butterfly8};

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

        let mut twiddles = Vec::with_capacity(twiddle_limit * 2);
        for column in 1..twiddle_limit {
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(1 * column, len));
            twiddles.push(twiddles::compute_dft_twiddle_inverse::<T>(2 * column, len));
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

        
        for (column, twiddle_chunk) in (1..self.height/2 + 1).zip(self.twiddles.chunks_exact(2)) {
            let input0fwd = split_buffer[0][column];
            let input0rev = split_buffer[0][self.height - column];
            let input1fwd = split_buffer[1][column];
            let input1rev = split_buffer[1][self.height - column];
            let input2fwd = split_buffer[2][column];
            let input2rev = split_buffer[2][self.height - column];
            let input3fwd = split_buffer[3][column];
            let input3rev = split_buffer[3][self.height - column];   
            
            let out1fwd = input1fwd + input3rev;
            let out3fwd = input1rev - input3fwd;
            let out1rev = input3fwd + input1rev;
            let out3rev = input3rev - input1fwd;
            
            let mut fwd0: T = input0fwd;
            let mut rev0: T = input0rev;
            let mut fwd2: T = twiddle_chunk[1].re * input2fwd + twiddle_chunk[1].im * input2rev;
            let mut rev2: T = twiddle_chunk[1].re * input2rev - twiddle_chunk[1].im * input2fwd;

            let mut fwd1: T = twiddle_chunk[0].re * out1fwd   + twiddle_chunk[0].im * out3fwd;
            let mut rev1: T = twiddle_chunk[0].re * out1rev   + twiddle_chunk[0].im * out3rev;
            let mut fwd3: T = twiddle_chunk[0].re * out3fwd   - twiddle_chunk[0].im * out1fwd;
            let mut rev3: T = twiddle_chunk[0].re * out3rev   - twiddle_chunk[0].im * out1rev;

            Butterfly2::perform_dht_strided(&mut fwd0, &mut fwd2);
            Butterfly2::perform_dht_strided(&mut fwd0, &mut fwd1);
            Butterfly2::perform_dht_strided(&mut fwd2, &mut fwd3);
            Butterfly2::perform_dht_strided(&mut rev0, &mut rev2);
            Butterfly2::perform_dht_strided(&mut rev0, &mut rev1);
            Butterfly2::perform_dht_strided(&mut rev2, &mut rev3);
            
            split_buffer[0][column] = fwd0;
            split_buffer[1][column] = fwd2;
            split_buffer[2][column] = fwd1;
            split_buffer[3][column] = fwd3;
            split_buffer[0][self.height - column] = rev3;
            split_buffer[1][self.height - column] = rev1;
            split_buffer[2][self.height - column] = rev2;
            split_buffer[3][self.height - column] = rev0;
        }
    }

    fn perform_dht_inplace(&self, buffer: &mut [T], scratch: &mut [T]) {
        // Step 1: Transpose the width x height array to height x width
        array_utils::transpose_half_rev_out(buffer, scratch, 4, self.height);

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
        array_utils::transpose_half_rev_out(input, output, 4, self.height);

        // Step 2: Compute DHTs of size `height` down the rows of our transposed array
        self.height_size_fft.process_with_scratch(output, input);

        // Step 3: Apply twiddle factors
        self.apply_twiddles(output);
    }
}
boilerplate_dht!(MixedRadix4xn);

pub struct Radix4<T> {
    twiddles: Box<[Complex<T>]>,

    base_dht: Arc<dyn Dht<T>>,
    base_len: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    len: usize,
}

impl<T: FftNum> Radix4<T> {
    pub fn new(len: usize) -> Self {
        assert!(
            len.is_power_of_two(),
            "Radix4 algorithm requires a power-of-two input size. Got {}",
            len
        );

        // figure out which base length we're going to use
        let num_bits = len.trailing_zeros();
        let (base_len, base_dht) = match num_bits {
            0 => (len, Arc::new(Butterfly1::new()) as Arc<dyn Dht<T>>),
            1 => (len, Arc::new(Butterfly2::new()) as Arc<dyn Dht<T>>),
            2 => (len, Arc::new(Butterfly4::new()) as Arc<dyn Dht<T>>),
            _ => {
                if num_bits % 2 == 1 {
                    (8, Arc::new(Butterfly8::new()) as Arc<dyn Dht<T>>)
                } else {
                    (16, Arc::new(Butterfly16::new()) as Arc<dyn Dht<T>>)
                }
            }
        };

        // precompute the twiddle factors this algorithm will use.
        // we're doing the same precomputation of twiddle factors as the mixed radix algorithm where width=4 and height=len/4
        // but mixed radix only does one step and then calls itself recusrively, and this algorithm does every layer all the way down
        // so we're going to pack all the "layers" of twiddle factors into a single array, starting with the bottom layer and going up
        let mut twiddle_stride = len / (base_len * 4);
        let mut twiddle_factors = Vec::with_capacity(len);
        while twiddle_stride > 0 {
            let num_rows = len / (twiddle_stride * 4);
            for i in 1..num_rows {
                for k in 1..3 {
                    let twiddle = twiddles::compute_dft_twiddle_inverse(i * k * twiddle_stride, len);
                    twiddle_factors.push(twiddle);
                }
            }
            twiddle_stride >>= 2;
        }

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

            base_dht,
            base_len,

            len,
            inplace_scratch_len: len,
            outofplace_scratch_len: 0,
        }
    }

    fn perform_dht_inplace(&self, buffer: &mut [T], scratch: &mut [T]) {
        scratch.copy_from_slice(buffer);
        self.perform_dht_out_of_place(scratch, buffer, &mut [])
    }

    fn perform_dht_out_of_place(
        &self,
        input: &mut [T],
        output: &mut [T],
        _scratch: &mut [T],
    ) {
        // copy the data into the spectrum vector
        prepare_radix4(self.len, self.base_len, input, output, 1);

        // Base-level FFTs
        self.base_dht.process_with_scratch(output, &mut []);

        // cross-FFTs
        let mut current_size = self.base_len * 4;
        let mut layer_twiddles: &[Complex<T>] = &self.twiddles;

        while current_size <= self.len {
            let num_rows = self.len / current_size;

            for i in 0..num_rows {
                unsafe {
                    butterfly_4(
                        &mut output[i * current_size..],
                        layer_twiddles,
                        current_size / 4,
                    )
                }
            }

            //skip past all the twiddle factors used in this layer
            let twiddle_offset = (current_size * 2) / 4 - 2;
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            current_size *= 4;
        }
    }
}
boilerplate_dht!(Radix4);

fn prepare_radix4<T: FftNum>(
    size: usize,
    base_len: usize,
    input: &[T],
    output: &mut [T],
    stride: usize,
) {
    if size == base_len {
        unsafe {
            for i in 0..size {
                *output.get_unchecked_mut(i) = *input.get_unchecked(i * stride);
            }
        }
    } else {
        for i in 0..3 {
            prepare_radix4(
                size / 4,
                base_len,
                &input[i * stride..],
                &mut output[i * (size / 4)..],
                stride * 4,
            );
        }

        {
            let i = 3;
            prepare_radix4_reversed(
                size / 4,
                base_len,
                &input[i * stride..],
                &mut output[i * (size / 4)..],
                stride * 4,
            );
        }
    }
}

fn prepare_radix4_reversed<T: FftNum>(
    size: usize,
    base_len: usize,
    input: &[T],
    output: &mut [T],
    stride: usize,
) {
    if size == base_len {
        unsafe {
            for i in 0..size {
                *output.get_unchecked_mut(i) = *input.get_unchecked((size - i - 1) * stride);
            }
        }
    } else {
        {
            let i = 0;
            prepare_radix4(
                size / 4,
                base_len,
                &input[i * stride..],
                &mut output[i * (size / 4)..],
                stride * 4,
            );
        }
        for i in 1..4 {
            prepare_radix4_reversed(
                size / 4,
                base_len,
                &input[i * stride..],
                &mut output[i * (size / 4)..],
                stride * 4,
            );
        }
    }
}

unsafe fn butterfly_4<T: FftNum>(
    data: &mut [T],
    twiddles: &[Complex<T>],
    num_ffts: usize,
) {
    let butterfly4 = Butterfly4::new();

    // For the rest of this function, we're going to be dealing with the buffer in 4 chunks simultaneously, so go ahead and make that split now
    let split_buffer = {
        let mut buffer_chunks = data.chunks_exact_mut(num_ffts);

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

        butterfly4.perform_dht_butterfly(&mut col0);

        for i in 0..4 {
            split_buffer[i][0] = col0[i];
        }
    }

    
    for (column, twiddle_chunk) in (1..num_ffts/2 + 1).zip(twiddles.chunks_exact(2)) {
        let input0fwd = split_buffer[0][column];
        let input0rev = split_buffer[0][num_ffts - column];
        let input1fwd = split_buffer[1][column];
        let input1rev = split_buffer[1][num_ffts - column];
        let input2fwd = split_buffer[2][column];
        let input2rev = split_buffer[2][num_ffts - column];
        let input3fwd = split_buffer[3][column];
        let input3rev = split_buffer[3][num_ffts - column];   
        
        let out1fwd = input1fwd + input3rev;
        let out3fwd = input1rev - input3fwd;
        let out1rev = input3fwd + input1rev;
        let out3rev = input3rev - input1fwd;
        
        let mut fwd0: T = input0fwd;
        let mut rev0: T = input0rev;
        let mut fwd2: T = twiddle_chunk[1].re * input2fwd + twiddle_chunk[1].im * input2rev;
        let mut rev2: T = twiddle_chunk[1].re * input2rev - twiddle_chunk[1].im * input2fwd;

        let mut fwd1: T = twiddle_chunk[0].re * out1fwd   + twiddle_chunk[0].im * out3fwd;
        let mut rev1: T = twiddle_chunk[0].re * out1rev   + twiddle_chunk[0].im * out3rev;
        let mut fwd3: T = twiddle_chunk[0].re * out3fwd   - twiddle_chunk[0].im * out1fwd;
        let mut rev3: T = twiddle_chunk[0].re * out3rev   - twiddle_chunk[0].im * out1rev;

        Butterfly2::perform_dht_strided(&mut fwd0, &mut fwd2);
        Butterfly2::perform_dht_strided(&mut fwd0, &mut fwd1);
        Butterfly2::perform_dht_strided(&mut fwd2, &mut fwd3);
        Butterfly2::perform_dht_strided(&mut rev0, &mut rev2);
        Butterfly2::perform_dht_strided(&mut rev0, &mut rev1);
        Butterfly2::perform_dht_strided(&mut rev2, &mut rev3);
        
        split_buffer[0][column] = fwd0;
        split_buffer[1][column] = fwd2;
        split_buffer[2][column] = fwd1;
        split_buffer[3][column] = fwd3;
        split_buffer[0][num_ffts - column] = rev3;
        split_buffer[1][num_ffts - column] = rev1;
        split_buffer[2][num_ffts - column] = rev2;
        split_buffer[3][num_ffts - column] = rev0;
    }
}

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

    #[test]
    fn test_radix4_correct() {
        for power2 in 2..7 {
            test_radix4_with_length(1 << power2);
        }
    }

    fn test_radix4_with_length(len: usize) {
        let dht = Radix4::<f32>::new(len);

        let height_dht = Arc::new(DhtNaive::new(len / 4)) as Arc<dyn Dht<f32>>;
        let control_dht = MixedRadix4xn::new(height_dht);

        check_dht_algorithm(&control_dht, len);
        check_dht_algorithm(&dht, len);
    }
}