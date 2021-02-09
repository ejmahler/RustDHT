use num_traits::{Float, Zero};

use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rand::{rngs::StdRng, SeedableRng};

use rustfft::{FftNum, Length};
use crate::{scalar::DhtNaive, Dht};

/// The seed for the random number generator used to generate
/// random signals. It's defined here so that we have deterministic
/// tests
const RNG_SEED: [u8; 32] = [
    1, 9, 1, 0, 1, 1, 4, 3, 1, 4, 9, 8, 4, 1, 4, 8, 2, 8, 1, 2, 2, 2, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

pub fn random_real_signal<T: FftNum + SampleUniform>(length: usize) -> Vec<T> {
    let mut result = Vec::with_capacity(length);
    let normal_dist: Uniform<T> = Uniform::new(T::zero(), T::from_f32(10.0).unwrap());
    let mut rng: StdRng = SeedableRng::from_seed(RNG_SEED);
    for _ in 0..length {
        result.push(normal_dist.sample(&mut rng));
    }
    return result;
}

pub fn compare_real_vectors<T: FftNum + Float>(vec1: &[T], vec2: &[T]) -> bool {
    assert_eq!(vec1.len(), vec2.len());
    let mut error = T::zero();
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        error = error + (a - b).abs();
    }
    return (error.to_f64().unwrap() / vec1.len() as f64) < 0.1f64;
}

fn _transpose_diagnostic_real<T: FftNum + Float>(expected: &[T], actual: &[T]) {
    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        if (e - a).abs().to_f32().unwrap() > 0.01 {
            if let Some(found_index) = expected
                .iter()
                .position(|&ev| (ev - a).abs().to_f32().unwrap() < 0.01)
            {
                println!("{} incorrectly contained {}", i, found_index);
            } else {
                println!("{} X", i);
            }
        }
    }
}

#[allow(unused)]
pub fn check_dht_algorithm<T: FftNum + Float + SampleUniform>(
    fft: &dyn Dht<T>,
    len: usize,
) {
    assert_eq!(
        fft.len(),
        len,
        "Algorithm reported incorrect size. Expected {}, got {}",
        len,
        fft.len()
    );

    let n = 1;

    let dht = DhtNaive::new(len);

    let dirty_scratch_value = T::one() * T::from_i32(100).unwrap();

    // set up buffers
    let reference_input = random_real_signal(len * n);
    let mut expected_output = reference_input.clone();
    let mut dft_scratch = vec![Zero::zero(); dht.get_inplace_scratch_len()];
    dht.process_with_scratch(&mut expected_output, &mut dft_scratch);

    // test process()
    {
        let mut buffer = reference_input.clone();

        fft.process(&mut buffer);
        
        assert!(
            compare_real_vectors(&expected_output, &buffer),
            "process() failed, length = {}",
            len
        );
    }

    // test process_with_scratch()
    {
        let mut buffer = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];

        fft.process_with_scratch(&mut buffer, &mut scratch);

        assert!(
            compare_real_vectors(&expected_output, &buffer),
            "process_with_scratch() failed, length = {}",
            len
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            buffer.copy_from_slice(&reference_input);

            fft.process_with_scratch(&mut buffer, &mut scratch);

            assert!(compare_real_vectors(&expected_output, &buffer), "process_with_scratch() failed the 'dirty scratch' test, length = {}", len);
        }
    }

    // test process_outofplace_with_scratch()
    {
        let mut input = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_outofplace_scratch_len()];
        let mut output = expected_output.clone();

        fft.process_outofplace_with_scratch(&mut input, &mut output, &mut scratch);

        assert!(
            compare_real_vectors(&expected_output, &output),
            "process_outofplace_with_scratch() failed, length = {}",
            len
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            input.copy_from_slice(&reference_input);

            fft.process_outofplace_with_scratch(&mut input, &mut output, &mut scratch);

            assert!(
                compare_real_vectors(&expected_output, &output),
                "process_outofplace_with_scratch() failed the 'dirty scratch' test, length = {}",
                len
            );
        }
    }
}

// A fake DHT algorithm that requests much more scratch than it needs. You can use this as an inner DHT to other algorithms to test their scratch-supplying logic
#[derive(Debug)]
pub struct BigScratchAlgorithm {
    pub len: usize,

    pub inplace_scratch: usize,
    pub outofplace_scratch: usize,
}
impl<T: FftNum> Dht<T> for BigScratchAlgorithm {
    fn process_with_scratch(&self, _buffer: &mut [T], scratch: &mut [T]) {
        assert!(
            scratch.len() >= self.inplace_scratch,
            "Not enough inplace scratch provided, self={:?}, provided scratch={}",
            &self,
            scratch.len()
        );
    }
    fn process_outofplace_with_scratch(
        &self,
        _input: &mut [T],
        _output: &mut [T],
        scratch: &mut [T],
    ) {
        assert!(
            scratch.len() >= self.outofplace_scratch,
            "Not enough OOP scratch provided, self={:?}, provided scratch={}",
            &self,
            scratch.len()
        );
    }
    fn get_inplace_scratch_len(&self) -> usize {
        self.inplace_scratch
    }
    fn get_outofplace_scratch_len(&self) -> usize {
        self.outofplace_scratch
    }
}
impl Length for BigScratchAlgorithm {
    fn len(&self) -> usize {
        self.len
    }
}