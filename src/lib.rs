use num_traits::Zero;
use rustfft::{FftNum, Length};

#[macro_use]
mod macros;

pub mod scalar;
mod twiddles;
mod array_utils;

#[cfg(test)]
mod test_utils;

/// Trait for algorithms that compute DHTs.
///
/// This trait has a few methods for computing DHTs. Its most conveinent method is [`process(slice)`](crate::Dht::process).
/// It takes in a slice of `T` and computes a DHT on that slice, in-place. It may copy the data over to internal scratch buffers
/// if that speeds up the computation, but the output will always end up in the same slice as the input.
pub trait Dht<T: FftNum>: Length + Sync + Send {
    /// Computes a DHT in-place.
    ///
    /// Convenience method that allocates a `Vec` with the required scratch space and calls `self.process_with_scratch`.
    /// If you want to re-use that allocation across multiple DHT computations, consider calling `process_with_scratch` instead.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() % self.len() > 0`
    /// - `buffer.len() < self.len()`
    fn process(&self, buffer: &mut [T]) {
        let mut scratch = vec![Zero::zero(); self.get_inplace_scratch_len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    /// Divides `buffer` into chunks of size `self.len()`, and computes a DHT on each chunk.
    ///
    /// Uses the `scratch` buffer as scratch space, so the contents of `scratch` should be considered garbage
    /// after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() % self.len() > 0`
    /// - `buffer.len() < self.len()`
    /// - `scratch.len() < self.get_inplace_scratch_len()`
    fn process_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);

    /// Divides `input` and `output` into chunks of size `self.len()`, and computes a DHT on each chunk.
    ///
    /// This method uses both the `input` buffer and `scratch` buffer as scratch space, so the contents of both should be
    /// considered garbage after calling.
    ///
    /// This is a more niche way of computing a DHT. It's useful to avoid a `copy_from_slice()` if you need the output
    /// in a different buffer than the input for some reason. This happens frequently in RustDHT internals, but is probably
    /// less common among RustDHT users.
    ///
    /// For many DHT sizes, `self.get_outofplace_scratch_len()` returns 0
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `output.len() != input.len()`
    /// - `input.len() % self.len() > 0`
    /// - `input.len() < self.len()`
    /// - `scratch.len() < self.get_outofplace_scratch_len()`
    fn process_outofplace_with_scratch(
        &self,
        input: &mut [T],
        output: &mut [T],
        scratch: &mut [T],
    );

    /// Returns the size of the scratch buffer required by `process_with_scratch`
    ///
    /// For most DHT sizes, this method will return `self.len()`. For a few small sizes it will return 0, and for some special DHT sizes
    /// (Sizes that require the use of Bluestein's Algorithm), this may return a scratch size larger than `self.len()`.
    /// The returned value may change from one version of RustDHT to the next.
    fn get_inplace_scratch_len(&self) -> usize;

    /// Returns the size of the scratch buffer required by `process_outofplace_with_scratch`
    ///
    /// For most DHT sizes, this method will return 0. For some special DHT sizes
    /// (Sizes that require the use of Bluestein's Algorithm), this may return a scratch size larger than `self.len()`.
    /// The returned value may change from one version of RustDHT to the next.
    fn get_outofplace_scratch_len(&self) -> usize;
}

// Prints an error raised by an in-place DHT algorithm's `process_inplace` method
// Marked cold and inline never to keep all formatting code out of the many monomorphized process_inplace methods
#[cold]
#[inline(never)]
fn dht_error_inplace(
    expected_len: usize,
    actual_len: usize,
    expected_scratch: usize,
    actual_scratch: usize,
) {
    assert!(
        actual_len >= expected_len,
        "Provided DHT buffer was too small. Expected len = {}, got len = {}",
        expected_len,
        actual_len
    );
    assert_eq!(
        actual_len % expected_len,
        0,
        "Input DHT buffer must be a multiple of DHT length. Expected multiple of {}, got len = {}",
        expected_len,
        actual_len
    );
    assert!(
        actual_scratch >= expected_scratch,
        "Not enough scratch space was provided. Expected scratch len >= {}, got scratch len = {}",
        expected_scratch,
        actual_scratch
    );
}

// Prints an error raised by an in-place DHT algorithm's `process_inplace` method
// Marked cold and inline never to keep all formatting code out of the many monomorphized process_inplace methods
#[cold]
#[inline(never)]
fn dht_error_outofplace(
    expected_len: usize,
    actual_input: usize,
    actual_output: usize,
    expected_scratch: usize,
    actual_scratch: usize,
) {
    assert_eq!(actual_input, actual_output, "Provided DHT input buffer and output buffer must have the same length. Got input.len() = {}, output.len() = {}", actual_input, actual_output);
    assert!(
        actual_input >= expected_len,
        "Provided DHT buffer was too small. Expected len = {}, got len = {}",
        expected_len,
        actual_input
    );
    assert_eq!(
        actual_input % expected_len,
        0,
        "Input DHT buffer must be a multiple of DHT length. Expected multiple of {}, got len = {}",
        expected_len,
        actual_input
    );
    assert!(
        actual_scratch >= expected_scratch,
        "Not enough scratch space was provided. Expected scratch len >= {}, got scratch len = {}",
        expected_scratch,
        actual_scratch
    );
}