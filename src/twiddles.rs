use num_complex::Complex;
use rustfft::FftNum;


pub fn compute_dht_twiddle<T: FftNum>(
    index: usize,
    fft_len: usize,
) -> T {
    let constant = 2f64 * std::f64::consts::PI / fft_len as f64;
    let angle = constant * index as f64;

    T::from_f64(angle.cos()).unwrap() + T::from_f64(angle.sin()).unwrap()
}

pub fn compute_dft_twiddle_inverse<T: FftNum>(
    index: usize,
    fft_len: usize,
) -> Complex<T> {
    let constant = 2f64 * std::f64::consts::PI / fft_len as f64;
    let angle = constant * index as f64;

    let result = Complex {
        re: T::from_f64(angle.cos()).unwrap(),
        im: T::from_f64(angle.sin()).unwrap(),
    };

    result
}

pub fn compute_dft_twiddle_forward<T: FftNum>(
    index: usize,
    fft_len: usize,
) -> Complex<T> {
    let constant = -2f64 * std::f64::consts::PI / fft_len as f64;
    let angle = constant * index as f64;

    let result = Complex {
        re: T::from_f64(angle.cos()).unwrap(),
        im: T::from_f64(angle.sin()).unwrap(),
    };

    result
}