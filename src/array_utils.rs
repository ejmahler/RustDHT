

// Loop over exact chunks of the provided buffer. Very similar in semantics to ChunksExactMut, but generates smaller code and requires no modulo operations
// Returns Ok() if every element ended up in a chunk, Err() if there was a remainder
pub fn iter_chunks<T>(
    mut buffer: &mut [T],
    chunk_size: usize,
    mut chunk_fn: impl FnMut(&mut [T]),
) -> Result<(), ()> {
    // Loop over the buffer, splicing off chunk_size at a time, and calling chunk_fn on each
    while buffer.len() >= chunk_size {
        let (head, tail) = buffer.split_at_mut(chunk_size);
        buffer = tail;

        chunk_fn(head);
    }

    // We have a remainder if there's data still in the buffer -- in which case we want to indicate to the caller that there was an unwanted remainder
    if buffer.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

// Loop over exact zipped chunks of the 2 provided buffers. Very similar in semantics to ChunksExactMut.zip(ChunksExactMut), but generates smaller code and requires no modulo operations
// Returns Ok() if every element of both buffers ended up in a chunk, Err() if there was a remainder
pub fn iter_chunks_zipped<T>(
    mut buffer1: &mut [T],
    mut buffer2: &mut [T],
    chunk_size: usize,
    mut chunk_fn: impl FnMut(&mut [T], &mut [T]),
) -> Result<(), ()> {
    // If the two buffers aren't the same size, record the fact that they're different, then snip them to be the same size
    let uneven = if buffer1.len() > buffer2.len() {
        buffer1 = &mut buffer1[..buffer2.len()];
        true
    } else if buffer2.len() < buffer1.len() {
        buffer2 = &mut buffer2[..buffer1.len()];
        true
    } else {
        false
    };

    // Now that we know the two slices are the same length, loop over each one, splicing off chunk_size at a time, and calling chunk_fn on each
    while buffer1.len() >= chunk_size && buffer2.len() >= chunk_size {
        let (head1, tail1) = buffer1.split_at_mut(chunk_size);
        buffer1 = tail1;

        let (head2, tail2) = buffer2.split_at_mut(chunk_size);
        buffer2 = tail2;

        chunk_fn(head1, head2);
    }

    // We have a remainder if the 2 chunks were uneven to start with, or if there's still data in the buffers -- in which case we want to indicate to the caller that there was an unwanted remainder
    if !uneven && buffer1.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

#[derive(Copy, Clone)]
pub struct RawSlice<T> {
    ptr: *const T,
    slice_len: usize,
}
impl<T> RawSlice<T> {
    #[inline(always)]
    pub fn new(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub unsafe fn new_transmuted<U>(slice: &[U]) -> Self {
        Self {
            ptr: slice.as_ptr() as *const T,
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slice_len
    }
}
impl<T: Copy> RawSlice<T> {
    #[inline(always)]
    pub unsafe fn load(&self, index: usize) -> T {
        debug_assert!(index < self.slice_len);
        *self.ptr.add(index)
    }
}

/// A RawSliceMut is a normal mutable slice, but aliasable. Its functionality is severely limited.
#[derive(Copy, Clone)]
pub struct RawSliceMut<T> {
    ptr: *mut T,
    slice_len: usize,
}
impl<T> RawSliceMut<T> {
    #[inline(always)]
    pub fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub unsafe fn new_transmuted<U>(slice: &mut [U]) -> Self {
        Self {
            ptr: slice.as_mut_ptr() as *mut T,
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slice_len
    }
    #[inline(always)]
    pub unsafe fn store(&self, value: T, index: usize) {
        debug_assert!(index < self.slice_len);
        *self.ptr.add(index) = value;
    }
}

// Transpose the input to the output, but reverse the second half of the output rows.
// Intended for use with MixedRadix#xn. Fastest if width is less than ~10
#[inline(always)]
pub fn transpose_half_rev_out<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize) {
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len(), width * height);

    let reversal_row_begin = width / 2 + 1;
    for out_row in 0..height {
        let out_row_rev = height - out_row - 1;
        for out_column in 0..reversal_row_begin {
            unsafe { *output.get_unchecked_mut(out_column * height + out_row) = *input.get_unchecked(out_row * width + out_column) };
        }
        for out_column in reversal_row_begin..width {
            unsafe { *output.get_unchecked_mut(out_column * height + out_row_rev) = *input.get_unchecked(out_row * width + out_column) };
        }
    }
}