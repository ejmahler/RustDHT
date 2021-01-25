

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