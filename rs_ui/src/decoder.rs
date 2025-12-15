pub fn grayscale_to_rgba(gray_data: &[u8]) -> Box<[u8]> {
    gray_data
        .iter()
        .flat_map(|&luma| [luma, luma, luma, 255])
        .collect()
}

pub fn bgr_to_rgba(rgb_data: &[u8]) -> Box<[u8]> {
    let num_pixels = rgb_data.len() / 3;

    // This is fast (no zeroing)
    let mut buffer = Box::new_uninit_slice(num_pixels * 4);

    // This allows SIMD optimization
    for (in_chunk, out_chunk) in rgb_data.chunks_exact(3).zip(buffer.chunks_exact_mut(4)) {
        // .write() is safe; it writes the value and returns a mutable reference
        out_chunk[0].write(in_chunk[2]); // R
        out_chunk[1].write(in_chunk[1]); // G
        out_chunk[2].write(in_chunk[0]); // B
        out_chunk[3].write(255); // A
    }

    // SAFETY: We have iterated through every chunk and guaranteed
    // that every byte has been written to via the loop above.
    unsafe { buffer.assume_init() }
}
