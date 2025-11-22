
pub fn grayscale_to_rgba(gray_data: &[u8]) -> Box<[u8]> {
    gray_data
        .iter()
        .flat_map(|&luma| [luma, luma, luma, 255])
        .collect()
}

pub fn bgr_to_rgba(rgb_data: &[u8]) -> Box<[u8]> {
    let num_pixels = rgb_data.len() / 3;

    // 1. Allocate uninitialized memory on the heap.
    // This is fast (no zeroing), but the type is Box<[MaybeUninit<u8>]>
    let mut buffer = Box::new_uninit_slice(num_pixels * 4);

    // 2. Write to the buffer using 'zip'
    // This allows SIMD optimization just like the Vec version.
    for (in_chunk, out_chunk) in rgb_data.chunks_exact(3).zip(buffer.chunks_exact_mut(4)) {
        // .write() is safe; it writes the value and returns a mutable reference
        out_chunk[0].write(in_chunk[2]); // R
        out_chunk[1].write(in_chunk[1]); // G
        out_chunk[2].write(in_chunk[0]); // B
        out_chunk[3].write(255);         // A
    }

    // 3. Convert to Box<[u8]>
    // SAFETY: We have iterated through every chunk and guaranteed
    // that every byte has been written to via the loop above.
    unsafe { buffer.assume_init() }
}

#[derive(Debug)]
pub struct NamedImageView<'a> {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub pixels: &'a [u8],
}

pub fn decode_named_images(buffer: &'_ [u8]) -> Vec<NamedImageView<'_>> {
    let mut images = Vec::new();
    let mut offset = 0;

    // 64 bytes (name) + 4 (w) + 4 (h) + 4 (c) = 76 bytes
    const HEADER_SIZE: usize = 64 + 4 + 4 + 4;

    while offset < buffer.len() {
        // 1. Safety Check: Header availability
        if offset + HEADER_SIZE > buffer.len() {
            eprintln!("Buffer ends prematurely at header.");
            break;
        }

        // 2. Parse Name (First 64 bytes)
        let name_raw = &buffer[offset..offset+64];
        // Find the first null byte to trim the string, or take the whole thing
        let end = name_raw.iter().position(|&c| c == 0).unwrap_or(64);
        let name_str = str::from_utf8(&name_raw[..end])
            .unwrap_or("<invalid utf8>")
            .to_string();

        // 3. Parse Dimensions (Next 12 bytes)
        let w_bytes: [u8; 4] = buffer[offset+64..offset+68].try_into().unwrap();
        let h_bytes: [u8; 4] = buffer[offset+68..offset+72].try_into().unwrap();
        let c_bytes: [u8; 4] = buffer[offset+72..offset+76].try_into().unwrap();

        let width = u32::from_ne_bytes(w_bytes);
        let height = u32::from_ne_bytes(h_bytes);
        let channels = u32::from_ne_bytes(c_bytes);

        offset += HEADER_SIZE;

        // 4. Calculate Data Size
        // w * h * channels (assuming 1 byte per channel depth)
        let data_len = (width as usize) * (height as usize) * (channels as usize);

        // 5. Safety Check: Data availability
        if offset + data_len > buffer.len() {
            eprintln!("Buffer ends prematurely at data for image: {}", name_str);
            break;
        }

        // 6. Create View
        images.push(NamedImageView {
            name: name_str,
            width,
            height,
            channels,
            pixels: &buffer[offset..offset + data_len],
        });

        offset += data_len;
    }

    images
}