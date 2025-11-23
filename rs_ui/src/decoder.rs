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
        out_chunk[3].write(255); // A
    }

    // 3. Convert to Box<[u8]>
    // SAFETY: We have iterated through every chunk and guaranteed
    // that every byte has been written to via the loop above.
    unsafe { buffer.assume_init() }
}

#[derive(Debug)]
pub struct DecodedImagesView<'a> {
    pub description: &'a str,
    pub images: Vec<NamedImageView<'a>>,
}

#[derive(Debug)]
pub struct NamedImageView<'a> {
    pub name: &'a str,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub pixels: &'a [u8],
}

pub fn decode_named_images(buffer: &'_ [u8]) -> DecodedImagesView<'_> {
    let mut images = Vec::new();

    const DESCRIPTION_SIZE: usize = 1024;
    const NAME_SIZE: usize = 55;
    const WIDTH_SIZE: usize = 4;
    const HEIGHT_SIZE: usize = 4;
    const CHANNELS_SIZE: usize = 1;
    const HEADER_SIZE: usize = NAME_SIZE + WIDTH_SIZE + HEIGHT_SIZE + CHANNELS_SIZE; // = 64 bytes

    let description = {
        let description_desc = &buffer[..DESCRIPTION_SIZE];
        let end = description_desc
            .iter()
            .take(DESCRIPTION_SIZE)
            .position(|&c| c == 0)
            .unwrap_or(DESCRIPTION_SIZE);

        str::from_utf8(&description_desc[..end]).unwrap_or("<invalid utf8>")
    };

    let mut offset = DESCRIPTION_SIZE;

    while offset < buffer.len() {
        // 1. Safety Check: Header availability
        if offset + HEADER_SIZE > buffer.len() {
            eprintln!("Buffer ends prematurely at header.");
            break;
        }

        let offset_header_end = offset + HEADER_SIZE;

        // 2. Parse Name (First 64 bytes)
        let name_raw = &buffer[offset..offset + NAME_SIZE];
        offset += NAME_SIZE;
        // Find the first null byte to trim the string, or take the whole thing
        let end = name_raw
            .iter()
            .take(NAME_SIZE)
            .position(|&c| c == 0)
            .unwrap_or(NAME_SIZE);
        let name_str = str::from_utf8(&name_raw[..end]).unwrap_or("<invalid utf8>");

        // 3. Parse Dimensions (Next 12 bytes)
        let width = u32::from_ne_bytes([
            buffer[offset],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ]);
        offset += WIDTH_SIZE;
        let height = u32::from_ne_bytes([
            buffer[offset],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ]);
        offset += HEIGHT_SIZE;
        let channels = buffer[offset];
        offset += CHANNELS_SIZE;

        assert_eq!(offset_header_end, offset);

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

    DecodedImagesView {
        description,
        images,
    }
}
