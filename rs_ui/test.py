import struct
import array
from time import sleep
from rs_ipc import SharedMessage, OperationMode

from PIL import Image
import os.path

from rs_ipc.rs_ipc import ReaderWaitPolicy


def create_image_bytes(image_id, width, height, pixels_data):
    # 1. Define the format for the fixed-size "header" part of the struct
    # = : standard size, no alignment (crucial for C struct compatibility)
    # Q : unsigned long long (8 bytes) for uint64_t id
    # I : unsigned int (4 bytes) for uint32_t width
    # I : unsigned int (4 bytes) for uint32_t height
    header_format = '=QII'

    # 2. Pack the header fields
    try:
        header = struct.pack(header_format, image_id, width, height)
    except struct.error as e:
        print(f"Error packing header: {e}")
        print("Please ensure image_id, width, and height are valid integers.")
        return None

    # 3. Ensure the pixel data is in a raw bytes format
    pixel_bytes = b''
    if isinstance(pixels_data, (bytes, bytearray)):
        pixel_bytes = pixels_data
    elif isinstance(pixels_data, list):
        # This is less efficient for large data, but good for demonstration.
        # Note: all values in the list must be valid bytes (0-255)
        try:
            # Using array.array is more efficient than bytes(list)
            # for large lists of numbers
            pixel_array = array.array('B', pixels_data)
            pixel_bytes = pixel_array.tobytes()
        except (ValueError, OverflowError) as e:
            print(f"Error converting pixel list to bytes: {e}")
            print("Ensure all items in the pixel list are integers from 0 to 255.")
            return None
    else:
        print(f"Unsupported pixels_data type: {type(pixels_data)}")
        print("Please provide bytes, bytearray, or a list of ints [0-255].")
        return None

    # 4. Concatenate the header and the pixel data
    # This directly mimics the C struct's memory layout
    image_bytes = header + pixel_bytes

    return image_bytes

def process_and_pack_image(filename, image_id):
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        print("Please make sure the image is in the same directory as the script.")
        # Let's create a dummy placeholder image to allow the script to run
        print("Creating a 16x16 dummy image as a placeholder...")
        try:
            # Create a simple red dummy image
            img = Image.new('RGBA', (16, 16), (255, 0, 0, 255))
            width, height = img.size
            pixel_data = img.tobytes()
            print(f"Using dummy image: {width}x{height}")
        except Exception as e:
            print(f"Failed to create dummy image: {e}")
            return None # Skip this file
    else:
        # File exists, load it
        try:
            with Image.open(filename) as img:
                # Convert to RGBA to ensure 4 bytes per pixel
                rgba_img = img.convert('RGBA')
                width, height = rgba_img.size
                pixel_data = rgba_img.tobytes()

                print(f"Loaded image: {width}x{height}")
                print(f"Pixel data size: {len(pixel_data)} bytes")
                # Expected size: width * height * 4 (for RGBA)
                expected_size = width * height * 4
                if len(pixel_data) != expected_size:
                    print(f"Warning: Pixel data size ({len(pixel_data)}) does not match "
                          f"expected RGBA size ({expected_size}).")

        except Exception as e:
            print(f"Error loading or converting image '{filename}': {e}")
            return None

    # Now pack the data
    packed_image = create_image_bytes(image_id, width, height, pixel_data)

    if packed_image:
        print(f"Successfully packed image data for ID {image_id}.")
        print(f"Total bytes length: {len(packed_image)}")

        # Verification
        header_size = struct.calcsize('=QII') # 16 bytes
        header = packed_image[:header_size]
        pixels = packed_image[header_size:]

        unpacked_id, unpacked_width, unpacked_height = struct.unpack('=QII', header)

        assert unpacked_id == image_id
        assert unpacked_width == width
        assert unpacked_height == height
        assert pixels == pixel_data

        print("  Verification successful!")

    return packed_image

if __name__ == "__main__":
    # Before running, make sure you have Pillow installed:
    # pip install Pillow

    # And that 'ferris.png' and 'ferris2.png' are in the same folder.

    image_files_to_pack = [
        ("assets/ferris.png", 1001),
        ("assets/ferris2.png", 1002)
    ]

    all_packed_data = []

    print("--- Starting Image Packing Process ---")

    memory: SharedMessage = SharedMessage.open(
        "test", OperationMode.WriteSync(ReaderWaitPolicy.All())
    )

    i = 0
    while True:
        i += 1
        filename, img_id = image_files_to_pack[i % 2]
        packed_data = process_and_pack_image(filename, img_id)
        if packed_data:
            all_packed_data.append(packed_data)
        memory.write(packed_data)
        sleep(1)