import struct
import array
from time import sleep

import numpy
from rs_ipc import SharedMessage, OperationMode

from PIL import Image
import os.path

from rs_ipc.rs_ipc import ReaderWaitPolicy


def create_image_bytes(width, height, pixels_data: numpy.ndarray):
    # 1. Define the format for the fixed-size "header" part of the struct
    # = : standard size, no alignment (crucial for C struct compatibility)
    # I : unsigned int (4 bytes) for uint32_t width
    # I : unsigned int (4 bytes) for uint32_t height
    header_format = '=II'

    # 2. Pack the header fields
    try:
        header = struct.pack(header_format, width, height)
    except struct.error as e:
        print(f"Error packing header: {e}")
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
        "test", OperationMode.WriteSync
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