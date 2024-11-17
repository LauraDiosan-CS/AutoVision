import pickle

import numpy as np
from ripc import SharedMemoryReader, SharedMemoryWriter


def send_dict_via_shared_memory(d, pipe_mem_name, pipe_mem_size):
    # Serialize the dictionary into bytes
    serialized_data = pickle.dumps(d)

    # Create a SharedMemoryWriter object
    writer = SharedMemoryWriter(name=pipe_mem_name, size=pipe_mem_size)

    # Write the serialized data to shared memory
    writer.write(serialized_data)

    # Get the size of the data written
    size = writer.size()

    # Return the writer and size for later use
    return writer, size

def receive_dict_via_shared_memory(pipe_mem_name, size):
    # Create a SharedMemoryReader object
    reader = SharedMemoryReader(name=pipe_mem_name)

    # Read the data from shared memory
    data = reader.blocking_read()

    # Deserialize the dictionary from bytes
    deserialized_data = pickle.loads(data)

    return deserialized_data

# Create a more complex dictionary resembling images
def create_image_dict():
    # Simulate image data with NumPy arrays
    image_dict = {
        'image1': np.random.rand(64, 64, 3),  # Random 64x64 RGB image
        'image2': np.random.rand(128, 128, 3),  # Random 128x128 RGB image
        'image3': np.random.rand(256, 256, 3),  # Random 256x256 RGB image
        'metadata': {'author': 'John Doe', 'date': '2024-09-12'}
    }
    return image_dict

def print_dict_diff(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    all_keys = keys1.union(keys2)

    for key in all_keys:
        if key not in dict1:
            print(f"Key '{key}' is missing in the first dictionary.")
        elif key not in dict2:
            print(f"Key '{key}' is missing in the second dictionary.")
        else:
            val1, val2 = dict1[key], dict2[key]
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    print(f"Difference in array for key '{key}':")
                    print(f"Original array shape: {val1.shape}, Received array shape: {val2.shape}")
                    # Optionally, show more detailed differences if needed
            elif val1 != val2:
                print(f"Difference for key '{key}':")
                print(f"Original value: {val1}")
                print(f"Received value: {val2}")

original_dict = create_image_dict()
# Send the dictionary
writer, size = send_dict_via_shared_memory(original_dict, "my_pipe", 10000000)

# Receive the dictionary
received_dict = receive_dict_via_shared_memory("my_pipe", size)

# Verify that the received dictionary is the same as the original
print("Original dictionary:", original_dict)
print("Received dictionary:", received_dict)
# Print differences
print("Comparing original and received dictionaries:")
print_dict_diff(original_dict, received_dict)