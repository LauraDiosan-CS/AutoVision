# Step 1: Create a bytes-like object
data = bytearray(b'Hello, World!')

# Step 2: Create a memoryview of the bytes-like object
mv = memoryview(data)

# Step 3: Convert the memoryview back to bytes
bytes_data = mv.tobytes()

data[0] = ord("X")

print(f"Original data: {data}")
print(f"Memoryview: {mv}")
print(f"Converted to bytes: {bytes_data}")