from multiprocessing import shared_memory


class SharedMemoryWriter:
    __slots__ = ['shared_mem', 'version']

    def __init__(self, topic: str, size: int):
        self.shared_mem = shared_memory.SharedMemory(create=True, name=topic, size=size)
        self.version = 0

    def write(self, data: bytes):
        int_byte_count = 8
        version_and_size_len = int_byte_count + int_byte_count
        if len(data) > self.shared_mem.size - version_and_size_len:
            raise ValueError(f"Data size is too big. Max size allowed is {self.shared_mem.size - version_and_size_len}")

        self.shared_mem.buf[:int_byte_count] = self.version.to_bytes(int_byte_count)
        self.shared_mem.buf[int_byte_count:version_and_size_len] = len(data).to_bytes(int_byte_count)
        self.shared_mem.buf[version_and_size_len:version_and_size_len + len(data)] = data
        self.version += 1

    def close(self):
        self.shared_mem.unlink()
        self.shared_mem.close()


class SharedMemoryReader:
    __slots__ = ['shared_mem', 'last_read_version']

    def __init__(self, topic: str):
        self.shared_mem = shared_memory.SharedMemory(name=topic)
        self.last_read_version = 0

    def read(self) -> (bytearray | None):
        int_byte_count = 8
        version_and_size_len = int_byte_count + int_byte_count
        version = int.from_bytes(self.shared_mem.buf[:int_byte_count])

        if version == self.last_read_version:
            return None
        self.last_read_version = version

        data_size = int.from_bytes(self.shared_mem.buf[int_byte_count:version_and_size_len])
        buffer = bytearray(data_size)
        buffer[:] = self.shared_mem.buf[version_and_size_len:version_and_size_len + data_size].tobytes()

        return buffer

    def close(self):
        self.shared_mem.close()