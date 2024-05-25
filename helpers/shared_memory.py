import time
from multiprocessing import shared_memory


class SharedMemoryWriter:
    __slots__ = ['shared_mem', 'version']

    def __init__(self, topic: str, size: int = None, create: bool = True):
        if create:
            self.shared_mem = shared_memory.SharedMemory(create=create, name=topic, size=size)
        else:
            self.shared_mem = shared_memory.SharedMemory(create=False, name=topic)
        print(f"Creating shared memory with name: {topic}:{self.shared_mem.name} and size: {self.shared_mem.size}")

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

    def __init__(self, topic: str, size: int = None,  create: bool = False):
        if create:
            self.shared_mem = shared_memory.SharedMemory(name=topic, create=create, size=size)
        else:
            self.shared_mem = shared_memory.SharedMemory(name=topic, create=False)
        self.last_read_version = 0

    def read(self, skip_same_version: bool = True) -> bytes | None:
        int_byte_count = 8
        version_and_size_len = int_byte_count + int_byte_count
        version = int.from_bytes(self.shared_mem.buf[:int_byte_count])

        if skip_same_version and version == self.last_read_version:
            return None
        self.last_read_version = version

        data_size = int.from_bytes(self.shared_mem.buf[int_byte_count:version_and_size_len])
        # buffer = bytearray(data_size)
        # buffer[:] = self.shared_mem.buf[version_and_size_len:version_and_size_len + data_size].tobytes()
        return self.shared_mem.buf[version_and_size_len:version_and_size_len + data_size].tobytes() # TODO check

    def read_new_version(self, timeout: int = 0.01) -> bytes:
        data = self.read()

        while data is None:
            time.sleep(timeout)
            data = self.read()

        return data

    def close(self):
        self.shared_mem.close()