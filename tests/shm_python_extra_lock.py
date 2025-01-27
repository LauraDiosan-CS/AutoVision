import pickle
import struct
from multiprocessing import Condition
from multiprocessing.shared_memory import SharedMemory as PySharedMemory


class SharedMemory(object):
    def __init__(self, memory: PySharedMemory):
        self._shm = memory
        self._last_written_version = 0
        self._last_read_version = 0
        self._initialize_fields(self._shm.size - self.HEADER_SIZE)
        self._lock = Condition()

    @staticmethod
    def create(name: str, size: int, mode) -> 'SharedMemory':
        """
        :param name: is recommended to start with a '/'
        :param size: cannot be 0
        """
        return SharedMemory(PySharedMemory(name, True, size))

    @staticmethod
    def open(name: str, mode) -> 'SharedMemory':
        """
        :param name: is recommended to start with a '/'
        """
        return SharedMemory(PySharedMemory(name, False))

    def write(self, bytes_to_write: bytes) -> None:
        """
        Writes the bytes into the shared memory
        """
        with self._lock:
            new_version = self._set_bytes(bytes_to_write)
            self._last_written_version = new_version
            self._lock.notify()

    def try_read(self) -> bytes | None:
        """
        :returns: the message, or None if it's the same version as the last time or if the shared memory is closed
        """
        with self._lock:
            if self._get_closed():
                return None
            new_version = self._get_version()
            if new_version == self._last_read_version:
                return None
            self._last_read_version = new_version
            return self._get_bytes()

    def blocking_read(self) -> bytes | None:
        """
        Keeps checking the shared memory until there is a new version to read,
        This function also releases the GIL, while waiting for a new message
        :returns: the message, or None if the shared memory is closed
        """
        while True:
            with self._lock:
                if self._get_closed():
                    return None
                new_version = self._get_version()
                if new_version == self._last_read_version:
                    self._lock.wait()
                    continue
                self._last_read_version = new_version
                return self._get_bytes()

    def is_new_version_available(self) -> bool:
        """
        Check if the next read will return a new message
        :returns: true if there is a new version
        """
        with self._lock:
            return self._get_version() != self._last_read_version

    def last_written_version(self) -> int:
        """
        :returns: the latest version that was written
        """
        return self._last_written_version

    def last_read_version(self) -> int:
        """
        :returns: the latest version that was read
        """
        return self._last_read_version

    def name(self) -> str:
        """
        :returns: the name of this shared memory file
        """
        return self._shm.name

    def memory_size(self) -> int:
        """
        :returns: Amount of bytes allocated in this shared memory
        """
        return self._shm.size

    def is_closed(self) -> bool:
        """
        Check if the shared memory has been closed by the writer
        :returns: true if the writer has marked this as closed
        """
        with self._lock:
            return self._get_closed()

    def close(self):
        """
        Close the shared memory block.
        """
        with self._lock:
            self._set_closed(True)
            self._lock.notify_all()
        self._shm.close()

    # Define offsets based on Rust struct alignment (assuming 64-bit architecture)
    CLOSED_OFFSET = 0        # AtomicBool (1 byte)
    VERSION_OFFSET = 8       # AtomicUsize (8 bytes)
    SIZE_OFFSET = 16         # usize (8 bytes)
    BYTES_OFFSET = 24        # [u8] starts here
    HEADER_SIZE = BYTES_OFFSET  # Total header size before the byte array

    def _initialize_fields(self, size: int):
        """
        Initialize the shared memory fields to default values.
        """
        # Initialize `closed` to False (0)
        self._shm.buf[self.CLOSED_OFFSET] = 0
        # Initialize `version` to 0
        struct.pack_into('Q', self._shm.buf, self.VERSION_OFFSET, 0)
        # Initialize `size`
        struct.pack_into('Q', self._shm.buf, self.SIZE_OFFSET, 0)
        # Initialize `bytes` to zeros
        for i in range(size):
            self._shm.buf[self.BYTES_OFFSET + i] = 0

    def unlink(self):
        """
        Unlink (delete) the shared memory block. Use with caution.
        """
        self._shm.unlink()

    # ----------------- Field Access Methods -----------------

    def _get_closed(self):
        """
        Get the value of `closed`.
        """
        return bool(self._shm.buf[self.CLOSED_OFFSET])

    def _set_closed(self, value: bool):
        """
        Set the value of `closed`.
        """
        self._shm.buf[self.CLOSED_OFFSET] = 1 if value else 0

    def _get_version(self):
        """
        Get the value of `version`.
        """
        version, = struct.unpack_from('Q', self._shm.buf, self.VERSION_OFFSET)
        return version

    def _set_version(self, version: int):
        """
        Set the value of `version`.
        """
        struct.pack_into('Q', self._shm.buf, self.VERSION_OFFSET, version)

    def _increment_version(self) -> int:
        """
        Atomically increment the `version`.
        """
        version, = struct.unpack_from('Q', self._shm.buf, self.VERSION_OFFSET)
        version += 1
        struct.pack_into('Q', self._shm.buf, self.VERSION_OFFSET, version)
        return version

    def _get_size(self):
        """
        Get the value of `size`.
        """
        size, = struct.unpack_from('Q', self._shm.buf, self.SIZE_OFFSET)
        return size

    def _get_bytes(self):
        """
        Get a copy of the byte array.
        """
        return bytes(self._shm.buf[self.BYTES_OFFSET: self.BYTES_OFFSET + self._get_size()])

    def _set_bytes(self, data: bytes) -> int:
        """
        Set the byte array without truncating or padding.
        Updates the `size` field to reflect the length of `data`.

        :param data: Byte data to write to shared memory.
        """
        data_length = len(data)
        # Ensure the shared memory has enough space
        if self.HEADER_SIZE + data_length > self._shm.size:
            raise ValueError("Not enough shared memory to write data.")

        new_version = self._increment_version()
        # Update the size field
        struct.pack_into('Q', self._shm.buf, self.SIZE_OFFSET, data_length)

        # Write data using memoryview for efficient bulk assignment
        mv = memoryview(self._shm.buf)
        mv[self.BYTES_OFFSET : self.BYTES_OFFSET + data_length] = data
        return new_version

if __name__ == "__main__":
    # Creating a new shared memory block
    shm = SharedMemory.create("test", 1024, mode=None)

    shm.write(pickle.dumps("Hello, World!", protocol=pickle.HIGHEST_PROTOCOL))
    assert shm.is_new_version_available() is True
    res = shm.try_read()
    txt = pickle.loads(res)
    assert txt == "Hello, World!"
    assert shm.last_written_version() == shm.last_read_version()
    assert shm.is_new_version_available() is False
    shm.write(b"Hello, World!2")
    assert shm.blocking_read() == b"Hello, World!2"
    assert shm.last_written_version() == shm.last_read_version()
    assert shm.last_read_version() == 2
    print("Shared memory test passed")

    shm.close()
    shm.unlink()  # Only do this when you're sure no other process is using the shared memory