import struct
import time
from multiprocessing import shared_memory

# Define offsets based on the memory layout
MUTEX_OFFSET = 0
CLOSED_OFFSET = MUTEX_OFFSET + 64  # Assuming pthread_mutex_t is 40 bytes
VERSION_OFFSET = CLOSED_OFFSET + 8  # AtomicBool (1 byte)
SIZE_OFFSET = VERSION_OFFSET + 8       # AtomicUsize (8 bytes)
BYTES_OFFSET = SIZE_OFFSET + 8         # Start of byte array
HEADER_SIZE = BYTES_OFFSET  # Total header size before the byte array

import ctypes
import ctypes.util

# Load the pthread library
libc_name = ctypes.util.find_library("pthread")
if not libc_name:
    raise RuntimeError("Could not find pthread library")
libc = ctypes.CDLL(libc_name, use_errno=True)

# Define pthread types
class pthread_mutex_t(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char * 40)]  # Size may vary; adjust as needed

class pthread_mutexattr_t(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char * 8)]   # Size may vary; adjust as needed

# Define pthread functions
pthread_mutex_init = libc.pthread_mutex_init
pthread_mutex_init.argtypes = [ctypes.POINTER(pthread_mutex_t), ctypes.POINTER(pthread_mutexattr_t)]
pthread_mutex_init.restype = ctypes.c_int

pthread_mutex_lock = libc.pthread_mutex_lock
pthread_mutex_lock.argtypes = [ctypes.POINTER(pthread_mutex_t)]
pthread_mutex_lock.restype = ctypes.c_int

pthread_mutex_unlock = libc.pthread_mutex_unlock
pthread_mutex_unlock.argtypes = [ctypes.POINTER(pthread_mutex_t)]
pthread_mutex_unlock.restype = ctypes.c_int

pthread_mutex_destroy = libc.pthread_mutex_destroy
pthread_mutex_destroy.argtypes = [ctypes.POINTER(pthread_mutex_t)]
pthread_mutex_destroy.restype = ctypes.c_int

pthread_mutexattr_init = libc.pthread_mutexattr_init
pthread_mutexattr_init.argtypes = [ctypes.POINTER(pthread_mutexattr_t)]
pthread_mutexattr_init.restype = ctypes.c_int

pthread_mutexattr_setpshared = libc.pthread_mutexattr_setpshared
pthread_mutexattr_setpshared.argtypes = [ctypes.POINTER(pthread_mutexattr_t), ctypes.c_int]
pthread_mutexattr_setpshared.restype = ctypes.c_int

# Constants
PTHREAD_PROCESS_SHARED = 1
def initialize_mutex(shm_buffer):
    # Create ctypes pointers to the mutex in shared memory
    mutex_ptr = ctypes.cast(ctypes.addressof(shm_buffer) + MUTEX_OFFSET, ctypes.POINTER(pthread_mutex_t))

    # Initialize mutex attributes
    attr = pthread_mutexattr_t()
    res = pthread_mutexattr_init(ctypes.byref(attr))
    if res != 0:
        raise OSError(ctypes.get_errno(), "pthread_mutexattr_init failed")

    res = pthread_mutexattr_setpshared(ctypes.byref(attr), PTHREAD_PROCESS_SHARED)
    if res != 0:
        raise OSError(ctypes.get_errno(), "pthread_mutexattr_setpshared failed")

    # Initialize the mutex
    res = pthread_mutex_init(mutex_ptr, ctypes.byref(attr))
    if res != 0:
        raise OSError(ctypes.get_errno(), "pthread_mutex_init failed")
def lock_mutex(shm_buffer):
    mutex_ptr = ctypes.cast(ctypes.addressof(shm_buffer) + MUTEX_OFFSET, ctypes.POINTER(pthread_mutex_t))
    res = pthread_mutex_lock(mutex_ptr)
    if res != 0:
        raise OSError(ctypes.get_errno(), "pthread_mutex_lock failed")

def unlock_mutex(shm_buffer):
    mutex_ptr = ctypes.cast(ctypes.addressof(shm_buffer) + MUTEX_OFFSET, ctypes.POINTER(pthread_mutex_t))
    res = pthread_mutex_unlock(mutex_ptr)
    if res != 0:
        raise OSError(ctypes.get_errno(), "pthread_mutex_unlock failed")

class SharedMemory:
    def __init__(self, memory: shared_memory.SharedMemory, is_new: bool = False):
        self._shm = memory
        self._is_new = is_new
        self._last_written_version = 0
        self._last_read_version = 0
        self._buffer = (ctypes.c_char * self._shm.size).from_buffer(self._shm.buf)

        if self._is_new:
            # Initialize the mutex
            initialize_mutex(self._buffer)
            # Initialize other fields
            self._initialize_fields(self._shm.size - HEADER_SIZE)

    @staticmethod
    def create(name: str, size: int, mode) -> 'SharedMemory':
        """
        :param name: is recommended to start with a '/'
        :param size: cannot be 0
        """
        shm = shared_memory.SharedMemory(name=name, create=True, size=size+HEADER_SIZE)
        return SharedMemory(shm, is_new=True)

    @staticmethod
    def open(name: str, mode) -> 'SharedMemory':
        """
        :param name: is recommended to start with a '/'
        """
        shm = shared_memory.SharedMemory(name=name, create=False)
        return SharedMemory(shm, is_new=False)

    def write(self, bytes_to_write: bytes) -> None:
        """
        Writes the bytes into the shared memory
        """
        lock_mutex(self._buffer)
        try:
            new_version = self._set_bytes(bytes_to_write)
            self._last_written_version = new_version # self._increment_version()
        finally:
            unlock_mutex(self._buffer)

    def try_read(self) -> bytes | None:
        """
        :returns: the message, or None if it's the same version as the last time or if the shared memory is closed
        """
        lock_mutex(self._buffer)
        try:
            if self._get_closed():
                return None
            new_version = self._get_version()
            if new_version == self._last_read_version:
                return None
            self._last_read_version = new_version
            return self._get_bytes()
        finally:
            unlock_mutex(self._buffer)

    def blocking_read(self) -> bytes | None:
        """
        Keeps checking the shared memory until there is a new version to read,
        This function also releases the GIL, while waiting for a new message
        :returns: the message, or None if the shared memory is closed
        """
        while True:
            lock_mutex(self._buffer)
            try:
                if self._get_closed():
                    return None
                new_version = self._get_version()
                if new_version == self._last_read_version:
                    # Not a new version; release the lock and wait
                    pass
                else:
                    self._last_read_version = new_version
                    return self._get_bytes()
            finally:
                unlock_mutex(self._buffer)

            time.sleep(0.0005)

    def is_new_version_available(self) -> bool:
        """
        Check if the next read will return a new message
        :returns: true if there is a new version
        """
        lock_mutex(self._buffer)
        try:
            return self._get_version() != self._last_read_version
        finally:
            unlock_mutex(self._buffer)

    def last_written_version(self) -> int:
        """
        :returns: the latest version that was written
        """
        lock_mutex(self._buffer)
        try:
            return self._last_written_version
        finally:
            unlock_mutex(self._buffer)

    def last_read_version(self) -> int:
        """
        :returns: the latest version that was read
        """
        lock_mutex(self._buffer)
        try:
            return self._last_read_version
        finally:
            unlock_mutex(self._buffer)

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
        lock_mutex(self._buffer)
        try:
            return self._get_closed()
        finally:
            unlock_mutex(self._buffer)

    def close(self):
        """
        Close the shared memory block.
        """
        lock_mutex(self._buffer)
        try:
            self._set_closed(True)
        finally:
            unlock_mutex(self._buffer)

        # self._shm.close()

    def unlink(self):
        """
        Unlink (delete) the shared memory block. Use with caution.
        """
        self._shm.unlink()

    # ----------------- Field Access Methods -----------------

    def _initialize_fields(self, size: int):
        """
        Initialize the shared memory fields to default values.
        """
        self._set_closed(False)
        self._set_version(0)
        self._set_size(0)
        # Initialize bytes to zeros
        for i in range(size):
            self._buffer[BYTES_OFFSET + i] = 0

    def _get_closed(self) -> bool:
        """
        Get the value of `closed`.
        """
        return bool(self._shm.buf[CLOSED_OFFSET])

    def _set_closed(self, value: bool):
        """
        Set the value of `closed`.
        """
        self._shm.buf[CLOSED_OFFSET] = 1 if value else 0

    def _get_version(self) -> int:
        """
        Get the value of `version`.
        """
        return struct.unpack_from('Q', self._buffer, VERSION_OFFSET)[0]

    def _set_version(self, version: int):
        """
        Set the value of `version`.
        """
        struct.pack_into('Q', self._buffer, VERSION_OFFSET, version)

    def _increment_version(self) -> int:
        """
        Atomically increment the `version`.
        """
        version = self._get_version()
        version += 1
        self._set_version(version)
        return version

    def _get_size(self) -> int:
        """
        Get the value of `size`.
        """
        return struct.unpack_from('Q', self._buffer, SIZE_OFFSET)[0]

    def _set_size(self, size: int):
        """
        Set the value of `size`.
        """
        struct.pack_into('Q', self._buffer, SIZE_OFFSET, size)

    def _get_bytes(self) -> bytes:
        """
        Get a copy of the byte array.
        """
        size = self._get_size()
        return bytes(self._buffer[BYTES_OFFSET : BYTES_OFFSET + size])

    def _set_bytes(self, data: bytes) -> int:
        """
        Set the byte array without truncating or padding.
        Updates the `size` field to reflect the length of `data`.

        :param data: Byte data to write to shared memory.
        """
        data_length = len(data)
        # Ensure the shared memory has enough space
        if HEADER_SIZE + data_length > self._shm.size:
            raise ValueError("Not enough shared memory to write data.")

        new_version = self._increment_version()
        # Update the size field
        self._set_size(data_length)

        # Write data
        mv = memoryview(self._shm.buf)
        mv[BYTES_OFFSET : BYTES_OFFSET + data_length] = data
        return new_version

    # ----------------- Context Manager Support -----------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
if __name__ == "__main__":
    shm = SharedMemory.create("test", 1024)

    shm.write(b"Hello, World!")
    assert shm.is_new_version_available() is True
    assert shm.try_read() == b"Hello, World!"
    assert shm.last_written_version() == shm.last_read_version()
    assert shm.is_new_version_available() is False
    shm.write(b"Hello, World!2")
    assert shm.blocking_read() == b"Hello, World!2"
    assert shm.last_written_version() == shm.last_read_version()
    assert shm.last_read_version() == 2
    print("Shared memory test passed")
    shm.blocking_read()

    shm.close()
    shm.unlink()  # Only do this when you're sure no other process is using the shared memory