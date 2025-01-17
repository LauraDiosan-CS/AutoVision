import multiprocessing as mp
import pickle
import struct
import time
import unittest
from dataclasses import dataclass

# For Python's built-in shared memory
from multiprocessing import shared_memory

import numpy as np
from ripc import SharedMemoryReader


@dataclass(slots=True)
class MiniPipeData:
    frame: np.ndarray
    frame_version: int
    unfiltered_frame: np.ndarray

###############################################################################
#                           RIPC SHM: Child Writer
###############################################################################
def shm_writer_proc(
    shm_name,
    shm_size,
    attempts,
    width,
    height,
    bytes_per_pixel,
    safe_to_start: mp.Value,
    last_read_version: mp.Value
):
    """
    Child process:
    Opens RIPC SharedMemoryWriter, writes `MiniPipeData` objects `attempts` times.
    It waits until the parent has read the last message (by comparing version)
    before writing again.
    """
    from ripc import SharedMemoryWriter  # re-import inside child
    writer = SharedMemoryWriter(shm_name, shm_size)
    safe_to_start.value = True

    frame_size = width * height * bytes_per_pixel
    i = 0
    while i < attempts:
        if writer.last_written_version() > last_read_version.value:
            # Means we've written something the parent hasn't read yet
            continue

        # Generate random frame + wrap in MiniPipeData
        frame = np.random.randint(0, 256, frame_size, dtype=np.uint8)
        data_obj = MiniPipeData(
            frame=frame,
            frame_version=i,
            unfiltered_frame=frame
        )
        # Serialize
        data_serialized = pickle.dumps(data_obj, protocol=pickle.HIGHEST_PROTOCOL)
        # Write to shared memory
        writer.write(data_serialized)
        i += 1

    while True:
        if writer.last_written_version() == last_read_version.value:
            break
        time.sleep(1)

###############################################################################
#                         mp.Pipe: Child Writer
###############################################################################
def pipe_writer_proc(child_conn, attempts, width, height, bytes_per_pixel):
    """
    Child process:
    Sends `MiniPipeData` objects `attempts` times through one end of an mp.Pipe.
    """
    frame_size = width * height * bytes_per_pixel

    for i in range(attempts):
        # Generate random frame + wrap in MiniPipeData
        frame = np.random.randint(0, 256, frame_size, dtype=np.uint8)
        data_obj = MiniPipeData(
            frame=frame,
            frame_version=i,
            unfiltered_frame=frame
        )
        data_serialized = pickle.dumps(data_obj, protocol=pickle.HIGHEST_PROTOCOL)
        # Send it
        child_conn.send(data_serialized)

    child_conn.close()

###############################################################################
#           Python Built-in SharedMemory: Child Writer
###############################################################################
def python_shm_writer_proc(
    shm_name: str,
    shm_size: int,
    attempts: int,
    width: int,
    height: int,
    bytes_per_pixel: int,
    safe_to_start: mp.Value,
    last_read_version: mp.Value,
    last_written_version: mp.Value
):
    """
    Child process:
    1) Creates a python shared_memory.SharedMemory(name=shm_name, create=True).
    2) Repeatedly writes a pickled `MiniPipeData` into that buffer.
       - The first 4 bytes are the length of the pickled data (little-endian).
       - The next `length` bytes are the data itself.
    3) Uses version counters (last_read_version, last_written_version) to avoid overwriting
       data the parent hasn't read yet.
    """
    shm_block = shared_memory.SharedMemory(name=shm_name, create=True, size=shm_size)
    safe_to_start.value = True

    frame_size = width * height * bytes_per_pixel
    buffer_np = np.ndarray((shm_size,), dtype=np.uint8, buffer=shm_block.buf)

    i = 0
    while i < attempts:
        if last_written_version.value > last_read_version.value:
            # Means we've written something not yet read by the parent
            continue

        # Generate random frame + wrap in MiniPipeData
        frame = np.random.randint(0, 256, frame_size, dtype=np.uint8)
        data_obj = MiniPipeData(
            frame=frame,
            frame_version=i,
            unfiltered_frame=frame
        )
        data_serialized = pickle.dumps(data_obj, protocol=pickle.HIGHEST_PROTOCOL)

        # Write length (4 bytes) + data into shared memory
        data_len = len(data_serialized)
        struct.pack_into('<I', buffer_np, 0, data_len)  # little-endian uint32
        buffer_np[4:4+data_len] = np.frombuffer(data_serialized, dtype=np.uint8)

        # Increment the "written" version
        last_written_version.value += 1
        i += 1

###############################################################################
#                       Actual Test Class
###############################################################################
class TestSharedMemoryVsMpPipe(unittest.TestCase):

    ###########################################################################
    #                           Utility for printing
    ###########################################################################
    def _print_performance_results(self, label, width, height, bpp, attempts, total_duration):
        avg_duration = total_duration / attempts
        total_ms = total_duration * 1000
        avg_ms   = avg_duration * 1000

        print(f"\n{label} Performance Results:")
        print(f"{'Parameter':<30} {'Value':>20}")
        print("-" * 50)
        print(f"{'Width':<30} {width:>20}")
        print(f"{'Height':<30} {height:>20}")
        print(f"{'Bytes per Pixel':<30} {bpp:>20}")
        print(f"{'Attempts':<30} {attempts:>20}")
        print("-" * 50)
        print(f"{'Total Duration (ms)':<30} {total_ms:>20.4f}")
        print(f"{'Avg Duration/Attempt (ms)':<30} {avg_ms:>20.6f}")
        print("-" * 50)

    ###########################################################################
    #          RIPC SHM test (2 processes) for MiniPipeData
    ###########################################################################
    def run_performance_test_pipe_data_shm_2procs(self, width, height, bytes_per_pixel, attempts, shm_name_base):
        """
        1) Spawn child to write `MiniPipeData` to RIPC shared memory in a loop.
        2) Parent reads using `try_read()`, measuring each read’s latency.
        """
        frame_size = width * height * bytes_per_pixel
        shm_size = frame_size * 3  # Enough for 3 frames if needed
        shm_name = f"{shm_name_base}_writer"

        safe_to_start = mp.Value('b', False)
        last_read_version = mp.Value('i', 0)

        # Start child process (RIPC writer)
        writer_proc = mp.Process(
            target=shm_writer_proc,
            args=(shm_name, shm_size, attempts, width, height, bytes_per_pixel, safe_to_start, last_read_version)
        )
        writer_proc.start()

        # Wait until the child has created the RIPC SharedMemoryWriter
        while not safe_to_start.value:
            time.sleep(0.01)

        # Parent: open the corresponding RIPC SharedMemoryReader
        reader = SharedMemoryReader(shm_name)

        total_duration = 0.0
        for _ in range(attempts):
            # V1
            time.sleep(0.1)  # give writer a chance to write more
            start_time = time.perf_counter()
            data_bytes = reader.blocking_read()
            last_read_version.value = reader.last_read_version()

            # V2
            # data_bytes = None
            # while data_bytes is None:
            #     start_time = time.perf_counter()
            #     data_bytes = reader.try_read()
            #     last_read_version.value = reader.last_read_version()

            # Deserialize to simulate usage
            _ = pickle.loads(data_bytes)
            total_duration += (time.perf_counter() - start_time)


        writer_proc.join()
        self._print_performance_results(
            label="MiniPipeData (RIPC SHM, 2 procs)",
            width=width,
            height=height,
            bpp=bytes_per_pixel,
            attempts=attempts,
            total_duration=total_duration
        )

    ###########################################################################
    #           mp.Pipe test (2 processes) for MiniPipeData
    ###########################################################################
    def run_performance_test_pipe_data_mp_pipe_2procs(self, width, height, bytes_per_pixel, attempts):
        """
        1) Spawn child to send `MiniPipeData` objects in a loop over mp.Pipe.
        2) Parent measures how long each recv() call takes.
        """
        parent_conn, child_conn = mp.Pipe(duplex=False)

        writer_proc = mp.Process(
            target=pipe_writer_proc,
            args=(child_conn, attempts, width, height, bytes_per_pixel)
        )
        writer_proc.start()

        total_duration = 0.0
        for _ in range(attempts):
            start_time = time.perf_counter()
            received_data = parent_conn.recv()  # blocks until data is available
            _ = pickle.loads(received_data)
            total_duration += (time.perf_counter() - start_time)

        writer_proc.join()
        parent_conn.close()
        self._print_performance_results(
            label="MiniPipeData (mp.Pipe, 2 procs)",
            width=width,
            height=height,
            bpp=bytes_per_pixel,
            attempts=attempts,
            total_duration=total_duration
        )

    ###########################################################################
    #     Python built-in shared_memory test (2 processes) for MiniPipeData
    ###########################################################################
    def run_performance_test_pysharedmem_2procs(self, width, height, bytes_per_pixel, attempts):
        """
        1) Spawn child that creates a python shared_memory.SharedMemory block
           and writes data attempts times.
        2) Parent attaches (create=False), reads data each time, measuring read + unpickle time.
        """
        frame_size = width * height * bytes_per_pixel
        # We'll store entire pickled data in the buffer, so let's pick a size:
        #   - upper bound for pickled data is ~some multiple of frame_size.
        # For safety, let's do 2-3x the frame_size. But let's be consistent with RIPC approach:
        shm_size = frame_size * 3
        shm_name = f"py_shm_{width}x{height}"

        safe_to_start = mp.Value('b', False)
        last_read_version = mp.Value('i', 0)
        last_written_version = mp.Value('i', 0)

        # Start the child that creates the shared memory (create=True)
        writer_proc = mp.Process(
            target=python_shm_writer_proc,
            args=(
                shm_name,
                shm_size,
                attempts,
                width,
                height,
                bytes_per_pixel,
                safe_to_start,
                last_read_version,
                last_written_version
            )
        )
        writer_proc.start()

        # Wait until child is ready
        while not safe_to_start.value:
            time.sleep(0.01)

        # Parent attaches to existing shared memory (create=False)
        shm_block = shared_memory.SharedMemory(name=shm_name, create=False)
        buffer_np = np.ndarray((shm_size,), dtype=np.uint8, buffer=shm_block.buf)

        total_duration = 0.0
        for i in range(attempts):
            start_time = time.perf_counter()

            # Wait until there's new data (written_version > read_version)
            # while last_written_version.value <= last_read_version.value:
            #     time.sleep(0.0001)  # quick poll

            # Read 4 bytes for length
            data_len = struct.unpack_from('<I', buffer_np, 0)[0]
            data_bytes = buffer_np[4:4+data_len].tobytes()
            _ = pickle.loads(data_bytes)

            last_read_version.value += 1
            total_duration += (time.perf_counter() - start_time)
            time.sleep(0.1)  # give writer a chance to write more

        writer_proc.join()
        shm_block.close()
        # The child created it, so child is responsible for unlink() or
        # you can do that here after .join() if you prefer:
        # shm_block.unlink()  # optional

        self._print_performance_results(
            label="MiniPipeData (Python shm, 2 procs)",
            width=width,
            height=height,
            bpp=bytes_per_pixel,
            attempts=attempts,
            total_duration=total_duration
        )

    ###########################################################################
    #                    MAIN TEST: SHM vs mp.Pipe (2 processes)
    ###########################################################################
    def test_shared_memory_vs_mp_pipe(self):
        """
        We only test MiniPipeData. For each test case, do:
          1) RIPC SharedMemory (2-proc) test
          2) mp.Pipe (2-proc) test
          3) Python built-in shared_memory (2-proc) test
        """
        test_cases = [
            {'width': 640,  'height': 480,  'bpp': 3, 'attempts': 500, 'shm_name': '/shm_640x480'},
            {'width': 1280, 'height': 720,  'bpp': 3, 'attempts': 500, 'shm_name': '/shm_1280x720'},
            {'width': 1920, 'height': 1080, 'bpp': 3, 'attempts': 500,  'shm_name': '/shm_1920x1080'},
            {'width': 3840, 'height': 2160, 'bpp': 3, 'attempts': 500,  'shm_name': '/shm_3840x2160'},
        ]

        for params in test_cases:
            w     = params['width']
            h     = params['height']
            bpp   = params['bpp']
            att   = params['attempts']
            shm_n = params['shm_name']

            # 1) RIPC SharedMemory (2 procs)
            with self.subTest(msg="MiniPipeData RIPC SHM 2-procs", params=params):
                self.run_performance_test_pipe_data_shm_2procs(w, h, bpp, att, shm_n)

            # 2) mp.Pipe (2 procs)
            with self.subTest(msg="MiniPipeData mp.Pipe 2-procs", params=params):
                self.run_performance_test_pipe_data_mp_pipe_2procs(w, h, bpp, att)

            # 3) Python shared_memory (2 procs)
            with self.subTest(msg="MiniPipeData Python shm 2-procs", params=params):
                self.run_performance_test_pysharedmem_2procs(w, h, bpp, att)


if __name__ == '__main__':
    unittest.main()