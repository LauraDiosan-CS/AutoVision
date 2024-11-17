import pickle
import unittest
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from ripc import SharedMemoryWriter, SharedMemoryReader

@dataclass(slots=True)
class MiniPipeData:
    frame: np.array
    frame_version: int
    unfiltered_frame: np.array

class TestSharedMemoryReadMethods(unittest.TestCase):
    def run_performance_test_pipe_data(self, width, height, bytes_per_pixel, attempts, shm_name_base):
        frame_size = width * height * bytes_per_pixel

        # Create shared memory writers and readers with explicit names
        shm_size = frame_size * 3 # Allocate enough memory for 3 frames (pipeData)

        # Writer and reader for try_read method
        name_try_read = f"{shm_name_base}_try_read_writer"
        writer_try_read = SharedMemoryWriter(name_try_read, shm_size)
        reader_try_read = SharedMemoryReader(name_try_read)

        # Writer and reader for try_read_into method
        name_try_read_into = f"{shm_name_base}_try_read_into_writer"
        writer_try_read_into = SharedMemoryWriter(name_try_read_into, shm_size)
        reader_try_read_into = SharedMemoryReader(name_try_read_into)

        # Prepare a contiguous and writable buffer for try_read_into method
        buffer = np.empty(shm_size+100, dtype=np.uint8)

        # Verify that the buffer is C-contiguous and writable
        self.assertTrue(buffer.flags['C_CONTIGUOUS'], "Buffer is not contiguous")
        self.assertTrue(buffer.flags['WRITEABLE'], "Buffer is not writable")

        total_try_read_duration = 0.0
        total_try_read_into_duration = 0.0

        for _ in range(attempts):
            # Generate random frames
            frame = np.random.randint(0, 256, frame_size, dtype=np.uint8)

            pipeData = MiniPipeData(frame=frame, frame_version=0, unfiltered_frame=frame)
            pipeData = pickle.dumps(pipeData, protocol=pickle.HIGHEST_PROTOCOL)
            pipeData_copy = deepcopy(pipeData)

            # Write frames to shared memory
            writer_try_read.write(pipeData)
            writer_try_read_into.write(pipeData_copy)

            # Measure time for try_read_into method
            start_time = time.perf_counter()

            bytes_read = reader_try_read_into.try_read_into(buffer)
            pipeData = pickle.loads(buffer[:bytes_read])

            try_read_into_duration = time.perf_counter() - start_time
            self.assertGreater(bytes_read, 0, "Failed to read frame into buffer")
            total_try_read_into_duration += try_read_into_duration

            # Measure time for try_read method
            start_time = time.perf_counter()

            read_frame = reader_try_read.try_read()
            pipeData = pickle.loads(read_frame)

            try_read_duration = time.perf_counter() - start_time
            self.assertIsNotNone(read_frame, "Failed to read frame using try_read")
            total_try_read_duration += try_read_duration

        # Calculate average durations
        avg_try_read_duration = total_try_read_duration / attempts
        avg_try_read_into_duration = total_try_read_into_duration / attempts

        # Convert durations to milliseconds
        total_try_read_duration_ms = total_try_read_duration * 1000
        avg_try_read_duration_ms = avg_try_read_duration * 1000
        total_try_read_into_duration_ms = total_try_read_into_duration * 1000
        avg_try_read_into_duration_ms = avg_try_read_into_duration * 1000

        # Print the performance results in a table format
        print("\nPipeData Performance Results:")
        print(f"{'Parameter':<30} {'Value':>20}")
        print("-" * 50)
        print(f"{'Width':<30} {width:>20}")
        print(f"{'Height':<30} {height:>20}")
        print(f"{'Bytes per Pixel':<30} {bytes_per_pixel:>20}")
        print(f"{'Attempts':<30} {attempts:>20}")
        print("-" * 50)
        print(f"{'':<35} {'try_read':>15} {'try_read_into':>20} {'Difference':>15}")
        print("-" * 75)
        print(f"{'Total Duration (ms)':<35} {total_try_read_duration_ms:>15.4f} {total_try_read_into_duration_ms:>20.4f} {total_try_read_duration_ms - total_try_read_into_duration_ms:>15.4f}")
        print(f"{'Average Duration per Attempt (ms)':<35} {avg_try_read_duration_ms:>15.6f} {avg_try_read_into_duration_ms:>20.6f} {avg_try_read_duration_ms - avg_try_read_into_duration_ms:>15.6f}")
        print("-" * 75)

        # Determine which method is faster
        if avg_try_read_duration_ms < avg_try_read_into_duration_ms:
            diff = avg_try_read_into_duration_ms - avg_try_read_duration_ms
            faster_method = f"try_read is faster by {diff:.6f} ms per attempt."
        else:
            diff = avg_try_read_duration_ms - avg_try_read_into_duration_ms
            faster_method = f"try_read_into is faster by {diff:.6f} ms per attempt."

        print(f"{'Result':<30} {faster_method}")
        # Clean up
        writer_try_read.close()
        writer_try_read_into.close()

    def run_performance_test_frame(self, width, height, bytes_per_pixel, attempts, shm_name_base):
        frame_size = width * height * bytes_per_pixel

        # Create shared memory writers and readers with explicit names
        shm_size = frame_size

        # Writer and reader for try_read method
        name_try_read = f"{shm_name_base}_try_read_writer"
        writer_try_read = SharedMemoryWriter(name_try_read, shm_size)
        reader_try_read = SharedMemoryReader(name_try_read)

        # Writer and reader for try_read_into method
        name_try_read_into = f"{shm_name_base}_try_read_into_writer"
        writer_try_read_into = SharedMemoryWriter(name_try_read_into, shm_size)
        reader_try_read_into = SharedMemoryReader(name_try_read_into)

        # Prepare a contiguous and writable buffer for try_read_into method
        buffer = np.empty(shm_size+100, dtype=np.uint8)

        # Verify that the buffer is C-contiguous and writable
        self.assertTrue(buffer.flags['C_CONTIGUOUS'], "Buffer is not contiguous")
        self.assertTrue(buffer.flags['WRITEABLE'], "Buffer is not writable")

        total_try_read_duration = 0.0
        total_try_read_into_duration = 0.0

        for _ in range(attempts):
            # Generate random frames
            frame = np.random.randint(0, 256, frame_size, dtype=np.uint8).tobytes()
            frame_copy = deepcopy(frame)

            # Write frames to shared memory
            writer_try_read.write(frame)
            writer_try_read_into.write(frame_copy)

            # Measure time for try_read_into method
            start_time = time.perf_counter()

            bytes_read = reader_try_read_into.try_read_into(buffer)
            array = buffer[:bytes_read].reshape((height, width, bytes_per_pixel))

            try_read_into_duration = time.perf_counter() - start_time
            self.assertGreater(bytes_read, 0, "Failed to read frame into buffer")
            total_try_read_into_duration += try_read_into_duration

            # Measure time for try_read method
            start_time = time.perf_counter()

            read_frame = reader_try_read.try_read()
            array = np.frombuffer(read_frame, dtype=np.uint8).reshape((height, width, bytes_per_pixel))

            try_read_duration = time.perf_counter() - start_time
            self.assertIsNotNone(read_frame, "Failed to read frame using try_read")
            total_try_read_duration += try_read_duration

        # Calculate average durations
        avg_try_read_duration = total_try_read_duration / attempts
        avg_try_read_into_duration = total_try_read_into_duration / attempts

        # Convert durations to milliseconds
        total_try_read_duration_ms = total_try_read_duration * 1000
        avg_try_read_duration_ms = avg_try_read_duration * 1000
        total_try_read_into_duration_ms = total_try_read_into_duration * 1000
        avg_try_read_into_duration_ms = avg_try_read_into_duration * 1000

        # Print the performance results in a table format
        print("\nFrame Performance Results:")
        print(f"{'Parameter':<30} {'Value':>20}")
        print("-" * 50)
        print(f"{'Width':<30} {width:>20}")
        print(f"{'Height':<30} {height:>20}")
        print(f"{'Bytes per Pixel':<30} {bytes_per_pixel:>20}")
        print(f"{'Attempts':<30} {attempts:>20}")
        print("-" * 50)
        print(f"{'':<35} {'try_read':>15} {'try_read_into':>20} {'Difference':>15}")
        print("-" * 75)
        print(f"{'Total Duration (ms)':<35} {total_try_read_duration_ms:>15.4f} {total_try_read_into_duration_ms:>20.4f} {total_try_read_duration_ms - total_try_read_into_duration_ms:>15.4f}")
        print(f"{'Average Duration per Attempt (ms)':<35} {avg_try_read_duration_ms:>15.6f} {avg_try_read_into_duration_ms:>20.6f} {avg_try_read_duration_ms - avg_try_read_into_duration_ms:>15.6f}")
        print("-" * 75)

        # Determine which method is faster
        if avg_try_read_duration_ms < avg_try_read_into_duration_ms:
            diff = avg_try_read_into_duration_ms - avg_try_read_duration_ms
            faster_method = f"try_read is faster by {diff:.6f} ms per attempt."
        else:
            diff = avg_try_read_duration_ms - avg_try_read_into_duration_ms
            faster_method = f"try_read_into is faster by {diff:.6f} ms per attempt."

        print(f"{'Result':<35} {faster_method}")

        # Clean up
        writer_try_read.close()
        writer_try_read_into.close()

    def test_shared_memory_read_methods(self):
        # Define different test cases with various parameters
        test_cases = [
            {'width': 640, 'height': 480, 'bytes_per_pixel': 3, 'attempts': 10000, 'shm_name_base': '/test_shm_480p'},
            {'width': 1280, 'height': 720, 'bytes_per_pixel': 3, 'attempts': 10000, 'shm_name_base': '/test_shm_720p'},
            {'width': 1920, 'height': 1080, 'bytes_per_pixel': 3, 'attempts': 10000, 'shm_name_base': '/test_shm_1080p'},
            {'width': 3840, 'height': 2160, 'bytes_per_pixel': 3, 'attempts': 5000, 'shm_name_base': '/test_shm_4k'},
        ]

        for params in test_cases:
            with self.subTest(params=params):
                self.run_performance_test_pipe_data(
                    width=params['width'],
                    height=params['height'],
                    bytes_per_pixel=params['bytes_per_pixel'],
                    attempts=params['attempts'],
                    shm_name_base=params['shm_name_base']
                )

            with self.subTest(params=params):
                self.run_performance_test_frame(
                    width=params['width'],
                    height=params['height'],
                    bytes_per_pixel=params['bytes_per_pixel'],
                    attempts=params['attempts'],
                    shm_name_base=params['shm_name_base']
                )

# PipeData Performance Results:
# Parameter                                     Value
# --------------------------------------------------
# Width                                           640
# Height                                          480
# Bytes per Pixel                                   3
# Attempts                                      10000
# --------------------------------------------------
#                                            try_read        try_read_into      Difference
# ---------------------------------------------------------------------------
# Total Duration (ms)                        840.3921             894.4493        -54.0572
# Average Duration per Attempt (ms)          0.084039             0.089445       -0.005406
# ---------------------------------------------------------------------------
# Result                         try_read is faster by 0.005406 ms per attempt.
#
# Frame Performance Results:
# Parameter                                     Value
# --------------------------------------------------
# Width                                           640
# Height                                          480
# Bytes per Pixel                                   3
# Attempts                                      10000
# --------------------------------------------------
#                                            try_read        try_read_into      Difference
# ---------------------------------------------------------------------------
# Total Duration (ms)                        409.9131             400.3449          9.5682
# Average Duration per Attempt (ms)          0.040991             0.040034        0.000957
# ---------------------------------------------------------------------------
# Result                              try_read_into is faster by 0.000957 ms per attempt.
#
# PipeData Performance Results:
# Parameter                                     Value
# --------------------------------------------------
# Width                                          1280
# Height                                          720
# Bytes per Pixel                                   3
# Attempts                                      10000
# --------------------------------------------------
#                                            try_read        try_read_into      Difference
# ---------------------------------------------------------------------------
# Total Duration (ms)                       4283.5423            4210.6382         72.9040
# Average Duration per Attempt (ms)          0.428354             0.421064        0.007290
# ---------------------------------------------------------------------------
# Result                         try_read_into is faster by 0.007290 ms per attempt.
#
# Frame Performance Results:
# Parameter                                     Value
# --------------------------------------------------
# Width                                          1280
# Height                                          720
# Bytes per Pixel                                   3
# Attempts                                      10000
# --------------------------------------------------
#                                            try_read        try_read_into      Difference
# ---------------------------------------------------------------------------
# Total Duration (ms)                       1600.6843            1815.8669       -215.1825
# Average Duration per Attempt (ms)          0.160068             0.181587       -0.021518
# ---------------------------------------------------------------------------
# Result                              try_read is faster by 0.021518 ms per attempt.
#
# PipeData Performance Results:
# Parameter                                     Value
# --------------------------------------------------
# Width                                          1920
# Height                                         1080
# Bytes per Pixel                                   3
# Attempts                                      10000
# --------------------------------------------------
#                                            try_read        try_read_into      Difference
# ---------------------------------------------------------------------------
# Total Duration (ms)                      11815.5526           10047.3625       1768.1901
# Average Duration per Attempt (ms)          1.181555             1.004736        0.176819
# ---------------------------------------------------------------------------
# Result                         try_read_into is faster by 0.176819 ms per attempt.
#
# Frame Performance Results:
# Parameter                                     Value
# --------------------------------------------------
# Width                                          1920
# Height                                         1080
# Bytes per Pixel                                   3
# Attempts                                      10000
# --------------------------------------------------
#                                            try_read        try_read_into      Difference
# ---------------------------------------------------------------------------
# Total Duration (ms)                       5859.9623            5400.0886        459.8736
# Average Duration per Attempt (ms)          0.585996             0.540009        0.045987
# ---------------------------------------------------------------------------
# Result                              try_read_into is faster by 0.045987 ms per attempt.
#
# PipeData Performance Results:
# Parameter                                     Value
# --------------------------------------------------
# Width                                          3840
# Height                                         2160
# Bytes per Pixel                                   3
# Attempts                                       5000
# --------------------------------------------------
#                                            try_read        try_read_into      Difference
# ---------------------------------------------------------------------------
# Total Duration (ms)                      13590.6982           13106.5288        484.1693
# Average Duration per Attempt (ms)          2.718140             2.621306        0.096834
# ---------------------------------------------------------------------------
# Result                         try_read_into is faster by 0.096834 ms per attempt.
#
# Frame Performance Results:
# Parameter                                     Value
# --------------------------------------------------
# Width                                          3840
# Height                                         2160
# Bytes per Pixel                                   3
# Attempts                                       5000
# --------------------------------------------------
#                                            try_read        try_read_into      Difference
# ---------------------------------------------------------------------------
# Total Duration (ms)                       7151.5106            7106.1712         45.3394
# Average Duration per Attempt (ms)          1.430302             1.421234        0.009068
# ---------------------------------------------------------------------------
# Result                              try_read_into is faster by 0.009068 ms per attempt.

if __name__ == '__main__':
    unittest.main()