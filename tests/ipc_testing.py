import multiprocessing as mp
import time
import pickle

from rs_ipc import SharedMessage, OperationMode, ReaderWaitPolicy
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from tests.shm_python_extra_lock import SharedMemory as SharedMemoryLock
from tests.shm_python_full import SharedMemory as SharedMemoryPth

import polars as pl  # Added Polars import


@dataclass(slots=True)
class MockPipeData:
    data: np.ndarray
    send_time: float

    @property
    def size_human_readable(self) -> str:
        """Returns the size in a human-readable format (GB, MB, KB, Bytes)."""
        size = self.data.nbytes
        for unit in ["Bytes", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"  # Fallback for extreme sizes


def generate_unique_shm_name(base_name, impl_name):
    """
    Generate a unique shared memory name by appending the implementation name and timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{base_name}_{impl_name}_{timestamp}"


def mp_pipe_writer_proc(pipe, attempts, mock_pipe_data):
    """
    Writer process for multiprocessing Pipe.
    Sends MockPipeData objects through the Pipe `attempts` times.
    """
    i = 0
    while i < attempts:
        mock_pipe_data.send_time = time.time_ns()
        pipe.send(pickle.dumps(mock_pipe_data, protocol=pickle.HIGHEST_PROTOCOL))
        i += 1
    pipe.close()


def mp_queue_writer_proc(queue, attempts, mock_pipe_data, received: mp.Value):
    """
    Writer process for multiprocessing Queue.
    Sends MockPipeData objects through the Queue `attempts` times.
    """
    for _ in range(attempts):
        mock_pipe_data.send_time = time.time_ns()
        queue.put(pickle.dumps(mock_pipe_data, protocol=pickle.HIGHEST_PROTOCOL))
        while not received.value:
            continue
        received.value = False
    queue.close()  # Close the queue to indicate no more data will be sent


def mp_pipe_perf_test(mock_pipe_data, data_size, attempts):
    """
    Performance test for multiprocessing Pipe-based implementation.
    """
    parent_conn, child_conn = mp.Pipe()

    # Create the writer process
    writer_proc = mp.Process(
        target=mp_pipe_writer_proc, args=(child_conn, attempts, mock_pipe_data)
    )
    writer_proc.start()

    total_duration_ns = 0.0
    for attempt in range(attempts):
        if attempt % 250 == 0:
            print(f"Processing attempt {attempt + 1}/{attempts}...")
        # Read serialized data from the pipe
        data_bytes = parent_conn.recv()
        recv_pipe_data: MockPipeData = pickle.loads(data_bytes)

        # Validate and record duration
        total_duration_ns += time.time_ns() - recv_pipe_data.send_time
        assert recv_pipe_data.data.size == data_size
        assert np.array_equal(mock_pipe_data.data, recv_pipe_data.data)

    writer_proc.join()
    avg_duration_ms = (
        total_duration_ns / attempts
    ) / 1e6  # Convert average to milliseconds
    return total_duration_ns / 1e9, avg_duration_ms, mock_pipe_data.size_human_readable


def mp_queue_perf_test(mock_pipe_data, data_size, attempts):
    """
    Performance test for multiprocessing Queue-based implementation.
    """
    queue = mp.Queue(maxsize=10)
    received = mp.Value("b", False)

    # Create the writer process
    writer_proc = mp.Process(
        target=mp_queue_writer_proc, args=(queue, attempts, mock_pipe_data, received)
    )
    writer_proc.start()

    total_duration_ns = 0.0
    for attempt in range(attempts):
        if attempt % 250 == 0:
            print(f"Processing attempt {attempt + 1}/{attempts}...")

        # Read serialized data from the queue
        data_bytes = queue.get()  # This will block until data is available
        recv_pipe_data: MockPipeData = pickle.loads(data_bytes)

        # Validate and record duration
        total_duration_ns += time.time_ns() - recv_pipe_data.send_time
        assert recv_pipe_data.data.size == data_size, "Data size mismatch."
        assert np.array_equal(
            mock_pipe_data.data, recv_pipe_data.data
        ), "Data content mismatch."
        received.value = True

    writer_proc.join()  # Wait for the writer process to finish
    avg_duration_ms = (
        total_duration_ns / attempts
    ) / 1e6  # Convert average to milliseconds
    queue.close()  # Ensure the queue is properly closed
    queue.join_thread()  # Wait for the queue thread to finish

    return total_duration_ns / 1e9, avg_duration_ms, mock_pipe_data.size_human_readable


def rs_ipc_shm_writer_proc(
    shm_name,
    op_mode,
    attempts,
    mock_pipe_data,
):
    """
    Child process:
    Writes MockPipeData objects to shared memory `attempts` times.
    Waits for the parent to read before writing again.
    """
    writer = SharedMessage.open(
        shm_name,
        (
            OperationMode.WriteSync(ReaderWaitPolicy.All())
            if op_mode == "SYNC"
            else OperationMode.WriteAsync(ReaderWaitPolicy.All())
        ),
    )

    i = 0
    while i < attempts:
        mock_pipe_data.send_time = time.time_ns()
        # Serialize MockPipeData and write to shared memory
        data_serialized = pickle.dumps(mock_pipe_data, protocol=pickle.HIGHEST_PROTOCOL)
        writer.write(data_serialized)
        i += 1

    writer.stop()


def py_ipc_shm_writer_proc(
    shm_impl,
    shm_name,
    attempts,
    mock_pipe_data,
    last_read_version,
    lock: mp.Lock = None,
):
    """
    Child process:
    Writes MockPipeData objects to shared memory `attempts` times.
    Waits for the parent to read before writing again.
    """
    writer = shm_impl.open(shm_name, mode=OperationMode.WriteSync)
    if lock is not None:
        writer._lock = lock

    i = 0
    while i < attempts:
        # Wait until the parent has read the last written version
        mock_pipe_data.send_time = time.time_ns()

        while writer.last_written_version() > last_read_version.value:
            continue

        # Serialize MockPipeData and write to shared memory
        data_serialized = pickle.dumps(mock_pipe_data, protocol=pickle.HIGHEST_PROTOCOL)
        writer.write(data_serialized)
        i += 1

    # Ensure all writes are read before exiting
    while writer.last_written_version() != last_read_version.value:
        continue

    writer.close()
    writer.unlink()


def rs_ipc_shm_perf_test(
    mock_pipe_data,
    shm_impl_name,
    writer_op_mode,
    data_size,
    attempts,
    shm_name_base,
    method="blocking_read",
):
    """
    Runs the performance test using the specified method (`blocking_read` or `try_read`) with the chosen SHM implementation.
    """
    shm_size = int(data_size * 11 / 10)
    shm_name = generate_unique_shm_name(shm_name_base, shm_impl_name)

    last_read_version = mp.Value("i", 0)

    # Create SharedMemoryReader
    reader = SharedMessage.create(shm_name, shm_size, OperationMode.ReadSync())

    # Start the writer process
    writer_proc = mp.Process(
        target=rs_ipc_shm_writer_proc,
        args=(
            shm_name,
            "SYNC" if writer_op_mode == OperationMode.WriteSync else "ASYNC",
            attempts,
            mock_pipe_data,
        ),
    )
    writer_proc.start()

    total_duration = 0.0
    for attempt in range(attempts):
        if attempt % 250 == 0:
            print(f"Processing attempt {attempt + 1}/{attempts}...")

        if method == "blocking_read":
            data_bytes = reader.read(block=True)
        elif method == "try_read":
            data_bytes = None
            while data_bytes is None and not reader.is_stopped():
                data_bytes = reader.read(block=False)
        else:
            raise ValueError(f"Unknown method: {method}")

        if data_bytes is not None:
            last_read_version.value = reader.last_read_version()

            recv_pipe_data: MockPipeData = pickle.loads(data_bytes)

            total_duration += time.time_ns() - recv_pipe_data.send_time

            assert recv_pipe_data.data.size == data_size
            assert np.array_equal(mock_pipe_data.data, recv_pipe_data.data)

    writer_proc.join()

    avg_duration_ms = (
        total_duration / attempts
    ) / 1e6  # Convert average to milliseconds
    return total_duration / 1e9, avg_duration_ms, mock_pipe_data.size_human_readable


def py_ipc_shm_perf_test(
    mock_pipe_data,
    shm_impl_name,
    data_size,
    attempts,
    shm_name_base,
    method="blocking_read",
):
    """
    Runs the performance test using the specified method (`blocking_read` or `try_read`) with the chosen SHM implementation.
    """
    shm_size = int(data_size * 11 / 10)
    shm_name = generate_unique_shm_name(shm_name_base, shm_impl_name)

    last_read_version = mp.Value("i", 0)

    if shm_impl_name == "shm_lock":
        shm_impl = SharedMemoryLock
    else:
        shm_impl = SharedMemoryPth

    # Create SharedMemoryReader
    reader = shm_impl.create(shm_name, shm_size, mode=OperationMode.ReadSync)

    lock = None
    if shm_impl_name == "shm_lock":
        lock = reader._lock

    # Start the writer process
    writer_proc = mp.Process(
        target=py_ipc_shm_writer_proc,
        args=(
            shm_impl,
            shm_name,
            attempts,
            mock_pipe_data,
            last_read_version,
            lock,
        ),
    )
    writer_proc.start()

    total_duration = 0.0
    for attempt in range(attempts):
        if attempt % 250 == 0:
            print(f"Processing attempt {attempt + 1}/{attempts}...")

        if method == "blocking_read":
            data_bytes = reader.blocking_read()
        elif method == "try_read":
            data_bytes = None
            while data_bytes is None and not reader.is_closed():
                data_bytes = reader.try_read()
        else:
            raise ValueError(f"Unknown method: {method}")

        if data_bytes is not None:
            last_read_version.value = reader.last_read_version()

            recv_pipe_data: MockPipeData = pickle.loads(data_bytes)

            total_duration += time.time_ns() - recv_pipe_data.send_time

            assert recv_pipe_data.data.size == data_size
            assert np.array_equal(mock_pipe_data.data, recv_pipe_data.data)

    writer_proc.join()

    avg_duration_ms = (
        total_duration / attempts
    ) / 1e6  # Convert average to milliseconds
    return total_duration / 1e9, avg_duration_ms, mock_pipe_data.size_human_readable


def compare_methods(test_cases, ipc_to_test, size_multiplier, attempts=500):
    """
    Compares different IPC methods for the given test cases.
    Returns a list of dictionaries, each representing a test case with all IPC methods' metrics.
    """
    results = []
    np.random.seed(42)
    for test_case in test_cases:
        width = test_case["width"]
        height = test_case["height"]
        bpp = test_case["bpp"]

        # Calculate MockPipeData size with multiplier
        frame_size = width * height * bpp
        data_size = frame_size * size_multiplier

        mock_pipe_data = MockPipeData(
            data=np.random.randint(0, 255, data_size, dtype=np.uint8), send_time=0.0
        )

        # Initialize the result dictionary for this test case
        result = {
            "Resolution": f"{width}x{height}",
            "PipeData Size": MockPipeData(
                data=np.random.randint(0, 255, data_size, dtype=np.uint8), send_time=0.0
            ).size_human_readable,
        }

        for ipc_name in ipc_to_test:
            print(f"Running test for {width}x{height} using {ipc_name}...")
            if ipc_name == "mp_pipe":
                total, avg, data_size_human_readable = mp_pipe_perf_test(
                    mock_pipe_data, data_size, attempts
                )
                # Add metrics with IPC method as prefix
                result[f"{ipc_name} Blocking Avg (ms)"] = avg
                result[f"{ipc_name} Try Avg (ms)"] = -1.0  # Not applicable
                result[f"{ipc_name} Blocking Total (s)"] = total
                result[f"{ipc_name} Try Total (s)"] = -1.0
                continue

            if ipc_name == "mp_queue":
                total, avg, data_size_human_readable = mp_queue_perf_test(
                    mock_pipe_data, data_size, attempts
                )
                # Add metrics with IPC method as prefix
                result[f"{ipc_name} Blocking Avg (ms)"] = avg
                result[f"{ipc_name} Try Avg (ms)"] = -1.0  # Not applicable
                result[f"{ipc_name} Blocking Total (s)"] = total
                result[f"{ipc_name} Try Total (s)"] = -1.0
                continue

            # For SHM implementations
            if ipc_name == "shm_lock" or ipc_name == "shm_pth":
                blocking_total, blocking_avg, data_size_human_readable = (
                    py_ipc_shm_perf_test(
                        mock_pipe_data,
                        ipc_name,
                        data_size,
                        attempts,
                        test_case["shm_name"],
                        method="blocking_read",
                    )
                )
                try_read_total, try_read_avg, _ = py_ipc_shm_perf_test(
                    mock_pipe_data,
                    ipc_name,
                    data_size,
                    attempts,
                    test_case["shm_name"],
                    method="try_read",
                )
                # Add metrics with IPC method as prefix
                result[f"{ipc_name} Blocking Avg (ms)"] = blocking_avg
                result[f"{ipc_name} Try Avg (ms)"] = try_read_avg
                result[f"{ipc_name} Blocking Total (s)"] = blocking_total
                result[f"{ipc_name} Try Total (s)"] = try_read_total
                continue

            writer_op_mode = OperationMode.WriteAsync
            blocking_total, blocking_avg, data_size_human_readable = (
                rs_ipc_shm_perf_test(
                    mock_pipe_data,
                    ipc_name,
                    writer_op_mode,
                    data_size,
                    attempts,
                    test_case["shm_name"],
                    method="blocking_read",
                )
            )
            try_read_total, try_read_avg, _ = rs_ipc_shm_perf_test(
                mock_pipe_data,
                ipc_name,
                writer_op_mode,
                data_size,
                attempts,
                test_case["shm_name"],
                method="try_read",
            )
            # Add metrics with IPC method as prefix
            result[f"{ipc_name} Blocking Avg (ms)"] = blocking_avg
            result[f"{ipc_name} Try Avg (ms)"] = try_read_avg
            result[f"{ipc_name} Blocking Total (s)"] = blocking_total
            result[f"{ipc_name} Try Total (s)"] = try_read_total

        results.append(result)

    return results


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # Define test cases
    test_cases = [
        {"width": 256, "height": 256, "bpp": 3, "shm_name": "/shm_256x256"},
        {"width": 512, "height": 512, "bpp": 3, "shm_name": "/shm_512x512"},
        {"width": 640, "height": 480, "bpp": 3, "shm_name": "/shm_640x480"},
        {"width": 1280, "height": 720, "bpp": 3, "shm_name": "/shm_1280x720"},
        {"width": 1920, "height": 1080, "bpp": 3, "shm_name": "/shm_1920x1080"},
    ]

    # Shared memory implementations to test
    ipc_to_test = [
        "mp_pipe",
        # "mp_queue",
        "shm_lock",
        "shm_pth",
        "shm_rs",
    ]

    size_multiplier = 3
    attempts = 5

    results = compare_methods(
        test_cases, ipc_to_test, size_multiplier, attempts=attempts
    )

    all_results_df = pl.DataFrame(results)

    # Create the summary table using Polars
    def format_to_two_decimals(df):
        for col in df.columns:
            if df[col].dtype in [
                pl.Float32,
                pl.Float64,
            ]:  # Check if the column is a float
                df = df.with_columns(
                    df[col].map_elements(lambda x: round(x, 3))  # Apply rounding
                )
        return df

    all_results_df = format_to_two_decimals(all_results_df)

    # Reorganize and group metrics
    metric_groups = [
        "Blocking Avg (ms)",
        "Try Avg (ms)",
        "Blocking Total (s)",
        "Try Total (s)",
    ]

    # Start with the common columns
    common_columns = ["Resolution", "PipeData Size"]

    column_order = common_columns + [
        f"{ipc_name} {metric}" for metric in metric_groups for ipc_name in ipc_to_test
    ]

    # Reorder the DataFrame
    all_results_df = all_results_df.select(column_order)

    # Iterate over metric groups and display results for each
    for metric in metric_groups:
        metric_columns = common_columns + [
            f"{ipc_name} {metric}" for ipc_name in ipc_to_test
        ]

        metric_df = all_results_df.select(metric_columns)

        print(f"\nPerformance Comparison for {metric.upper()}:")
        print(metric_df)

    all_results_df.write_csv(f"BenchmarkResults_{size_multiplier}_{attempts}.csv")

    print("\nSummary Performance Table:")
    print(all_results_df)