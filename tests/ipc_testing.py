import multiprocessing as mp
import time
import pickle

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from ripc import SharedMessage as SharedMemoryRust, OpenMode

from tests.shm_python_extra_lock import SharedMemory as SharedMemoryPythonWithExtraLock
from tests.shm_python_full import SharedMemory as SharedMemoryPython
from ripc import SharedQueue

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
        mock_pipe_data.send_time = time.perf_counter()
        pipe.send(pickle.dumps(mock_pipe_data, protocol=pickle.HIGHEST_PROTOCOL))
        i += 1
    pipe.close()


def mp_queue_writer_proc(queue, attempts, mock_pipe_data, received: mp.Value):
    """
    Writer process for multiprocessing Queue.
    Sends MockPipeData objects through the Queue `attempts` times.
    """
    for _ in range(attempts):
        mock_pipe_data.send_time = time.perf_counter()
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

    total_duration = 0.0
    for attempt in range(attempts):
        if attempt % 250 == 0:
            print(f"Processing attempt {attempt + 1}/{attempts}...")
        # Read serialized data from the pipe
        data_bytes = parent_conn.recv()
        recv_pipe_data: MockPipeData = pickle.loads(data_bytes)

        # Validate and record duration
        total_duration += time.perf_counter() - recv_pipe_data.send_time
        assert recv_pipe_data.data.size == data_size
        assert np.array_equal(mock_pipe_data.data, recv_pipe_data.data)

    writer_proc.join()
    avg_duration_ms = (
        total_duration / attempts
    ) * 1000  # Convert average to milliseconds
    return total_duration, avg_duration_ms, mock_pipe_data.size_human_readable


def mp_queue_perf_test(mock_pipe_data, data_size, attempts):
    """
    Performance test for multiprocessing Queue-based implementation.
    """
    queue = mp.Queue(maxsize=0)  # maxsize=0 means infinite size
    received = mp.Value("b", False)

    # Create the writer process
    writer_proc = mp.Process(
        target=mp_queue_writer_proc, args=(queue, attempts, mock_pipe_data, received)
    )
    writer_proc.start()

    total_duration = 0.0
    for attempt in range(attempts):
        if attempt % 250 == 0:
            print(f"Processing attempt {attempt + 1}/{attempts}...")
        # Read serialized data from the queue
        data_bytes = queue.get()  # This will block until data is available
        recv_pipe_data: MockPipeData = pickle.loads(data_bytes)

        # Validate and record duration
        total_duration += time.perf_counter() - recv_pipe_data.send_time
        assert recv_pipe_data.data.size == data_size, "Data size mismatch."
        assert np.array_equal(
            mock_pipe_data.data, recv_pipe_data.data
        ), "Data content mismatch."
        received.value = True

    writer_proc.join()  # Wait for the writer process to finish
    avg_duration_ms = (
        total_duration / attempts
    ) * 1000  # Convert average to milliseconds
    queue.close()  # Ensure the queue is properly closed
    queue.join_thread()  # Wait for the queue thread to finish

    return total_duration, avg_duration_ms, mock_pipe_data.size_human_readable


def ripc_shm_writer_proc(
    shm_impl,
    shm_impl_name,
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
    writer = shm_impl.open(shm_name, mode=OpenMode.WriteOnly)
    if lock is not None:
        writer._lock = lock

    i = 0
    while i < attempts:
        # Wait until the parent has read the last written version
        while writer.last_written_version() > last_read_version.value:
            continue

        mock_pipe_data.send_time = time.perf_counter()

        # Serialize MockPipeData and write to shared memory
        data_serialized = pickle.dumps(mock_pipe_data, protocol=pickle.HIGHEST_PROTOCOL)
        writer.write(data_serialized)
        i += 1

    # Ensure all writes are read before exiting
    time.sleep(0.1)
    if shm_impl_name != "shm_queue":
        while writer.last_written_version() != last_read_version.value:
            time.sleep(0.1)

    try:
        writer.close()
        writer.unlink()
    except Exception as e:
        pass


def ripc_shm_queue_writer_proc(shm_name, attempts, mock_pipe_data, received: mp.Value):
    """
    Child process:
    Writes MockPipeData objects to shared memory `attempts` times.
    Waits for the parent to read before writing again.
    """
    writer = SharedQueue.open(shm_name, mode=OpenMode.WriteOnly)

    i = 0
    while i < attempts:
        mock_pipe_data.send_time = time.perf_counter()
        print(f"Writing {i}th data")
        # Serialize MockPipeData and write to shared memory
        data_serialized = pickle.dumps(mock_pipe_data, protocol=pickle.HIGHEST_PROTOCOL)
        writer.write(data_serialized)
        print(f"Written {i}th data")
        i += 1

        while not received.value:
            continue
        received.value = False

    # Ensure all writes are read before exiting
    time.sleep(0.1)
    while not writer.is_closed():
        time.sleep(0.1)


def ripc_shm_queue_perf_test(
    mock_pipe_data, data_size, attempts, shm_name_base, method="blocking_read"
):
    try:
        """
        Runs the performance test using the specified method (`blocking_read` or `try_read`) with the chosen SHM implementation.
        """
        shm_name = generate_unique_shm_name(shm_name_base, "shm_queue")
        reader = SharedQueue.create(
            shm_name, mode=OpenMode.ReadOnly, max_element_size=data_size
        )

        received = mp.Value("b", False)

        # Start the writer process
        writer_proc = mp.Process(
            target=ripc_shm_queue_writer_proc,
            args=(shm_name, attempts, mock_pipe_data, received),
        )
        writer_proc.start()

        total_duration = 0.0
        for attempt in range(attempts):
            if attempt % 1 == 0:
                print(f"Processing attempt {attempt + 1}/{attempts}...")

            if method == "blocking_read":
                data_bytes = reader.blocking_read()
            elif method == "try_read":
                data_bytes = None
                while data_bytes is None and not reader.is_closed():
                    data_bytes = reader.try_read()
            else:
                raise ValueError(f"Unknown method: {method}")

            received.value = True

            if data_bytes is not None:
                recv_pipe_data: MockPipeData = pickle.loads(data_bytes)
                total_duration += time.perf_counter() - recv_pipe_data.send_time

                assert recv_pipe_data.data.size == data_size
                assert np.array_equal(mock_pipe_data.data, recv_pipe_data.data)

            writer_proc.join()

            avg_duration_ms = (
                total_duration / attempts
            ) * 1000  # Convert average to milliseconds
            return total_duration, avg_duration_ms, mock_pipe_data.size_human_readable
    except Exception as e:
        print(e)


def ripc_shm_perf_test(
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
    shm_size = data_size * 3 // 2
    shm_name = generate_unique_shm_name(shm_name_base, shm_impl_name)
    shm_impl = select_ipc_implementation(shm_impl_name)

    last_read_version = mp.Value("i", 0)

    # Create SharedMemoryReader
    reader = shm_impl.create(shm_name, shm_size, mode=OpenMode.ReadOnly)

    if shm_impl_name == "shm_lock":
        lock = reader._lock
    else:
        lock = None

    # Start the writer process
    writer_proc = mp.Process(
        target=ripc_shm_writer_proc,
        args=(
            shm_impl,
            shm_impl_name,
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

            total_duration += time.perf_counter() - recv_pipe_data.send_time

            assert recv_pipe_data.data.size == data_size
            assert np.array_equal(mock_pipe_data.data, recv_pipe_data.data)

        writer_proc.join()

        avg_duration_ms = (
            total_duration / attempts
        ) * 1000  # Convert average to milliseconds
        return total_duration, avg_duration_ms, mock_pipe_data.size_human_readable


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

            if ipc_name == "shm_queue":
                blocking_total, blocking_avg, data_size_human_readable = (
                    ripc_shm_queue_perf_test(
                        mock_pipe_data,
                        data_size,
                        attempts,
                        test_case["shm_name"],
                        method="blocking_read",
                    )
                )
                try_read_total, try_read_avg, _ = ripc_shm_queue_perf_test(
                    mock_pipe_data,
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

            # For SHM implementations
            blocking_total, blocking_avg, data_size_human_readable = ripc_shm_perf_test(
                mock_pipe_data,
                ipc_name,
                data_size,
                attempts,
                test_case["shm_name"],
                method="blocking_read",
            )
            try_read_total, try_read_avg, _ = ripc_shm_perf_test(
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

        results.append(result)

    return results


def select_ipc_implementation(implementation: str):
    """
    Select the shared memory implementation based on the provided name.
    """
    if implementation == "shm_ripc":
        return SharedMemoryRust
    elif implementation == "shm_lock":
        return SharedMemoryPythonWithExtraLock
    elif implementation == "shm_pth":
        return SharedMemoryPython
    elif implementation == "shm_queue":
        return SharedQueue
    else:
        raise ValueError(f"Unknown SHM implementation: {implementation}")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # Define test cases
    test_cases = [
        {"width": 640, "height": 480, "bpp": 3, "shm_name": "/shm_640x480"},
        {"width": 1280, "height": 720, "bpp": 3, "shm_name": "/shm_1280x720"},
        {"width": 1920, "height": 1080, "bpp": 3, "shm_name": "/shm_1920x1080"},
        {"width": 3840, "height": 2160, "bpp": 3, "shm_name": "/shm_3840x2160"},
    ]

    # Shared memory implementations to test
    ipc_to_test = [
        "shm_queue",
        "mp_pipe",
        "shm_ripc",
        "shm_lock",
        "shm_pth",
        "mp_queue",
    ]

    size_multiplier = 3
    attempts = 5000

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