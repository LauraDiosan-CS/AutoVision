import ctypes
import multiprocessing as mp
import pickle
import time

from ripc import SharedMemoryWriter, SharedMemoryReader, SharedMemoryCircularQueue
from config import Config
from controllers.controller import Controller
from helpers.controlled_process import ControlledProcess
from helpers.helpers import initialize_config
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from mockcameraprocess import MockCameraProcess


class MultiProcessingManager(ControlledProcess):
    __slots__ = ['keep_running']

    def __init__(self, program_start_time: float, keep_running: mp.Value, name=None):
        super().__init__(name=name, program_start_time=program_start_time)
        self.keep_running = keep_running

    def run(self):
        control_loop_pipe_writer = SharedMemoryWriter(name=Config.control_loop_memory_name, size=Config.pipe_memory_size)
        visualization_queue = SharedMemoryCircularQueue.create(Config.visualization_memory_name, Config.pipe_memory_size, Config.visualizer_queue_element_count)
        self.finish_setup()

        last_processed_frame_versions = [mp.Value(ctypes.c_int, 0) for _ in range(4)]
        start_video = mp.Value('b', False)

        camera_process = MockCameraProcess(start_video=start_video, keep_running=self.keep_running,
                                           last_processed_frame_versions=last_processed_frame_versions, program_start_time=self.program_start_time)
        camera_process.start()
        camera_process.wait_for_setup()

        print("Camera process started")

        pipelines, _, _ = initialize_config()

        parallel_processes = []

        for (index, pipeline) in enumerate(pipelines):
            match pipeline.name:
                case "lane_detection":
                    artificial_delay = 0.0
                case "sign_detection":
                    artificial_delay = 0.1
                case "traffic_light_detection":
                    artificial_delay = 0.15
                case "pedestrian_detection":
                    artificial_delay = 0.2
            # artificial_delay = 0.0

            process = SequentialFilterProcess(filters=pipeline.filters,
                                              keep_running=self.keep_running,
                                              last_processed_frame_version=last_processed_frame_versions[index],
                                              artificial_delay=artificial_delay,
                                              process_name=pipeline.name,
                                              program_start_time=self.program_start_time)
            process.start()
            parallel_processes.append(process)

        for process in parallel_processes:
            process.wait_for_setup()
        print("All SFP started")

        controller_process = Controller(self.keep_running)
        controller_process.start()
        print("Controller process started")

        shared_memory_readers = [SharedMemoryReader(name=pipeline.name) for pipeline in pipelines]

        current_pipe_data = PipeData(frame=None, frame_version=-1, depth_frame=None, unfiltered_frame=None, last_touched_process="None",
                                     creation_time=time.time())
        current_pipe_data.timing_info.start("Process Frame")

        start_video.value = True # All Setup finished allow the video reader to start

        is_full = False

        while self.keep_running.value:
            for reader in shared_memory_readers:
                pipe_data_bytes = reader.try_read()

                if pipe_data_bytes:

                    new_pipe_data: PipeData = pickle.loads(pipe_data_bytes)
                    new_pipe_data.timing_info.stop("Transfer Data (SPF -> MM)")
                    # print(f"New Pipe Data: {new_pipe_data.last_touched_process} {(time.perf_counter() - self.program_start_time):.2f} s")

                    new_pipe_data.timing_info.start("Merge Data",
                                                    parent=f"Data Lifecycle {new_pipe_data.last_touched_process}")

                    current_pipe_data.merge(new_pipe_data)

                    new_pipe_data.timing_info.stop("Merge Data")
                    current_pipe_data.timing_info.start("Transfer Data (MM -> Viz+)",
                                                        parent=f"Data Lifecycle {current_pipe_data.last_touched_process}")
                    # print(f"Timing_Info MM -> Viz: {current_pipe_data.timing_info}")
                    pickled_pipe_data = pickle.dumps(current_pipe_data, protocol=pickle.HIGHEST_PROTOCOL)

                    control_loop_pipe_writer.write(pickled_pipe_data)
                    visualization_queue.try_write(pickled_pipe_data)
                    current_pipe_data.timing_info.remove_recursive(
                        f"Data Lifecycle {current_pipe_data.last_touched_process}")

            if visualization_queue.is_full() and not is_full:
                print(f"!!! Visualization queue full with {len(visualization_queue)} elements !!!")
                is_full = True

        control_loop_pipe_writer.close()
        self.join_all_processes(controller_process, parallel_processes)

    @staticmethod
    def join_all_processes(controller_process, parallel_processes):
        print("Joining Controller process")
        controller_process.join()
        print("Controller process joined")

        print("Joining parallel processes")
        for process in parallel_processes:
            print(f"Joining {process.name}")
            process.terminate()
        print("All parallel processes joined")

        print("Exiting MultiProcessingManager")