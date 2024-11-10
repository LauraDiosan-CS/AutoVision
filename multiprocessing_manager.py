import ctypes
import multiprocessing as mp
import pickle
import time
from random import random

import numpy as np

from ripc import SharedMemoryWriter, SharedMemoryReader
from config import Config
from controllers.controller import Controller
from helpers.controlled_process import ControlledProcess
from helpers.helpers import initialize_config
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from videoreaderprocess import VideoReaderProcess


class MultiProcessingManager(ControlledProcess):
    __slots__ = ['keep_running']

    def __init__(self, keep_running: mp.Value, name=None):
        super().__init__(name=name)
        self.keep_running = keep_running

    def run(self):
        composite_pipe_writer = SharedMemoryWriter(name=Config.composite_pipe_memory_name, size=Config.pipe_memory_size)
        self.finish_setup()

        last_processed_frame_versions = [mp.Value(ctypes.c_int, 0) for _ in range(4)]
        start_video = mp.Value('b', False)

        video_reader_process = VideoReaderProcess(start_video=start_video, keep_running=self.keep_running, last_processed_frame_versions=last_processed_frame_versions)
        video_reader_process.start()
        video_reader_process.wait_for_setup()

        pipelines, _, _ = initialize_config()

        parallel_processes = []

        for (index, pipeline) in enumerate(pipelines):
            match pipeline.name:
                case "lane_detection":
                    artificial_delay = 0.0
                case "sign_detection":
                    artificial_delay = 0.1
                case "traffic_light_detection":
                    artificial_delay = 0.2
                case "pedestrian_detection":
                    artificial_delay = 0.3

            process = SequentialFilterProcess(filters=pipeline.filters,
                                              keep_running=self.keep_running,
                                              last_processed_frame_version=last_processed_frame_versions[index],
                                              artificial_delay=artificial_delay,
                                              process_name=pipeline.name)
            process.start()
            parallel_processes.append(process)

        for process in parallel_processes:
            process.wait_for_setup()

        controller_process = Controller(self.keep_running)
        controller_process.start()

        shared_memory_readers = [SharedMemoryReader(name=pipeline.name) for pipeline in pipelines]

        current_pipe_data = PipeData(frame=None, depth_frame=None, unfiltered_frame=None, last_touched_process="None",
                                     creation_time=time.time())
        current_pipe_data.timing_info.start("Process Frame")

        start_video.value = True # All Setup finished allow the video reader to start

        while self.keep_running.value:
            for reader in shared_memory_readers:
                pipe_data_bytes = reader.read()

                if pipe_data_bytes:
                    random_nr = np.random.randint(0, 10000)
                    print(
                        f"New Pipe Data: {random_nr} {(time.perf_counter() - Config.program_start_time):.2f} s")


                    new_pipe_data: PipeData = pickle.loads(pipe_data_bytes)
                    new_pipe_data.timing_info.stop("Transfer Data (SPF -> MM)")
                    print(f"New Pipe Data: {random_nr}:{new_pipe_data.last_touched_process} {(time.perf_counter() - Config.program_start_time):.2f} s")

                    new_pipe_data.timing_info.start("Merge Data",
                                                    parent=f"Data Lifecycle {new_pipe_data.last_touched_process}")

                    current_pipe_data.merge(new_pipe_data)

                    new_pipe_data.timing_info.stop("Merge Data")
                    current_pipe_data.timing_info.start("Transfer Data (MM -> Viz+)",
                                                        parent=f"Data Lifecycle {current_pipe_data.last_touched_process}")
                    # print(f"Timing_Info MM -> Viz: {current_pipe_data.timing_info}")
                    composite_pipe_writer.write(pickle.dumps(current_pipe_data, protocol=pickle.HIGHEST_PROTOCOL))

                    current_pipe_data.timing_info.remove_recursive(
                        f"Data Lifecycle {current_pipe_data.last_touched_process}")

        composite_pipe_writer.close()
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