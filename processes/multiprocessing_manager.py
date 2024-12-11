import ctypes
import multiprocessing as mp
import pickle
import time

import cv2
from ripc import SharedMemoryWriter, SharedMemoryReader, SharedMemoryCircularQueue

from configuration.config import Config
from control.controller import Controller
from perception.helpers import initialize_config
from perception.objects.pipe_data import PipeData
from processes.controlled_process import ControlledProcess
from processes.mock_camera_process import MockCameraProcess
from processes.sequential_filter_process import SequentialFilterProcess


class MultiProcessingManager(ControlledProcess):
    __slots__ = ['keep_running']

    def __init__(self, program_start_time: float, keep_running: mp.Value, name=None):
        super().__init__(name=name, program_start_time=program_start_time)
        self.keep_running = keep_running

    def run(self):
        try:
            control_loop_pipe_writer = SharedMemoryWriter(name=Config.control_loop_memory_name, size=Config.pipe_memory_size)
            visualization_queue: SharedMemoryCircularQueue = SharedMemoryCircularQueue.create(Config.visualization_memory_name, Config.pipe_memory_size, Config.visualizer_queue_element_count)

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
                artificial_delay = 0.0

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

            self.finish_setup()
            print("Setup finished")

            shared_memory_readers = [SharedMemoryReader(name=pipeline.name) for pipeline in pipelines]

            current_pipe_data = PipeData(frame=None, frame_version=-1, depth_frame=None, last_filter_process_name="None",
                                         creation_time=time.time(), raw_frame=None)
            current_pipe_data.timing_info.start("Process Frame")

            start_video.value = True # All Setup finished allow the video reader to start

            is_full = False

            while self.keep_running.value:
                for reader in shared_memory_readers:
                    pipe_data_bytes = reader.try_read()

                    if pipe_data_bytes:

                        new_pipe_data: PipeData = pickle.loads(pipe_data_bytes)
                        new_pipe_data.timing_info.stop(f"Transfer Data (SPF -> MM) {new_pipe_data.last_filter_process_name}")
                        new_pipe_data.timing_info.start(f"Merge Data {new_pipe_data.last_filter_process_name}",
                                                        parent=f"Data Lifecycle {new_pipe_data.last_filter_process_name}")
                        current_pipe_data.merge(new_pipe_data)
                        current_pipe_data.timing_info.stop(f"Merge Data {new_pipe_data.last_filter_process_name}")
                        current_pipe_data.timing_info.start(f"Transfer Data (MM -> Viz+) {new_pipe_data.last_filter_process_name}",
                                                            parent=f"Data Lifecycle {current_pipe_data.last_filter_process_name}")

                        pickled_pipe_data = pickle.dumps(current_pipe_data, protocol=pickle.HIGHEST_PROTOCOL)
                        control_loop_pipe_writer.write(pickled_pipe_data)

                        visualization_queue.try_write(pickled_pipe_data)
                        current_pipe_data.timing_info.remove_recursive(
                            f"Data Lifecycle {current_pipe_data.last_filter_process_name}")

                if visualization_queue.is_full() and not is_full:
                    print(f"!!! Visualization queue full with {len(visualization_queue)} elements !!! (your system is too slow lower the fps)")
                    is_full = True

            control_loop_pipe_writer.close()
            self.join_all_processes(controller_process, parallel_processes)
        except Exception as e:
            print(f"Error in {self.name}: {e}")
            self.keep_running.value = False

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