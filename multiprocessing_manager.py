import multiprocessing as mp
import pickle
import time

from ripc import SharedMemoryWriter, SharedMemoryReader
from config import Config
from controllers.controller import Controller
from helpers.controlled_process import ControlledProcess
from helpers.helpers import initialize_config
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess


class MultiProcessingManager(ControlledProcess):
    __slots__ = ['keep_running']

    def __init__(self, keep_running: mp.Value, shared_list: mp.Array, name=None):
        super().__init__(name=name)
        self.keep_running = keep_running
        self.shared_list = shared_list

    def run(self):
        composite_pipe_writer = SharedMemoryWriter(name=Config.composite_pipe_memory_name, size=Config.pipe_memory_size)
        self.finish_setup()

        parallel_config, video_info, video_rois = initialize_config()

        parallel_processes = []
        process_names = [f"SFP_{index}" for index in range(len(parallel_config))]

        for (index, config) in enumerate(parallel_config):
            process = SequentialFilterProcess(filter_configuration=config,
                                              keep_running=self.keep_running,
                                              processed_frame_count=self.shared_list[index],
                                              process_name=process_names[index])
            process.start()
            parallel_processes.append(process)

        for process in parallel_processes:
            process.wait_for_setup()

        controller_process = Controller(self.keep_running)
        controller_process.start()

        shared_memory_readers = [SharedMemoryReader(name=process_name) for process_name in process_names]

        current_pipe_data = PipeData(frame=None, depth_frame=None, unfiltered_frame=None, last_touched_process="None",
                                     creation_time=time.time())
        current_pipe_data.timing_info.start("Process Frame")
        # current_pipe_data = None

        while self.keep_running.value:
            for reader in shared_memory_readers:
                pipe_data_bytes = reader.blocking_read()
                if pipe_data_bytes is None:
                    self.join_all_processes(controller_process, parallel_processes)
                    return

                new_pipe_data: PipeData = pickle.loads(pipe_data_bytes)
                new_pipe_data.timing_info.stop("Transfer Data (SPF -> MM)")

                new_pipe_data.timing_info.start("Merge Data",
                                                parent=f"Data Lifecycle {new_pipe_data.last_touched_process}")

                # if current_pipe_data:
                #     current_pipe_data.merge(new_pipe_data)
                # else:
                #     current_pipe_data = new_pipe_data
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