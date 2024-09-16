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

    def __init__(self, keep_running: mp.Value):
        super().__init__()
        self.keep_running = keep_running

    def run(self):
        composite_pipe_writer = SharedMemoryWriter(name=Config.composite_pipe_memory_name, size=Config.pipe_memory_size)
        self.finish_setup()

        parallel_config, video_info, video_rois = initialize_config()

        parallel_processes = []
        process_names = [f"SFP_{index}" for index in range(len(parallel_config))]
        for (index, config) in enumerate(parallel_config):
            process = SequentialFilterProcess(filter_configuration=config,
                                              process_name=process_names[index],
                                              keep_running=self.keep_running)
            process.start()
            parallel_processes.append(process)

        controller_process = Controller(self.keep_running)
        controller_process.start()

        shared_memory_readers = [SharedMemoryReader(name=process_name) for process_name in process_names]

        # current_data = PipeData(frame=None, depth_frame=None, unfiltered_frame=None, last_touched_process="None",
        #                         creation_time=time.time())
        current_data = None

        while self.keep_running.value:
            for reader in shared_memory_readers:
                pipe_data_bytes = reader.blocking_read()
                if pipe_data_bytes is None:
                    self.join_all_processes(controller_process, parallel_processes)
                    return
                pipe_data: PipeData = pickle.loads(pipe_data_bytes)
                pipe_data.arrive_time = time.time()
                pipe_data.timings.stop("Transfer Data (SPF -> MM)")


                if current_data and current_data.last_touched_process != "None":
                    current_data.merge(pipe_data)
                    # print("merging")
                else:
                    current_data = pipe_data

                pipe_data.timings.start("Transfer Data (MM -> Viz+)", parent="Data Lifecycle")
                composite_pipe_writer.write(pickle.dumps(current_data, protocol=pickle.HIGHEST_PROTOCOL))

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