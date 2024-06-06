import multiprocessing as mp
import pickle
import time

from ripc import SharedMemoryWriter, SharedMemoryReader
from config import Config
from controllers.controller import Controller
from helpers.ControlledProcess import ControlledProcess
from helpers.helpers import Timer, initialize_config
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

        current_data = PipeData(frame=None, depth_frame=None, unfiltered_frame=None, last_touched_process="None")

        while self.keep_running.value:
            print("MultiProcessingManager: LoOping")
            for reader in shared_memory_readers:
                pipe_data_bytes = reader.read_in_place(ignore_same_version=True)
                while pipe_data_bytes is None:
                    if not self.keep_running.value:
                        self.join_all_processes(controller_process, parallel_processes)
                        return
                    time.sleep(0.01)
                    pipe_data_bytes = reader.read_in_place(ignore_same_version=True)

                pipe_data: PipeData = pickle.loads(pipe_data_bytes)

                with Timer(f"Frame Processing Loop for {current_data.last_touched_process}", min_print_time=0.01):

                    if current_data.last_touched_process != "None":
                        current_data.merge(pipe_data)
                    else:
                        current_data = pipe_data

                    composite_pipe_writer.write(pickle.dumps(current_data, protocol=pickle.HIGHEST_PROTOCOL))

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
