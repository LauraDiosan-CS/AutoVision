import ctypes
import multiprocessing as mp
import os
import pickle
import time

import ripc
from ripc import SharedMessage, SharedQueue, OpenMode

from configuration.config import Config
from control.controller import Controller
from perception.helpers import initialize_config
from perception.objects.pipe_data import PipeData
from perception.objects.save_info import SaveInfo
from processes.controlled_process import ControlledProcess
from processes.mock_camera_process import MockCameraProcess
from processes.sequential_filter_process import SequentialFilterProcess
from processes.video_writer_process import VideoWriterProcess


class MultiProcessingManager(ControlledProcess):
    __slots__ = ['keep_running', 'start_video', 'recording_dir_path']

    def __init__(self, program_start_time: float, keep_running: mp.Value, start_video: mp.Value, recording_dir_path: str, name=None):
        super().__init__(name=name, program_start_time=program_start_time)
        self.keep_running = keep_running
        self.start_video = start_video
        self.recording_dir_path = recording_dir_path

    def run(self):
        try:
            video_feed_shm = SharedMessage.create(name=Config.video_feed_memory_name, size=Config.frame_size)
            control_loop_shm = SharedMessage.create(name=Config.control_loop_memory_name,
                                                   size=Config.max_pipe_data_size)
            visualization_shm = SharedMessage.open(Config.visualization_memory_name, mode=OpenMode.WriteOnly)

            save_shm_queue = None
            video_writer_process = None
            if Config.save_processed_video:
                save_shm_queue = SharedQueue.create(Config.save_final_memory_name, Config.max_pipe_data_size, OpenMode.WriteOnly)

                video_name = os.path.splitext(Config.video_name)[0]

                save_info = SaveInfo(
                    video_path=os.path.join(self.recording_dir_path,
                                            f"Final_{video_name}.mp4"),
                    width=Config.width,
                    height=Config.height,
                    fps=Config.output_fps
                )

                video_writer_process = VideoWriterProcess(save_info=save_info,
                                                          shared_memory_name=Config.save_final_memory_name,
                                                          keep_running=self.keep_running,
                                                          program_start_time=self.program_start_time,
                                                            name="VideoWriterProcess")
                video_writer_process.start()

            pipelines, _, _ = initialize_config(Config.enable_pipeline_visualization)

            pipeline_processes: list[tuple[mp.Process, mp.Pipe]] = []
            last_read_frame_versions = []
            pipeline_shm_list = []

            for (index, pipeline) in enumerate(pipelines):
                last_read_frame_versions.append(mp.Value(ctypes.c_int, 0))
                pipeline_shm_list.append(SharedMessage.create(name=Config.shm_base_name + pipeline.name, size=Config.max_pipe_data_size))

                pipe, child_pipe = mp.Pipe()

                match pipeline.name:
                    case "LaneDetection":
                        artificial_delay = 0.0
                    case "SignDetection":
                        artificial_delay = 0.1
                    case "TrafficLightDetection":
                        artificial_delay = 0.15
                    case "PedestrianDetection":
                        artificial_delay = 0.2
                artificial_delay = 0.0

                process = SequentialFilterProcess(filters=pipeline.filters,
                                                  keep_running=self.keep_running,
                                                  last_processed_frame_version=last_read_frame_versions[index],
                                                  debug_pipe=child_pipe,
                                                  artificial_delay=artificial_delay,
                                                  process_name=pipeline.name,
                                                  program_start_time=self.program_start_time)

                process.start()
                pipeline_processes.append((process, pipe))
            print("[MPManager] All parallel processes started")

            camera_process = MockCameraProcess(start_video=self.start_video, keep_running=self.keep_running,
                                               last_read_frame_versions=last_read_frame_versions,
                                               program_start_time=self.program_start_time, name="CameraProcess")
            camera_process.start()
            print("[MPManager] CameraProcess started")

            controller_process = Controller(self.keep_running)
            controller_process.start()
            print("[MPManager] Controller process started")

            print("[MPManager] Setup finished")

            current_pipe_data = PipeData(frame=None, frame_version=-1, depth_frame=None,
                                         last_pipeline_name="None", creation_time=time.time(), raw_frame=None)

            current_pipe_data.timing_info.start("Process Video")
            self.start_video.value = True # All Setup finished allow the video reader to start
            write_count = 0
            while self.keep_running.value:
                start_time = time.perf_counter()
                pipe_data_list: list[PipeData | None] = ripc.read_all_map(pipeline_shm_list, deserialize_pipe_data)
                time_to_load = (time.perf_counter() - start_time) * 1000
                # pipe_data_bytes_list = ripc.read_all(pipeline_shm_list)
                # for pipe_data_bytes in pipe_data_bytes_list:
                for new_pipe_data in pipe_data_list:
                    if new_pipe_data:
                        # start_time = time.perf_counter()
                        # new_pipe_data: PipeData = pickle.loads(pipe_data_bytes)
                        # time_to_load = (time.perf_counter() - start_time) * 1000

                        dl = f"Data Lifecycle {new_pipe_data.last_pipeline_name[0]}"
                        tf1 = f"Transfer Data {new_pipe_data.last_pipeline_name[0]}"
                        md = f"Merge Data {new_pipe_data.last_pipeline_name[0]}"
                        tf2 = f"Transfer Merged Data {new_pipe_data.last_pipeline_name[0]}"

                        # new_pipe_data.timing_info.stop(tf1)

                        new_pipe_data.timing_info.start(md, parent=dl)
                        current_pipe_data.merge(new_pipe_data)
                        current_pipe_data.timing_info.stop(md)

                        current_pipe_data.timing_info.start(tf2, dl)

                        start_time = time.perf_counter()
                        pickled_pipe_data = pickle.dumps(current_pipe_data, protocol=pickle.HIGHEST_PROTOCOL)
                        time_to_pickle = (time.perf_counter() - start_time) * 1000

                        start_time = time.perf_counter()
                        visualization_shm.write_async(pickled_pipe_data)
                        control_loop_shm.write_async(pickled_pipe_data)
                        time_to_write = (time.perf_counter() - start_time) * 1000

                        if Config.save_processed_video and save_shm_queue is not None:
                            start_time = time.perf_counter()
                            save_shm_queue.write(pickled_pipe_data)
                            write_count += 1
                            time_to_write_save_queue = (time.perf_counter() - start_time) * 1000
                            print(f"[MP Manager] Wrote {write_count} elements to save queue")

                        print(f"[MP Manager] Time to load: {time_to_load:.2f} ms, Time to pickle: {time_to_pickle:.2f} ms, "
                              f"Time to write: {time_to_write:.2f} ms, Time to write save queue: {time_to_write_save_queue:.2f} ms")

                        current_pipe_data.timing_info.remove_recursive(dl)

                        del new_pipe_data

            control_loop_shm.close()
            video_feed_shm.close()
            visualization_shm.close()
            if save_shm_queue:
                save_shm_queue.close()

            for pipeline_shm in pipeline_shm_list:
                pipeline_shm.close()

            print("[MPManager] Joining ControllerProcess")
            controller_process.join()
            print("[MPManager] ControllerProcess joined")

            print("[MPManager] Joining CameraProcess")
            camera_process.join()
            print("[MPManager] CameraProcess joined")

            if video_writer_process:
                print("[MPManager] Joining VideoWriterProcess")
                video_writer_process.join()
                print("[MPManager] VideoWriterProcess joined")

            print("[MPManager] Joining all parallel processes")
            frames_indexes_dict = {}
            for process, debug_pipe in pipeline_processes:
                try:
                    processed_frame_indexes = debug_pipe.recv()
                    frames_indexes_dict[process.name] = processed_frame_indexes
                    print(f"[MPManager] {process.name} processed frame indexes: {processed_frame_indexes}")
                except EOFError:
                    print(f"[MPManager] Error: {process.name} debug pipe is closed")
                finally:
                    debug_pipe.close()
                process.join()
            print("[MPManager] All parallel processes joined")

            # visualize_frame_indexes(frames_indexes_dict)
            # visualize_cumulative_detections(frames_indexes_dict)
            # visualize_heatmap(frames_indexes_dict)

        except Exception as e:
            print(f"Error in {self.name}: {e}")
            self.keep_running.value = False


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def visualize_frame_indexes(frame_indexes_dict):
    plt.figure(figsize=(15, 8))

    # Assign unique y-values for each detection type
    y_values = {detection: idx for idx, detection in enumerate(frame_indexes_dict.keys(), start=1)}

    for detection, indexes in frame_indexes_dict.items():
        plt.scatter(indexes, [y_values[detection]] * len(indexes),
                    label=detection, alpha=0.6, marker='o')

    # Setting y-axis labels
    plt.yticks(list(y_values.values()), list(y_values.keys()))

    # Adding labels and title
    plt.xlabel('Frame Index')
    plt.title('Processed Frame Indexes per Detection Type')

    # Adding a legend
    plt.legend(loc='upper right')

    # Optional: Improve layout
    plt.tight_layout()

    # Display the plot
    plt.show()


def visualize_cumulative_detections(frame_indexes_dict):
    plt.figure(figsize=(15, 8))

    for detection, indexes in frame_indexes_dict.items():
        sorted_indexes = sorted(indexes)
        cumulative_counts = np.arange(1, len(sorted_indexes) + 1)
        plt.plot(sorted_indexes, cumulative_counts, label=detection)

    plt.xlabel('Frame Index')
    plt.ylabel('Cumulative Count')
    plt.title('Cumulative Processed Frame Indexes per Detection Type')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def deserialize_pipe_data(pipe_data_bytes: bytes) -> PipeData | None:
    pipe_data = pickle.loads(pipe_data_bytes)
    pipe_data.timing_info.stop(f"Transfer Data {pipe_data.last_pipeline_name[0]}") if pipe_data else None
    return pipe_data

def visualize_heatmap(frame_indexes_dict):
    # Determine the maximum frame index to define the range
    max_frame = max(max(indexes) for indexes in frame_indexes_dict.values())

    # Initialize a DataFrame with zeros
    df = pd.DataFrame(0, index=range(1, max_frame + 1), columns=frame_indexes_dict.keys())

    # Populate the DataFrame
    for detection, indexes in frame_indexes_dict.items():
        df.loc[indexes, detection] = 1  # Mark detections

    # Plot heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(df, cmap='YlGnBu', cbar=False)
    plt.xlabel('Detection Type')
    plt.ylabel('Frame Index')
    plt.title('Heatmap of Processed Frame Indexes per Detection Type')
    plt.tight_layout()
    plt.show()