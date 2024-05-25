import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np
import torch.multiprocessing as mp

from config import Config
from controllers.controller import Controller
from helpers.helpers import save_frames, Timer, initialize_config
from helpers.shared_memory import SharedMemoryReader, SharedMemoryWriter
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from objects.types.save_info import SaveInfo

# import pyrealsense2 as rs


def live_camera_process(terminate_flag: mp.Value, save_enabled_flag: mp.Value, in_pipes: list[mp.Pipe]):
    pipelineCamera = rs.pipeline()
    realsense_config = rs.config()
    realsense_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, Config.fps)
    realsense_config.enable_stream(rs.stream.color, Config.width, Config.height, rs.format.bgr8, Config.fps)
    pipelineCamera.start(realsense_config)

    save_queue = None
    save_process = None
    if save_enabled_flag.value:
        save_queue = mp.Queue()

        # Extract the name without the extension
        video_name = os.path.splitext(Config.video_name)[0]

        save_info = SaveInfo(
            video_path=os.path.join(Config.recordings_dir,
                                    f"{video_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"),
            width=Config.width,
            height=Config.height,
            fps=Config.fps
        )
        save_process = mp.Process(target=save_frames,
                                  args=(save_queue,
                                        save_info))
        save_process.start()

    while not terminate_flag.value:
        with Timer("Camera Process Loop", min_print_time=0.1):
            frames = pipelineCamera.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame and not depth_frame:
                print("!!! No frames received from camera !!!")
                continue

            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = np.asanyarray(depth_frame.get_data())

            data = PipeData(frame=color_frame,
                            depth_frame=depth_frame,
                            unfiltered_frame=color_frame.copy())
            for pipe in in_pipes:
                pipe.send(data)

            if save_enabled_flag.value and save_queue is not None:
                print(f"Save Queue size: {save_queue.qsize()}")
                save_queue.put(color_frame)
        time.sleep(1 / Config.fps)

    if save_queue is not None and save_process is not None:
        print("Joining save process")
        save_queue.put("STOP")
        save_process.join()  # Wait for the save process to finish current frame
        print("Save process joined")


def camera_process(terminate_flag: mp.Value, save_enabled_flag: mp.Value, in_pipes: list[mp.Pipe]):
    video_path = str(os.path.join(Config.videos_dir, Config.video_name))
    cap = cv2.VideoCapture(video_path)
    Config.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # This probably doesn't have any effect for other processes
    Config.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save_queue = None
    save_process = None
    if save_enabled_flag.value:
        save_queue = mp.Queue()

        # Extract the name without the extension
        video_name = os.path.splitext(Config.video_name)[0]

        save_info = SaveInfo(
            video_path=os.path.join(Config.recordings_dir,
                                    f"{video_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"),
            width=Config.width,
            height=Config.height,
            fps=Config.fps
        )
        save_process = mp.Process(target=save_frames,
                                  args=(save_queue,
                                        save_info))
        save_process.start()

    while not terminate_flag.value and cap.isOpened():
        with Timer("Camera Process Loop", min_print_time=0.1):
            ret, frame = cap.read()
            if ret:
                data = PipeData(frame=frame,
                                depth_frame=None,
                                unfiltered_frame=frame.copy())
                for pipe in in_pipes:
                    pipe.send(data)

                if save_enabled_flag.value and save_queue is not None:
                    print(f"Save Queue size: {save_queue.qsize()}")
                    save_queue.put(frame)
            else:
                print("!!! No frames received from camera !!!")
                break
        time.sleep(1 / Config.fps)

    if save_queue is not None and save_process is not None:
        print("Joining save process")
        save_queue.put("STOP")
        save_process.join()  # Wait for the save process to finish current frame
        print("Save process joined")

    cap.release()
    print("Exiting camera process")


class MultiProcessingManager(mp.Process):
    def __init__(self, keep_running: mp.Value):
        super().__init__()
        self.keep_running = keep_running
        self.camera_term_flag = mp.Value('b', False)

        self.http_connection_failed_count = 0

        self.parallel_processes = []
        self.controller_process = None
        self.controller_pipe = None

    def run(self):
        print("Starting MultiProcessingManager")
        parallel_config, video_info, video_rois = initialize_config()

        # if self.live_record:
        #     self.camera_process = mp.Process(target=live_camera_process, args=[self.camera_term_flag, self.save_enabled,
        #                                                                        [pipe[1] for pipe in self.in_pipes]])
        # else:
        #     self.camera_process = mp.Process(target=camera_process, args=[self.camera_term_flag, self.save_enabled,
        #                                                                   [pipe[1] for pipe in self.in_pipes]])
        # self.camera_process.start()

        self.controller_pipe, controller_pipe_child = mp.Pipe()
        self.controller_process = Controller(controller_pipe_child)
        self.controller_process.start()


        process_names = [f"SFP_{index}" for index in range(len(parallel_config))]
        shared_memory_readers = [SharedMemoryReader(topic=process_name, size=100 * 1024 * 1024, create=True) for process_name in process_names]

        for (index, config) in enumerate(parallel_config):
            process = SequentialFilterProcess(filter_configuration=config,
                                              process_name=process_names[index],
                                              keep_running=self.keep_running)
            process.start()
            self.parallel_processes.append(process)


        print("Creating shared memory writer")
        viz_memory_writer = SharedMemoryWriter(topic=Config.visualizer_shared_memory_name, create=False, size=100 * 1024 * 1024)
        print("Shared memory writer created")
        current_data = PipeData(frame=None, depth_frame=None, unfiltered_frame=None, last_touched_process="None")

        while not self.keep_running.value:

            for reader in shared_memory_readers:
                pipe_data_bytes = reader.read()
                while pipe_data_bytes is None:
                    if not self.keep_running.value:
                        self.join_all_processes()
                        return
                    time.sleep(0.01)
                    pipe_data_bytes = reader.read()

                pipe_data: PipeData = pickle.loads(pipe_data_bytes)

                with Timer(f"Frame Processing Loop for {current_data.last_touched_process}", min_print_time=0.01):

                    if current_data.last_touched_process != "None":
                        current_data.merge(pipe_data)
                    else:
                        current_data = pipe_data

                    self.controller_pipe.send(current_data)

                    viz_memory_writer.write(pickle.dumps(current_data, protocol=pickle.HIGHEST_PROTOCOL))

        self.join_all_processes()

    def join_all_processes(self):
        self.controller_pipe.send("STOP")  # Stop the controller process
        print("Joining Controller process")
        self.controller_process.join()
        print("Controller process joined")

        # for keep_running in self.keep_running:
        #     keep_running.value = False
        print("Joining parallel processes")
        for process in self.parallel_processes:
            print(f"Joining {process.name}")
            # process.join()
            process.terminate()
        print("All parallel processes joined")

        # print("Setting camera term flag")
        # self.camera_term_flag.value = True  # Terminate the camera process
        # print("Joining Camera process")
        # self.camera_process.terminate()
        # self.camera_process.join()
        # print("Camera process joined")
        print("Exiting MultiProcessingManager")