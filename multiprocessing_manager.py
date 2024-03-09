import os
import time
from datetime import datetime
from queue import Empty

import cv2
import numpy as np
import torch.multiprocessing as mp

from config import Config
from controllers.controller import Controller
from helpers.helpers import save_frames, Timer, initialize_config
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from objects.types.save_info import SaveInfo

import pyrealsense2 as rs


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
    def __init__(self, viz_pipe: mp.Pipe, terminate_flag: mp.Value, live_record: bool=False):
        super().__init__()
        self.live_record = live_record
        self.save_enabled = mp.Value('b', Config.save_video)
        self.terminate_flag = terminate_flag
        self.camera_term_flag = mp.Value('b', False)

        self.viz_pipe = viz_pipe
        self.http_connection_failed_count = 0

        self.parallel_processes = []
        self.controller_process = None
        self.controller_pipe = None
        self.camera_process = None
        self.output_queue = None

    def run(self):
        parallel_config, video_info, video_rois = initialize_config()

        self.in_pipes = [mp.Pipe(duplex=False) for _ in parallel_config]

        if self.live_record:
            self.camera_process = mp.Process(target=live_camera_process, args=[self.camera_term_flag, self.save_enabled,
                                                                               [pipe[1] for pipe in self.in_pipes]])
        else:
            self.camera_process = mp.Process(target=camera_process, args=[self.camera_term_flag, self.save_enabled,
                                                                          [pipe[1] for pipe in self.in_pipes]])
        self.camera_process.start()

        self.output_queue = mp.Queue()

        index = 0
        for config, in_pipe in zip(parallel_config, self.in_pipes):
            process = SequentialFilterProcess(config, in_pipe[0], self.output_queue, process_name=f"SFP_{index}")
            process.start()
            self.parallel_processes.append(process)
            index += 1

        self.controller_pipe, controller_pipe_child = mp.Pipe()
        self.controller_process = Controller(controller_pipe_child)
        self.controller_process.start()

        current_data = PipeData(frame=None, depth_frame=None, unfiltered_frame=None, last_touched_process="None")

        while True:
            print(f"Processed Frames Queue size: {self.output_queue.qsize()}")

            try:
                data: PipeData = self.output_queue.get(timeout=1)
            except Empty:  # If the queue is empty, continue to make sure the terminate flag is checked
                pass

            if self.terminate_flag.value:
                self.join_all_processes()
                break

            with Timer(f"Frame Processing Loop for {current_data.last_touched_process}", min_print_time=0.01):

                if current_data.last_touched_process != "None":
                    current_data.merge(data)
                else:
                    current_data = data

                self.controller_pipe.send(current_data)

                self.viz_pipe.send(current_data)
        print("Exiting MultiProcessingManager")

    def join_all_processes(self):
        self.save_enabled.value = False  # Stop saving
        self.controller_pipe.send("STOP")  # Stop the controller process
        for pipe in self.in_pipes:
            pipe[1].send("STOP")

        print("Joining Controller process")
        self.controller_process.join()
        print("Controller process joined")

        print("Joining parallel processes")
        for process in self.parallel_processes:
            print(f"Joining {process.name}")
            # process.join()
            process.terminate()
        print("All parallel processes joined")

        print("Setting camera term flag")
        self.camera_term_flag.value = True  # Terminate the camera process
        print("Joining Camera process")
        self.camera_process.terminate()
        # self.camera_process.join()
        print("Camera process joined")