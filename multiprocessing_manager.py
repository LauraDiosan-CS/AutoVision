import os
import time
from datetime import datetime

import cv2
import torch.multiprocessing as mp

from config import Config
from controller import Controller
from filters.base_filter import BaseFilter
from filters.draw_filter import DrawFilter
from helpers.helpers import save_frames, Timer, initialize_config
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from objects.types.save_info import SaveInfo
from objects.types.video_info import VideoInfo


def camera_process(in_queues: list[mp.Queue]):
    video_path = str(os.path.join(Config.videos_dir, Config.video_name))
    cap = cv2.VideoCapture(video_path)
    Config.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Config.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        with Timer("Camera Process Loop", min_print_time=0.01):
            ret, frame = cap.read()
            if ret:
                data = PipeData(frame=frame,
                                depth_frame=None,
                                unfiltered_frame=frame.copy())
                for queue in in_queues:
                    queue.put(data)
            else:
                break

            time.sleep(1/Config.fps)

    cap.release()


class MultiProcessingManager(mp.Process):
    def __init__(self, queue: mp.Queue, save_input=False, save_output=False):
        super().__init__()
        self.save_input = save_input
        self.save_output = save_output
        self.viz_queue = queue
        self.http_connection_failed_count = 0
        self.save_queue = None
        self.save_process = None
        self.save_enabled = None




    def run(self):
        parallel_config, video_info, video_rois = initialize_config()

        in_queues = [mp.Queue() for _ in range(len(parallel_config))]
        self.output_queue = mp.Queue()

        self.camera_proc = mp.Process(target=camera_process, args=[in_queues])
        self.camera_proc.start()

        self.parallel_processes = []
        for config, in_queue in zip(parallel_config, in_queues):
            process = SequentialFilterProcess(config, in_queue, self.output_queue)
            process.start()
            self.parallel_processes.append(process)

        # self.controller_pipe, controller_pipe_child = mp.Pipe()
        # self.controller = Controller(controller_pipe_child)
        # self.controller.start()

        self.draw_filter = DrawFilter(video_info=video_info)

        current_data = None

        # if self.save_input or self.save_output:
        #     self.initialize_saving()

        while True:
            data: PipeData = self.output_queue.get()
            if current_data is not None:
                current_data.merge(data)
            else:
                current_data = data

            # self.controller_pipe.send(data)

            if True:
                current_data = self.draw_filter.process(current_data)

            self.viz_queue.put(current_data)



    def initialize_saving(self):
        # Start a separate process for saving frames
        self.save_queue = mp.Queue()
        self.save_enabled = mp.Value('b', True)  # Shared boolean value for enabling saving

        # Extract the name without the extension
        video_name = os.path.splitext(Config.video_name)[0]

        save_info = SaveInfo(
            video_path=os.path.join(Config.recordings_dir,
                                    f"{video_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"),
            width=Config.width,
            height=Config.height,
            fps=Config.fps
        )
        self.save_process = mp.Process(target=save_frames,
                                       args=(self.save_queue,
                                             self.save_enabled,
                                             save_info))
        self.save_process.start()

    def process_frame(self, frame, depth_frame=None, apply_draw_filter=False):
        data: PipeData = PipeData(frame=frame,
                                  depth_frame=depth_frame,
                                  unfiltered_frame=frame.copy())

        if self.save_input and self.save_enabled is not None and self.save_enabled.value:
            self.save_queue.put(data.unfiltered_frame)
        elif self.save_output and self.save_enabled is not None and self.save_enabled.value:
            self.save_queue.put(data.frame)

        return data

    def finish_saving(self):
        with self.save_enabled.get_lock():
            if not self.save_enabled.value:
                print("Saving was not enabled")
                return

        with self.save_enabled.get_lock():
            self.save_enabled.value = False

        self.save_process.join()  # Wait for the save process to finish current frame
        self.camera_proc.join()