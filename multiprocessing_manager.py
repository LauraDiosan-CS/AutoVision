import os
import time
from datetime import datetime

import cv2
import torch.multiprocessing as mp

from config import Config
from controllers.controller import Controller
from helpers.helpers import save_frames, Timer, initialize_config
from objects.pipe_data import PipeData
from objects.sequential_filter_process import SequentialFilterProcess
from objects.types.save_info import SaveInfo


def camera_process(in_pipes: list[mp.Pipe]):
    video_path = str(os.path.join(Config.videos_dir, Config.video_name))
    cap = cv2.VideoCapture(video_path)
    Config.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Config.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        with Timer("Camera Process Loop", min_print_time=0.1):
            ret, frame = cap.read()
            if ret:
                data = PipeData(frame=frame,
                                depth_frame=None,
                                unfiltered_frame=frame.copy())
                for pipe in in_pipes:
                    pipe.send(data)
            else:
                break

            time.sleep(1 / Config.fps)

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

        in_pipes = [mp.Pipe(duplex=False) for _ in parallel_config]
        self.output_queue = mp.Queue()

        self.camera_proc = mp.Process(target=camera_process, args=[[pipe[1] for pipe in in_pipes]])
        self.camera_proc.start()

        self.parallel_processes = []

        index = 0
        for config, in_pipe in zip(parallel_config, in_pipes):
            process = SequentialFilterProcess(config, in_pipe[0], self.output_queue, process_name=f"SFP_{index}")
            process.start()
            self.parallel_processes.append(process)
            index += 1

        self.controller_pipe, controller_pipe_child = mp.Pipe()
        self.controller = Controller(controller_pipe_child)
        self.controller.start()

        current_data = PipeData(frame=None, depth_frame=None, unfiltered_frame=None, last_touched_process="None")

        if self.save_input or self.save_output:
            self.initialize_saving()

        while True:
            print(f"Processed Frames Queue size: {self.output_queue.qsize()}")
            data: PipeData = self.output_queue.get()
            with Timer(f"Frame Processing Loop for {current_data.last_touched_process}", min_print_time=0.01):

                if current_data.last_touched_process != "None":
                    current_data.merge(data)
                else:
                    current_data = data

                self.controller_pipe.send(current_data)


                self.viz_queue.put(current_data)

                if self.save_input:
                    if self.save_enabled is not None and self.save_enabled.value:
                        self.save_queue.put(current_data.unfiltered_frame)
                    else:
                        self.save_queue.put("STOP")

                if self.save_output:
                    if self.save_enabled is not None and self.save_enabled.value:
                        self.save_queue.put(current_data.frame)
                    else:
                        self.save_queue.put("STOP")

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
                                             save_info))
        self.save_process.start()

    def finish_saving(self):
        with self.save_enabled.get_lock():
            if not self.save_enabled.value:
                print("Saving was not enabled")
                return

        with self.save_enabled.get_lock():
            self.save_enabled.value = False

        self.controller.join()
        self.save_process.join()  # Wait for the save process to finish current frame
        self.camera_proc.join()

        for process in self.parallel_processes:
            process.join()