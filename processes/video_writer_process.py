import multiprocessing as mp
import pickle

import cv2
import ripc
from ripc import SharedQueue

from configuration.config import Config
from perception.helpers import get_roi_bbox_for_video
from perception.objects.save_info import SaveInfo
from perception.objects.video_info import VideoRois, VideoInfo
from perception.visualize_data import visualize_data
from processes.controlled_process import ControlledProcess


class VideoWriterProcess(ControlledProcess):
    def __init__(self, save_info: SaveInfo, shared_memory_name: str, keep_running: mp.Value, program_start_time: float, name: str = None):
        super().__init__(name=name, program_start_time=program_start_time)
        self.save_info = save_info
        self.shared_memory_name = shared_memory_name
        self.keep_running = keep_running
        self.program_start_time = program_start_time

    def run(self):
        try:
            save_queue = SharedQueue.open(self.shared_memory_name, mode=ripc.OpenMode.ReadOnly)
            video_writer = cv2.VideoWriter(self.save_info.video_path,
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           self.save_info.fps,
                                           (self.save_info.width, self.save_info.height))

            video_rois: VideoRois = get_roi_bbox_for_video(Config.video_name, Config.width, Config.height,
                                                           Config.roi_config_path)
            video_info = VideoInfo(video_name=Config.video_name, height=Config.height,
                                   width=Config.width, video_rois=video_rois)
            read_count = 0
            while self.keep_running.value:
                pipe_data_as_bytes = save_queue.blocking_read()
                read_count += 1
                print(f"[VideoWriterProcess] Read {read_count} elems from queue")

                if pipe_data_as_bytes is None:
                    break

                pipe_data = pickle.loads(pipe_data_as_bytes)

                drawn_frame = visualize_data(video_info=video_info, data=pipe_data, raw_frame=pipe_data.raw_frame)

                video_writer.write(drawn_frame)

            print("VideoWriterProcess: Video ended")
            video_writer.release()
        except Exception as e:
            print(f"VideoWriterProcess: Error: {e}")
            self.keep_running.value = False