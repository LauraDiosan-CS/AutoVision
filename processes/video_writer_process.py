import multiprocessing as mp
import pickle

import cv2
from rs_ipc import SharedMessage, OperationMode

from configuration.config import Config
from perception.helpers import get_roi_bbox_for_video
from perception.objects.save_info import SaveInfo
from perception.objects.video_info import VideoRois, VideoInfo
from perception.visualize_data import visualize_data


class VideoWriterProcess(mp.Process):
    def __init__(
        self,
        save_info: SaveInfo,
        shared_memory_name: str,
        keep_running: mp.Value,
        program_start_time: float,
        name: str = None,
    ):
        super().__init__(name=name)
        self.save_info = save_info
        self.shared_memory_name = shared_memory_name
        self.keep_running = keep_running
        self.program_start_time = program_start_time

    def run(self):
        try:
            video_feed_shm = SharedMessage.open(
                self.shared_memory_name, mode=OperationMode.ReadAsync()
            )  # ReadAsync will make it operate like a queue, as long as the writer side has ReaderWaitPolicy active
            video_writer = cv2.VideoWriter(
                self.save_info.video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.save_info.fps,
                (self.save_info.width, self.save_info.height),
            )

            video_rois: VideoRois = get_roi_bbox_for_video(
                Config.video_name, Config.width, Config.height, Config.roi_config_path
            )
            video_info = VideoInfo(
                video_name=Config.video_name,
                height=Config.height,
                width=Config.width,
                video_rois=video_rois,
            )
            read_count = 0
            while self.keep_running.value:
                pipe_data_as_bytes = video_feed_shm.read(block=True)
                read_count += 1
                # print(f"[VideoWriterProcess] Read {read_count} elems from queue")

                if pipe_data_as_bytes is None:
                    break

                pipe_data = pickle.loads(pipe_data_as_bytes)

                drawn_frame = visualize_data(
                    video_info=video_info, data=pipe_data, raw_frame=pipe_data.raw_frame
                )

                video_writer.write(drawn_frame)

            print("VideoWriterProcess: Video ended")
            video_writer.release()
            video_feed_shm.stop()
        except Exception as e:
            print(f"VideoWriterProcess: Error: {e}")
            self.keep_running.value = False