import multiprocessing as mp
import os
import pickle
import time

from perception.objects.video_info import VideoRois, VideoInfo
from perception.visualize_data import visualize_data
from rs_ipc import (
    SharedMessage,
    OperationMode,
    ReaderWaitPolicy,
    read_all_map,
)

from configuration.config import Config, ProcessingStrategy
from perception.helpers import initialize_config, get_roi_bbox_for_video, pack_named_images
from perception.objects.pipe_data import PipeData
from perception.objects.save_info import SaveInfo
from processes.control_process import Control
from processes.mock_camera_process import MockCameraProcess
from processes.sequential_filter_process import SequentialFilterProcess
from processes.video_writer_process import VideoWriterProcess


class MultiProcessingManager(mp.Process):
    __slots__ = ["keep_running", "start_video", "recording_dir_path"]

    def __init__(
        self,
        program_start_time: float,
        keep_running: mp.Value,
        start_video: mp.Value,
        recording_dir_path: str,
        final_frame_version: mp.Value,
        name=None,
    ):
        super().__init__(name=name)
        self.keep_running = keep_running
        self.start_video = start_video
        self.recording_dir_path = recording_dir_path
        self.final_frame_version = final_frame_version
        self.program_start_time = program_start_time

    def run(self):
        video_feed_shm = SharedMessage.create(
            name=Config.video_feed_memory_name,
            size=Config.frame_size,
            mode=OperationMode.CreateOnly,
            reader_wait_policy=ReaderWaitPolicy.All()
            if Config.processing_strategy
               == ProcessingStrategy.ALL_FRAMES_ALL_PROCESSES
            else ReaderWaitPolicy.Count(1)
        )
        control_loop_shm = SharedMessage.create(
            name=Config.control_loop_memory_name,
            size=Config.max_pipe_data_size,
            mode=OperationMode.WriteAsync,
            reader_wait_policy=ReaderWaitPolicy.Count(0)
        )

        save_shm_queue = None
        video_writer_process = None
        if Config.save_processed_video:
            save_shm_queue = SharedMessage.create(
                Config.save_final_memory_name,
                Config.max_pipe_data_size,
                OperationMode.WriteAsync,
                ReaderWaitPolicy.All()
            )  # ReadAsync will make it operate like a queue, as long as the writer side has ReaderWaitPolicy active

            video_name = os.path.splitext(Config.video_name)[0]

            save_info = SaveInfo(
                video_path=os.path.join(
                    self.recording_dir_path, f"Final_{video_name}.mp4"
                ),
                width=Config.width,
                height=Config.height,
                fps=Config.output_fps,
            )

            video_writer_process = VideoWriterProcess(
                save_info=save_info,
                shared_memory_name=Config.save_final_memory_name,
                keep_running=self.keep_running,
                program_start_time=self.program_start_time,
                name="VideoWriterProcess",
            )
            video_writer_process.start()

        pipelines, _, _ = initialize_config(Config.enable_pipeline_visualization)

        pipeline_processes: list[tuple[mp.Process, mp.Pipe]] = []
        pipeline_shm_list = []

        for index, pipeline in enumerate(pipelines):
            pipeline_shm_list.append(
                SharedMessage.create(
                    Config.shm_base_name + pipeline.name,
                    Config.max_pipe_data_size,
                    OperationMode.ReadSync,
                    ReaderWaitPolicy.Count(0)
                )
            )

            debug_pipe, child_debug_pipe = mp.Pipe()
            artificial_delay = 0.0

            process = SequentialFilterProcess(
                filters=pipeline.filters,
                keep_running=self.keep_running,
                debug_pipe=child_debug_pipe,
                artificial_delay=artificial_delay,
                process_name=pipeline.name,
                program_start_time=self.program_start_time,
            )

            process.start()
            pipeline_processes.append((process, debug_pipe))
        print("[MPManager] All parallel processes started")

        camera_process = MockCameraProcess(
            start_video=self.start_video,
            keep_running=self.keep_running,
            program_start_time=self.program_start_time,
            final_frame_version=self.final_frame_version,
            name="CameraProcess",
        )
        camera_process.start()
        print("[MPManager] CameraProcess started")

        control_process = Control(self.keep_running)
        control_process.start()
        print("[MPManager] Controller process started")

        print("[MPManager] Setup finished")
        self.start_video.value = True

        current_pipe_data = PipeData(
            frame=None,
            frame_version=-1,
            depth_frame=None,
            last_pipeline_name="None",
            creation_time=time.time(),
            raw_frame=None,
        )
        current_pipe_data.timing_info.start("Process Video (in Parallel)")

        video_rois: VideoRois = get_roi_bbox_for_video(
            Config.video_name, Config.width, Config.height, Config.roi_config_path
        )
        video_info = VideoInfo(
            video_name=Config.video_name,
            height=Config.height,
            width=Config.width,
            video_rois=video_rois,
        )

        rust_ui = SharedMessage.open(
            "rust_ui", OperationMode.WriteAsync
        )
        iteration_counter = 0
        while self.keep_running.value:
            pipe_data_list: list[PipeData | None] = read_all_map(
                pipeline_shm_list, deserialize_pipe_data
            )
            iteration_counter += 1
            for new_pipe_data in pipe_data_list:
                if new_pipe_data is not None:
                    dl = f"Data Lifecycle {new_pipe_data.last_pipeline_name[0]}"
                    md = f"Merge Data {new_pipe_data.last_pipeline_name[0]}"
                    tf2 = f"Transfer Merged Data {new_pipe_data.last_pipeline_name[0]}"

                    new_pipe_data.timing_info.start(md, parent=dl)
                    current_pipe_data.merge(new_pipe_data)
                    current_pipe_data.timing_info.stop(md)

                    current_pipe_data.timing_info.start(tf2, dl)
                    pickled_pipe_data = pickle.dumps(
                        current_pipe_data, protocol=pickle.HIGHEST_PROTOCOL
                    )
                    if not control_loop_shm.is_stopped():
                        control_loop_shm.write(pickled_pipe_data)

                    if (
                        Config.save_processed_video
                        and save_shm_queue
                        and not save_shm_queue.is_stopped()
                    ):
                        save_shm_queue.write(pickled_pipe_data)

                    current_pipe_data.timing_info.remove_recursive(dl)

                    del new_pipe_data

            if not rust_ui.is_stopped():
                drawn_frame = visualize_data(
                    video_info=video_info, data=current_pipe_data, raw_frame=current_pipe_data.raw_frame
                )
                squashed_frames = [("Main", drawn_frame)]
                if current_pipe_data.processed_frames is not None and len(current_pipe_data.processed_frames) > 0:
                    for name, pipeline_images in current_pipe_data.processed_frames.items():
                        for (index, image) in enumerate(pipeline_images):
                            squashed_frames.append((f"{name} {index}", image))

                images_bytes = bytes(pack_named_images(squashed_frames))
                # print(time.perf_counter() - start_time1)
                if len(images_bytes) >= rust_ui.payload_max_size():
                    print("IMAGES TOOO BIIIIIGGGGGG!!!!!!!!")
                rust_ui.write(images_bytes)

            if current_pipe_data.frame_version == self.final_frame_version.value:
                    print(f"[Main] Received final frame version: {current_pipe_data.frame_version}")
                    self.keep_running.value = False

        if save_shm_queue:
            save_shm_queue.stop()

        print("[MPManager] Exiting main loop")
        control_loop_shm.stop()
        video_feed_shm.stop()

        print("[MPManager] Joining ControllerProcess")
        control_process.join()
        print("[MPManager] ControllerProcess joined")

        print("[MPManager] Joining CameraProcess")
        camera_process.join()
        print("[MPManager] CameraProcess joined")

        print("[MPManager] Joining all parallel processes")
        frames_indexes_dict = {}
        for process, debug_pipe in pipeline_processes:
            try:
                processed_frame_indexes = debug_pipe.recv()
                frames_indexes_dict[process.name] = processed_frame_indexes
                # print(
                #     f"[MPManager] {process.name} processed frame indexes: {processed_frame_indexes}"
                # )
            except EOFError:
                print(f"[MPManager] Error: {process.name} debug pipe is closed")
            finally:
                debug_pipe.close()
            process.join()
        print("[MPManager] All parallel processes joined")

        if video_writer_process:
            print("[MPManager] Joining VideoWriterProcess")
            video_writer_process.join()
            print("[MPManager] VideoWriterProcess joined")

        rust_ui.stop()


def deserialize_pipe_data(pipe_data_bytes: bytes) -> PipeData:
    pipe_data = pickle.loads(pipe_data_bytes)
    pipe_data.timing_info.stop(f"Transfer Data {pipe_data.last_pipeline_name[0]}")
    return pipe_data