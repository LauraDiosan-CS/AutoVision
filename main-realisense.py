import argparse
import os
from datetime import datetime

import cv2
import torch.multiprocessing as mp
from config import Config
from filters.draw_filter import Visualizer
from helpers.helpers import stack_images_v2, draw_rois_and_wait, Timer, get_roi_bbox_for_video, save_frames
from multiprocessing_manager import MultiProcessingManager
from objects.types.save_info import SaveInfo
from objects.types.video_info import VideoRois, VideoInfo


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




def main():
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')

    viz_pipe, viz_pipe_mp_manager = mp.Pipe()
    mp_manager_terminate_flag = mp.Value('b', False)

    mp_manager = MultiProcessingManager(viz_pipe_mp_manager, mp_manager_terminate_flag)

    mp_manager.start()

    save_queue = None
    save_process = None
    if Config.save_processed_video:
        save_queue = mp.Queue()

        # Extract the name without the extension
        video_name = os.path.splitext(Config.video_name)[0]

        save_info = SaveInfo(
            video_path=os.path.join(Config.recordings_dir,
                                    f"Processed_{video_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"),
            width=Config.width,
            height=Config.height,
            fps=Config.fps
        )
        save_process = mp.Process(target=save_frames,
                                  args=(save_queue,
                                        save_info))
        save_process.start()

    video_rois: VideoRois = get_roi_bbox_for_video(Config.video_name, Config.roi_config_path)
    video_info = VideoInfo(video_name=Config.video_name, height=Config.height,
                           width=Config.width, video_rois=video_rois)

    data_visualizer = Visualizer(video_info=video_info)

    viz_pipe.recv() # Wait for the first frame to be processed
    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    while True:
        # print("Visualize Queue size : ", viz_pipe.qsize())
        data = viz_pipe.recv()

        with Timer("Main Process Loop"):
            if Config.apply_visualizer:
                visualized_frame = data_visualizer.draw_frame_based_on_data(data)

            if data.processed_frames is not None and len(data.processed_frames) > 0:
                squashed_frames = sum(data.processed_frames.values(), [])
                squashed_frames.append(visualized_frame)
                final_img = stack_images_v2(1, squashed_frames)
            else:
                final_img = visualized_frame

            cv2.imshow('CarVision', final_img)

            if Config.save_processed_video and save_queue is not None:
                print("Processed Save Queue size : ", save_queue.qsize())
                save_queue.put(visualized_frame)

            key = cv2.waitKey(5)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('x'):
                draw_rois_and_wait(visualized_frame, video_rois)
                cv2.waitKey(0)


    cv2.destroyAllWindows()

    mp_manager_terminate_flag.value = True
    print("Initiating Termination of MultiProcessingManager")

    if save_queue is not None and save_process is not None:
        print("Joining processed frame saving process")
        save_queue.put("STOP")
        save_process.join()
    print("Processed frame saving process joined")

    print("Joining MultiProcessingManager")
    mp_manager.terminate()
    print("MultiProcessingManager joined")


def update_config(config, args):
    for arg in vars(args):
        if hasattr(config, arg):
            setattr(config, arg, getattr(args, arg))
    Config.print_config()


if __name__ == '__main__':
    args = argparse.ArgumentParser().parse_args()
    update_config(Config, args)
    main()