import argparse
import os
import cv2
import torch.multiprocessing as mp

from helpers.helpers import parse_config, stack_images_v2, initialize_config, draw_rois_and_wait
from process_pipeline_manager import ProcessPipelineManager


def main(args):
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')

    video_path = str(os.path.join(args.videos_dir, args.video_name))

    cap = cv2.VideoCapture(video_path)
    args.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    args.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    parallel_config, video_info, video_rois = initialize_config(args)

    pool_manager = ProcessPipelineManager(parallel_config, video_info, save=True, args=args)

    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    replay_speed = 1
    ret, frame = None, None

    while cap.isOpened():
        if replay_speed < 1:
            replay_speed = 1

        for i in range(replay_speed):
            ret, frame = cap.read()

        if not ret:
            continue

        data = pool_manager.run(frame, visualize=True)
        data.processed_frames = None
        if data.processed_frames is not None and len(data.processed_frames) > 0:
            imgStack = stack_images_v2(1, data.processed_frames)
            cv2.imshow('CarVision', imgStack)
        else:
            cv2.imshow('CarVision', data.frame)

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('x'):
            draw_rois_and_wait(frame, video_rois)
            cv2.waitKey(0)
        elif key & 0xFF == ord('+'):
            replay_speed += 1
        elif key & 0xFF == ord('-'):
            replay_speed -= 1

    pool_manager.finish_saving()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    args = parse_config(parser, config_file_path)
    main(args)