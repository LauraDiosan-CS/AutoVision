import argparse
import yaml
import os
import cv2
from pipeline import Pipeline

def main(args):
    pipeline = Pipeline(args)

    video = os.path.join(args.video_dir, args.video)
    cap = cv2.VideoCapture(video)

    print(video)
    cv2.namedWindow('CarVision', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
        
            processed_frame = pipeline.run_seq(frame)

            if processed_frame is not None:
                cv2.imshow('CarVision', processed_frame)
            else:
                cv2.imshow('CarVision', frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()


def parse_config(parser):
    config_file = 'C:\\Users\\Rotaru Mira\\Desktop\\CarVision\\vision-pipeline\\config.yaml'

    with open(config_file, 'r') as config:
        data = yaml.safe_load(config)
        
        for arg in data.keys():
            parser.add_argument(f'--{arg}', default = data[arg])  
        args = parser.parse_args() 

    config.close()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_config(parser)
    main(args)
