import cv2

def save_frame(video_capture, frame, save_path):
    cv2.imwrite(save_path, frame)
    print(f"Frame saved at {save_path}")

def main():
    video_dir = 'C:\\Users\\Rotaru Mira\\Desktop\\CarVision\\laneDetect\\videos\\'
    video_name = 'steeringUpdated.MP4'
    video_path = video_dir + video_name
    cap = cv2.VideoCapture(video_path)
    save_dir = 'C:\\Users\\Rotaru Mira\\Desktop\\CarVision\\trafficSigns_extension\\'

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        cv2.imshow('Video Frame', frame)

        key = cv2.waitKey(30)

        if key == ord('s'):
            save_frame(cap, frame, f"{save_dir}{video_name}_{frame_number}.png")
            frame_number += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()