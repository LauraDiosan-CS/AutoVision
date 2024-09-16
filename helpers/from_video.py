import cv2
import numpy as np
import time
from tqdm import tqdm

current_datetime = time.strftime("%Y%m%d_%H%M", time.localtime())
log_file_path = f"coordinates_log_{current_datetime}.txt"


def process_frame(frame, start_time):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the image to get yellow colors
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find yellow pixel coordinates
    yellow_pixels = np.column_stack(np.where(yellow_mask > 0))

    if len(yellow_pixels) > 0:
        # Use k-means clustering to group yellow pixels into clusters
        num_clusters = 2  # You can adjust the number of clusters
        _, labels, centers = cv2.kmeans(yellow_pixels.astype(np.float32), num_clusters, None,
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.2),
                                        attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

        # Draw a red circle for each cluster center (mean coordinate for each section)
        for center in centers:
            cv2.circle(frame, (int(center[1]), int(center[0])), 5, (0, 0, 255), -1)

        # Calculate the mean of means (mean of cluster centers)
        overall_mean_coords = np.mean(centers, axis=0)

        timestamp_ms = int((time.time() - start_time) * 1000)

        actual_x = round(285 * overall_mean_coords[1] / frame.shape[0], 3)
        actual_y = round(532.5 * overall_mean_coords[0] / frame.shape[1], 3)

        with open(log_file_path, "a") as log_file:
            log_file.write(f"{timestamp_ms}, {actual_x}, {actual_y} \n")

        # Draw a blue circle for the mean of means
        cv2.circle(frame, (int(overall_mean_coords[1]), int(overall_mean_coords[0])), 20, (250, 20, 250), -1)
        frame = image = cv2.putText(frame, f'{actual_x}, {actual_y}',
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (250, 20, 250), 2, cv2.LINE_AA, False)


    cv2.imshow('Processed Frame', frame)


def main():
    video_path = 'videos/2023-12-15 15-53-35.mkv'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_writer = cv2.VideoWriter('output_video.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    pbar = tqdm(total=total_frames, desc='Processing Video')

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        process_frame(frame, start_time)

        # Write the processed frame to the output video
        output_video_writer.write(frame)

        pbar.update(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pbar.close()

    # Release the video capture object, close the progress bar, and the video writer
    cap.release()
    output_video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()