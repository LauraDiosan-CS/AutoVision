import cv2
import numpy as np

def process_frame(frame):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the image to get yellow colors
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find the coordinates of yellow pixels
    yellow_pixels = np.column_stack(np.where(yellow_mask > 0))

    if len(yellow_pixels) > 0:
        # Calculate the mean coordinates of yellow pixels
        mean_coords = np.mean(yellow_pixels, axis=0)

        # Print the coordinates in the command line
        print("Mean Coordinates of Yellow Pixels: x = {}, y = {}".format(mean_coords[1], mean_coords[0]))

        # Set all yellow pixels to red
        frame[yellow_mask > 0] = [0, 0, 255]  # BGR values for red

        # Draw a red dot at the calculated coordinates
        cv2.circle(frame, (int(mean_coords[1]), int(mean_coords[0])), 5, (0, 0, 255), -1)  # Swap x and y for OpenCV coordinates

    # Display the frame with the red dot
    cv2.imshow('Processed Frame', frame)

def main():
    # Open a video capture object with the path to your video file
    video_path = 'GX011680.MP4'  # Replace with the path to your video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # Process the current frame
        process_frame(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
