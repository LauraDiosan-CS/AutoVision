from enum import Enum

class VisualizationStrategy(Enum):
    ALL_FRAMES = 1
    NEWEST_FRAME = 2

class Config:
    models_dir_path = 'configuration/models/'
    videos_dir = 'files/videos'
    recordings_dir = 'files/videos/recordings'
    screenshot_dir = 'screenshots'
    pipeline_config_path = 'configuration/perception_config/parallel_pipeline_config.json'
    roi_config_path = 'configuration/perception_config/roi.json'
    video_name = "Raw_Car_Pov_Final.mp4"
    save_video = False
    save_processed_video = False
    http_connection_failed_limit = 0
    http_timeout = 5
    fps = 60
    width = 1280
    height = 720
    image_size = width * height * 3
    pipe_memory_size = image_size * 10
    visualizer_strategy = VisualizationStrategy.ALL_FRAMES
    visualizer_queue_element_count = fps * 4 * 2
    command_url = None  # 'http://10.0.0.2:8080/control'
    video_feed_memory_name = "video_feed"
    control_loop_memory_name = "control_loop"
    visualization_memory_name = "visualizer"

    def __init__(self):
        raise Exception("This class is not meant to be instantiated")

    @staticmethod
    def print_config():
        print("Config:")
        config_attrs = vars(Config)
        for attr, value in config_attrs.items():
            if not attr.startswith('__') and not callable(value):
                print(f"  {attr}: {value}")
        print()