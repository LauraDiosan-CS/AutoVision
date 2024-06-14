class Config:
    models_dir_path = 'objects/models/'
    videos_dir = 'videos'
    recordings_dir = 'videos/recordings'
    screenshot_dir = 'screenshots'
    pipeline_config_path = 'objects/config/lane_detect_config.json'
    # pipeline_config_path = 'objects/config/parallel_pipeline_config.json'
    roi_config_path = 'objects/config/roi.json'
    video_name = "qualifiers_5_720.mp4" #"curved_lines_2024_1.avi"
    apply_visualizer = True
    save_video = False
    save_processed_video = False
    http_connection_failed_limit = 0
    http_timeout = 5
    fps = 10
    width = 1280 #640
    height = 720 #480
    image_size = width * height * 3
    pipe_memory_size = image_size * 10
    command_url = None  # 'http://10.0.0.2:8080/control'
    video_feed_memory_name = "video_feed"
    composite_pipe_memory_name = "visualizer"

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