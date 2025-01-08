import os
from enum import Enum

class VisualizationStrategy(Enum):
    ALL_FRAMES = 1
    NEWEST_FRAME = 2

class MultiprocessingStrategy(Enum):
    LIVE = 1
    ALL_FRAMES_FASTEST_PROCESS = 2
    ALL_FRAMES_ALL_PROCESSES = 3



class Config:
    # Directories
    models_dir_path = 'configuration/models/'
    perception_config_dir = 'configuration/perception_config/'
    videos_dir = 'files/videos'
    recordings_dir = os.path.join(videos_dir, 'recordings')

    # Config Files
    pipeline_config_path = os.path.join(perception_config_dir, 'parallel_pipeline.json')
    roi_config_path = os.path.join(perception_config_dir, 'roi.json')

    # Video to process
    video_name = "Raw_Car_Pov_Final.mp4"
    color_channels = 3
    fps = 30
    width = 640 * 2
    height = 360 * 2

    # General Config
    shared_memory_size_multiplier = 10
    visualizer_queue_element_count = fps * 8
    save_queue_element_count = fps

    save_processed_video = True
    visualizer_strategy = VisualizationStrategy.NEWEST_FRAME
    mp_strategy = MultiprocessingStrategy.ALL_FRAMES_FASTEST_PROCESS

    # Shared Memory Config
    frame_size = width * height * color_channels
    shared_memory_size = frame_size * shared_memory_size_multiplier

    # Shared Memory Names
    video_feed_memory_name = "video_feed"
    control_loop_memory_name = "control_loop"
    visualization_memory_name = "visualizer"
    save_raw_memory_name = "save_raw"
    save_final_memory_name = "save_final"

    # HTTP Config
    http_connection_failed_limit = 0
    http_timeout = 5
    command_url = None  # 'http://10.0.0.2:8080/control'

    def __init__(self):
        raise Exception("Config class is not meant to be instantiated")

    @classmethod
    def as_json(cls):
        import json
        relevant_keys = ['video_name', 'width', 'height', 'fps', 'shared_memory_size_multiplier',
                         'visualizer_queue_element_count', 'save_queue_element_count',
                         'visualizer_strategy', 'mp_strategy']
        config_data = {
            k: v.name if isinstance(v, Enum) else v
            for k, v in cls.__dict__.items()
            if not k.startswith('__') and not callable(v) and not isinstance(v, classmethod) and k in relevant_keys
        }
        return json.dumps(config_data, indent=4)

if __name__ == "__main__":
    print(Config.as_json())