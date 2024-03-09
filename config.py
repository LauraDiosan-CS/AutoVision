class Config:
    models_dir_path = 'objects/models/'
    videos_dir = 'videos'
    recordings_dir = 'videos/recordings'
    pipeline_config_path = 'objects/config/parallel_pipeline_config.json'
    roi_config_path = 'objects/config/roi.json'
    video_name = 'curve_90_1.mp4'
    apply_visualizer = True
    save_video = False
    save_processed_video = False
    fps = 30
    width = 1920
    height = 1080
    command_url = 'http://10.0.0.2:8080/control'

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