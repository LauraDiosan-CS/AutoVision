class Config:
    models_dir_path = 'objects/models/'
    videos_dir = 'videos'
    recordings_dir = 'videos/recordings'
    pipeline_config_path = 'objects/config/lane_detect_config.json'
    roi_config_path = 'objects/config/roi.json'
    video_name = 'qualifiers_2.mp4'
    visualize_only_final = False
    save_video = True
    fps = 30
    width = 1920
    height = 1080
    command_url = 'http://car@car:8080/control'

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