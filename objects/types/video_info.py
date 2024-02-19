from collections import namedtuple

VideoRois = dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]]

VideoInfo = namedtuple("VideoInfo", ["video_name", "video_rois", "height", "width"])