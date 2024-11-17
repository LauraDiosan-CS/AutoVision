from collections import namedtuple

Coord = tuple[int, int]
VideoRois = dict[str, tuple[Coord, Coord, Coord, Coord]]

VideoInfo = namedtuple("VideoInfo", ["video_name", "video_rois", "height", "width"])