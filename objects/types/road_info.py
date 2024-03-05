from collections import namedtuple

RoadMarkings = namedtuple("RoadMarkings", ["left_line", "center_line", "right_line", "stop_lines","horizontals", "right_int", "center_int"])

RoadObject = namedtuple("RoadObject", ["bbox", "label", "conf", "distance"])