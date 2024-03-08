from collections import namedtuple

RoadMarkings = namedtuple("RoadMarkings", ["left_line", "center_line", "center_line_virtual",
                                           "right_line", "right_line_virtual", "stop_lines"])

RoadObject = namedtuple("RoadObject", ["bbox", "label", "conf", "distance"])