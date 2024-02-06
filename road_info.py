from collections import namedtuple

Line = namedtuple("Line", ["upper_x", "upper_y", "lower_x", "lower_y"])

RoadMarkings = namedtuple("RoadMarkings", ["left_line","center_line","right_line"])