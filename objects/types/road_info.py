from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])

Line = namedtuple("Line", ["upper_point", "lower_point"])

RoadMarkings = namedtuple("RoadMarkings", ["left_line","center_line","right_line"])