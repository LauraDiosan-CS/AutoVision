from dataclasses import dataclass
from typing import Optional
from perception.objects.line_segment import LineSegment

@dataclass(slots=True)
class RoadObject:
    bbox: list[float]
    label: str
    conf: float
    distance: float

@dataclass(slots=True)
class RoadMarkings:
    left_line: Optional[LineSegment]
    center_line: Optional[LineSegment]
    center_line_virtual: bool
    right_line: Optional[LineSegment]
    right_line_virtual: bool
    stop_lines: list[LineSegment]