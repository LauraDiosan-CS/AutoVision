import numpy as np


class LineSegment:
    def __init__(self, lower_x: int, lower_y: int, upper_x: int, upper_y: int):
        self.coordinates = np.array([[lower_x, lower_y], [upper_x, upper_y]], dtype=int)
        self.slope = (upper_y - lower_y) / (upper_x - lower_x) if upper_x - lower_x != 0 else np.inf

    @property
    def lower_x(self):
        return self.coordinates[0][0]

    @property
    def lower_y(self):
        return self.coordinates[0][1]

    @property
    def lower_point(self):
        return self.coordinates[0]

    @property
    def upper_x(self):
        return self.coordinates[1][0]

    @property
    def upper_y(self):
        return self.coordinates[1][1]

    @property
    def upper_point(self):
        return self.coordinates[1]

    def compute_vertical_distance(self):
        """
        Computes the vertical distance between the two endpoints of the line segment.
        """
        return abs(self.upper_y - self.lower_y)

    def compute_euclidean_distance(self) -> float:
        """
        Computes the Euclidean distance between the two endpoints of the line segment.
        """
        return np.sqrt((self.upper_x - self.lower_x) ** 2 + (self.upper_y - self.lower_y) ** 2)

    def compute_angle_with_ox_radians(self) -> float:
        """
        Computes the angle (in radians) that the line segment makes with the Ox axis.
        """
        return np.arctan2(abs(self.upper_y - self.lower_y), abs(self.upper_x - self.lower_x))

    def compute_intersecting_x_coordinate(self, horizontal_line_y) -> int:
        """
        Computes the x-coordinate of the intersection point between the line segment and a horizontal line.

        Args:
        - horizontal_line_y: The y-coordinate of the horizontal line.

        Returns:
        - The x-coordinate of the intersection point.
        """
        if self.slope == np.inf or self.slope == -np.inf:  # Vertical line
            print("Vertical line")
            return self.lower_x  # Intersection of vertical line with any horizontal line is its x-coordinate
        elif self.slope == 0:  # Horizontal line
            raise ValueError(f"The line segment {self.__repr__()} is horizontal, so it doesn't intersect with the "
                             f"horizontal line y={horizontal_line_y}")
        else:
            y_intercept = self.lower_y - self.slope * self.lower_x
            return int((horizontal_line_y - y_intercept) / self.slope)

    def check_is_horizontal(self, threshold_degrees) -> bool:
        if threshold_degrees < 0 or threshold_degrees > 45:
            raise ValueError("The threshold angle must be between 0 and 45 degrees")
        threshold_radians = np.deg2rad(threshold_degrees)
        angle_rads = self.compute_angle_with_ox_radians()
        return abs(angle_rads) < threshold_radians

    def discretize(self, num_points) -> list[tuple[int, int]]:
        """
        Discretizes the line segment into a certain number of segments.

        Args:
        - num_segments: The number of segments to divide the line segment into.

        Returns:
        - List of coordinates representing points along the line segment, evenly spaced.
        """
        x_values = np.linspace(self.lower_x, self.upper_x, num=num_points, dtype=int)
        y_values = np.linspace(self.lower_y, self.upper_y, num=num_points, dtype=int)
        return list(zip(x_values, y_values))

    def compute_distance_to_point(self, point: tuple[int, int]) -> float:
        """
        Computes the distance between the line segment and a point.

        Args:
        - point: The point to compute the distance to.

        Returns:
        - The distance between the line segment and the point.
        """
        x, y = point
        return (abs((self.upper_y - self.upper_x) * x -
                    (self.lower_y - self.lower_x) * y +
                    self.lower_y * self.upper_x - self.upper_y * self.lower_x) /
                self.compute_euclidean_distance())

    def compute_interesting_point(self, other):
        """
        Computes the intersection point between the two line segments.
        Args:
        - other: The other line segment.
        Returns:
        - The coordinates of the intersection point.
        """
        x1, y1 = self.lower_x, self.lower_y
        x2, y2 = self.upper_x, self.upper_y
        x3, y3 = other.lower_x, other.lower_y
        x4, y4 = other.upper_x, other.upper_y
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return x, y


    def __iter__(self):
        """
        Allows the line segment to be iterated over, yielding its endpoints in the following order: lower_x, lower_y,
        upper_x, upper_y

        Returns: lower_x, lower_y, upper_x, upper_y
        """
        return iter(self.coordinates.ravel())

    def __repr__(self):
        return f"LineSegment({self.lower_x}, {self.lower_y}, {self.upper_x}, {self.upper_y}) with slope {self.slope} \n"