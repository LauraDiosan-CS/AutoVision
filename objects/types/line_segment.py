import numpy as np


class LineSegment:
    def __init__(self,lower_x, lower_y, upper_x, upper_y):
        self.coordinates = np.array([[lower_x, lower_y], [upper_x, upper_y]])
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

    def compute_euclidean_distance(self):
        """
        Computes the Euclidean distance between the two endpoints of the line segment.
        """
        return np.sqrt((self.upper_x - self.lower_x) ** 2 + (self.upper_y - self.lower_y) ** 2)

    def compute_angle_with_ox_radians(self):
        """
        Computes the angle (in radians) that the line segment makes with the Ox axis.
        """
        return np.arctan2(self.upper_y - self.lower_y, self.upper_x - self.lower_x)

    def compute_intersecting_x_coordinate(self, horizontal_line_y):
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

    def check_is_horizontal(self, threshold_degrees):
        if threshold_degrees < 0 or threshold_degrees > 30:
            raise ValueError("The threshold angle must be between 0 and 30 degrees")
        threshold_radians = np.deg2rad(threshold_degrees)
        angle_rads = self.compute_angle_with_ox_radians()
        return abs(angle_rads) < threshold_radians

    def __iter__(self):
        """
        Allows the line segment to be iterated over, yielding its endpoints in the following order: x1, y1, x2, y2.

        Returns: x1, y1, x2, y2
        """
        return iter(self.coordinates.ravel())

    def __repr__(self):
        return f"LineSegment({self.lower_x}, {self.lower_y}, {self.upper_x}, {self.upper_y}) with slope {self.slope} \n"