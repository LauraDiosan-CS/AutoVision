import time
from typing import Optional, Tuple

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, target_value: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_value = target_value

        self.input_bounds: Optional[Tuple[float, float]] = None
        self.output_bounds: Optional[Tuple[float, float]] = None

        self.previous_time: float = time.perf_counter_ns() # time in nanoseconds
        self.cumulative_error: float = 0.0
        self.last_error: float = 0.0
        self.last_output: float = 0.0

        self.delta_time_ms_min = 1e-3

    def set_input_range(self, min_value: float, max_value: float) -> None:
        assert min_value < max_value, "Minimum input must be less than maximum input."
        self.input_bounds = (min_value, max_value)

    def set_output_range(self, min_value: float, max_value: float) -> None:
        assert min_value < max_value, "Minimum output must be less than maximum output."
        self.output_bounds = (min_value, max_value)

    def reset(self) -> None:
        self.previous_time = time.perf_counter_ns()
        self.cumulative_error = 0.0
        self.last_error = 0.0
        self.last_output = 0.0

    def _get_error(self, input_value: float) -> float:
        if self.input_bounds:
            input_value = max(min(input_value, self.input_bounds[1]), self.input_bounds[0])
        return self.target_value - input_value

    def compute(self, input_value: float, verbose: bool = False) -> float:
        error = self._get_error(input_value)
        current_time = time.perf_counter_ns()
        delta_time_ms = max((current_time - self.previous_time) / 1e-6, self.delta_time_ms_min)

        # Proportional Term
        proportional = self.kp * error

        # Integral Term
        if self.output_bounds is None or (
            self.output_bounds[0] <= self.last_output <= self.output_bounds[1]
        ) or (self.cumulative_error * error < 0):
            self.cumulative_error += error * delta_time_ms
        integral = self.ki * self.cumulative_error

        # Derivative Term
        derivative = self.kd * (error - self.last_error) / delta_time_ms

        # Total Output
        output = proportional + integral + derivative
        if self.output_bounds:
            output = max(min(output, self.output_bounds[1]), self.output_bounds[0])

        # Update state
        self.last_output = output
        self.last_error = error
        self.previous_time = current_time

        if verbose:
            print(f"Error: {error}, P: {proportional}, I: {integral}, D: {derivative}, Output: {output}")

        return output