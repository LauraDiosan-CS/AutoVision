from enum import Enum

from objects.types.road_info import RoadObject


class Behaviour(Enum):
    LaneKeeping = "LaneKeeping"
    Pause = "Pause"
    Pause3Seconds = "Pause3Seconds"
    Resume = "Resume"


class BehaviourPlanner:
    PEDESTRIAN_THRESHOLD_DISTANCE = 100
    TRAFFIC_LIGHT_THRESHOLD_DISTANCE = 100
    TRAFFIC_SIGN_THRESHOLD_DISTANCE = 100

    def __init__(self):
        self.stop_after_line_invisible = False
        self.reason = ""
        self.paused = False

    def run_iteration(self, traffic_signs: list[RoadObject],
                      horizontal_lines: list[RoadObject],
                      traffic_lights: list[RoadObject],
                      pedestrians: list[RoadObject],
                      ) -> Behaviour:

        command = Behaviour.LaneKeeping

        if self.paused:
            if (len(pedestrians) == 0 or pedestrians[0].distance > self.PEDESTRIAN_THRESHOLD_DISTANCE) \
                    and (len(traffic_signs) == 0 or
                         traffic_lights[0].distance > self.TRAFFIC_LIGHT_THRESHOLD_DISTANCE or
                         not any(map(lambda traffic_light: traffic_light.label == "red", traffic_lights)
                                 )
            ):
                self.paused = False
                command = Behaviour.Resume
        elif self.stop_after_line_invisible:
            if len(horizontal_lines) > 0:
                self.stop_after_line_invisible = False
                match self.reason:
                    case "Stop":
                        command = Behaviour.Pause3Seconds
                    case "Pedestrian":
                        command = Behaviour.Pause
                        self.paused = True
                    case "Red light":
                        command = Behaviour.Pause
                        self.paused = True
        elif len(traffic_signs) > 0 and traffic_signs[0].distance < self.TRAFFIC_SIGN_THRESHOLD_DISTANCE:
            match traffic_signs[0].label:
                case "stop":
                    self.stop_after_line_invisible = True
                    self.reason = "Stop"
                case "parking":
                    pass
        elif len(pedestrians) > 0 and pedestrians[0].distance < self.PEDESTRIAN_THRESHOLD_DISTANCE:
            self.stop_after_line_invisible = True
            self.reason = "Pedestrian"
        elif len(traffic_lights) > 0:
            close_lights = filter(lambda traffic_light: traffic_light.distance < self.TRAFFIC_LIGHT_THRESHOLD_DISTANCE,
                                  traffic_lights)
            if any(map(lambda traffic_light: traffic_light.label == "red", close_lights)):
                self.stop_after_line_invisible = True
                self.reason = "Red light"

        return command