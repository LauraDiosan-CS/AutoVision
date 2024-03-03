from objects.types.road_info import RoadObject


class BehaviourPlanner:
    HORIZ_LINE_THRESHOLD_DISTANCE = 100
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
                      ) -> str:

        command = "Lanekeeping"

        if self.paused:
            if (len(pedestrians) == 0 or pedestrians[0].distance > self.PEDESTRIAN_THRESHOLD_DISTANCE) \
                    and (len(traffic_signs) == 0 or
                         traffic_lights[0].distance > self.HORIZ_LINE_THRESHOLD_DISTANCE or
                         not any(map(lambda traffic_light: traffic_light.label == "red", traffic_lights)
                                 )
            ):
                self.paused = False
                command = "Resume"
        elif self.stop_after_line_invisible:
            if all(map(lambda horizontal_Line: horizontal_Line.distance > self.HORIZ_LINE_THRESHOLD_DISTANCE,
                       horizontal_lines)):
                self.stop_after_line_invisible = False
                match self.reason:
                    case "Stop":
                        command = "Pause 3"
                    case "Pedestrian":
                        command = "Pause"
                        self.paused = True
                    case "Red light":
                        command = "Pause"
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