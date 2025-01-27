import multiprocessing as mp

class ControlledProcess(mp.Process):
    def __init__(self, program_start_time: float, name=None):
        super().__init__(name=name)

        self.program_start_time = program_start_time