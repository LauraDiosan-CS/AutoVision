import multiprocessing as mp

class ControlledProcess(mp.Process):
    __slots__ = ['setup','start_time']

    def __init__(self, program_start_time: float, name=None):
        super().__init__(name=name)

        self.program_start_time = program_start_time
        self.setup = mp.Lock()
        self.setup.acquire()

    def wait_for_setup(self):
        self.setup.acquire()

    def finish_setup(self):
        self.setup.release()