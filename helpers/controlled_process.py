import multiprocessing as mp


class ControlledProcess(mp.Process):
    __slots__ = ['setup']

    def __init__(self, name=None):
        super().__init__(name=name)

        self.setup = mp.Lock()
        self.setup.acquire()

    def start(self):
        super().start()
        self.setup.acquire()

    def finish_setup(self):
        self.setup.release()
