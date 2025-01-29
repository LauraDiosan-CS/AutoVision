import multiprocessing as mp
from rs_ipc.rs_ipc import ReaderWaitPolicy, SharedMessage, OperationMode

from perception.objects.timingvisualizer import TimingVisualizer


class TimingManagerProcess(mp.Process):
    def __init__(self, program_start_time: float = 0.0, process_name: str = None):
        super().__init__(name=process_name)
        self.program_start_time = program_start_time

    def run(self):
        while True: