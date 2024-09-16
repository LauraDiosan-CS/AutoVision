import time
from itertools import count


class TimingInfo:
    def __init__(self):
        self.start_times = {}
        self.paused_times = {}
        self.timings = {}
        self.counts = {}
        self.hierarchy = {}
        self.root_label = None

    def __str__(self):
        return f"Timings: {self.timings}, Counts: {self.counts}, Hierarchy: {self.hierarchy}"

    __repr__ = __str__

    def start(self, label, parent=None, extra_time_seconds=0):
        if label in self.start_times:
            print(f"Timer '{label}' is already started.")
            return

        if parent is not None and parent not in self.start_times:
            raise ValueError(f"Parent timer '{parent}' is not started. Cannot start '{label}'.")

        self.start_times[label] = time.time() - extra_time_seconds

        if parent is None:
            if self.root_label:
                raise ValueError(f"Root label is already set to '{self.root_label}'. Cannot set to '{label}'.")
            self.root_label = label
            return

        if parent not in self.hierarchy:
            self.hierarchy[parent] = []
        if label not in self.hierarchy[parent]:  # Avoid duplicates
            self.hierarchy[parent].append(label)

    def stop(self, label):
        if label not in self.start_times:
            # print(f"Timer for '{label}' was not started.")
            return
        if label in self.hierarchy:
            for child in self.hierarchy[label]:
                if child in self.start_times:
                    # print(f"Stopping child: {child} for parent: {label}")
                    self.stop(child)

        end_time = time.time()
        elapsed = end_time - self.start_times.pop(label)
        if label in self.timings:
            self.timings[label] += elapsed
            self.counts[label] += 1
        else:
            self.timings[label] = elapsed
            self.counts[label] = 1


    def pause_all(self):
        if not self.start_times:
            print("No active timers to pause.")
            return
        current_time = time.time()
        for label in list(self.start_times.keys()):
            self.paused_times[label] = current_time - self.start_times.pop(label)
            if label not in self.timings: # Check if this would be the first time this label is stopped
                self.timings[label] = self.paused_times[label]
                self.counts[label] = 1
            else:
                self.timings[label] += self.paused_times[label]
                self.counts[label] += 1

    def restart_all(self):
        if not self.paused_times:
            print("No timers to restart.")
            return
        current_time = time.time()
        for label in list(self.paused_times.keys()):
            value = self.paused_times.pop(label)
            self.start_times[label] = current_time - value

            if self.counts[label] == 1: # Check if this label was only stopped by pause_all
                del self.timings[label]
            else:
                self.timings[label] -= value
                self.counts[label] -= 1

    def append_hierarchy(self, other, label: str):
        # Ensure the label exists in the current hierarchy
        if label not in self.hierarchy and label != self.root_label:
            raise ValueError(f"Label '{label}' does not exist in the current hierarchy.")

        # Ensure the other TimingInfo has a root
        if other.root_label is None:
            raise ValueError("The other TimingInfo has no root label.")

        # Append the other root label under the current label
        if label not in self.hierarchy:
            self.hierarchy[label] = []
        self.hierarchy[label].append(other.root_label)

        # Recursively copy the hierarchy of the other TimingInfo to self
        def copy_hierarchy(src_hierarchy, dst_hierarchy, src_label):
            if src_label in src_hierarchy:
                dst_hierarchy[src_label] = []
                for child in src_hierarchy[src_label]:
                    dst_hierarchy[src_label].append(child)
                    copy_hierarchy(src_hierarchy, dst_hierarchy, child)

        # Merge the hierarchy
        copy_hierarchy(other.hierarchy, self.hierarchy, other.root_label)

        # Merge timings and counts (since labels are unique, we can just update)
        self.timings.update(other.timings)
        self.counts.update(other.counts)

        # Merge start_times and paused_times
        self.start_times.update(other.start_times)
        self.paused_times.update(other.paused_times)



if __name__ == '__main__':
    # Create the first TimingInfo object
    timing1 = TimingInfo()
    timing1.start('A')  # Root
    timing1.start('B', 'A')
    timing1.start('C', 'B')
    timing1.stop('C')
    timing1.stop('B')
    timing1.stop('A')

    # Create the second TimingInfo object
    timing2 = TimingInfo()
    timing2.start('X')  # Root of the second hierarchy
    timing2.start('Y', 'X')
    timing2.start('Z', 'Y')
    timing2.stop('Z')
    timing2.stop('Y')
    timing2.stop('X')

    # Append timing2's hierarchy to timing1 under 'B'
    timing1.append_hierarchy(timing2, 'B')

    # Display the results
    print("Timing1 Hierarchy after appending Timing2:")
    print(timing1)