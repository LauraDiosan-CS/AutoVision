import time


class TimingInfo:
    def __init__(self):
        """
        !!! DO NOT REPEAT LABEL NAMES ACROSS DIFFERENT BRANCHES/HIERARCHIES !!!

        :param program_start_time_s: The program's start time in seconds (e.g. time.time_ns() / 1e9).
        """
        # Active timers: label -> start time (in seconds)
        self.start_times = {}
        # Paused timers: label -> paused duration so far (in seconds)
        self.paused_times = {}
        # Accumulated timings: label -> total elapsed (in seconds)
        self.timings = {}
        # Counts of how many times this label has been started/stopped
        self.counts = {}
        # Hierarchy (parent -> list of children labels)
        self.hierarchy = {}
        # Root label for this timing hierarchy
        self.root_label = None
        # Keep track of when this entire process started (in seconds)

    def __str__(self):
        def format_time(time_value_s: float) -> str:
            """Format a time duration (in seconds) to a human-readable string."""
            units = [("s", 1), ("min", 60), ("h", 3600)]
            # Default to milliseconds if it's very small
            time_str = f"{time_value_s * 1000:.0f} ms"
            for unit, limit in reversed(units):
                if time_value_s >= limit:
                    time_str = f"{time_value_s / limit:.2f} {unit}"
                    break
            return time_str

        def get_total_time(label: str) -> float:
            """Get the total elapsed time (in seconds) for a label."""
            total_time = self.timings.get(label, 0.0)
            if label in self.start_times:
                # If it's still active, add time since it was last started
                now_s = time.time_ns() / 1e9
                total_time += now_s - self.start_times[label]
            return total_time

        def build_hierarchy(label: str, indent: int = 0):
            lines = []
            total_time_s = get_total_time(label)
            count = self.counts.get(label, 0)
            avg_time_s = total_time_s / count if count > 0 else 0.0

            formatted_total_time = format_time(total_time_s)
            formatted_avg_time = format_time(avg_time_s)
            status = "(active)" if label in self.start_times else "(inactive)"

            lines.append(
                f"{'  ' * indent}{label}: "
                f"total {formatted_total_time}, "
                f"avg {formatted_avg_time}, "
                f"count {count} {status}"
            )

            if label in self.hierarchy:
                for child in self.hierarchy[label]:
                    lines.extend(build_hierarchy(child, indent + 1))
            return lines

        if self.root_label:
            hierarchy_lines = build_hierarchy(self.root_label)
            hierarchy_str = "\n".join(hierarchy_lines)
        else:
            hierarchy_str = "No active hierarchy."

        return f"\n\n{hierarchy_str}\n"

    __repr__ = __str__

    def start(self, label: str, parent: str = None, extra_time_seconds: float = 0.0):
        """Start a timer for `label`. If `parent` is provided, link in hierarchy."""
        if label in self.start_times:
            print(f"Timer '{label}' is already started.")
            return

        if parent is not None and parent not in self.start_times:
            raise ValueError(
                f"Parent timer '{parent}' is not started. Cannot start '{label}'."
            )

        # Convert from nanoseconds to seconds
        now_s = time.time_ns() / 1e9 - extra_time_seconds
        self.start_times[label] = now_s

        # If no parent, this is (or should be) the root label
        if parent is None:
            if self.root_label == label:
                return
            if self.root_label:
                raise ValueError(
                    f"Root label is already set to '{self.root_label}'. Cannot set to '{label}'."
                )
            self.root_label = label
            return

        # Otherwise, attach this label under its parent
        if parent not in self.hierarchy:
            self.hierarchy[parent] = []
        if label not in self.hierarchy[parent]:  # Avoid duplicates
            self.hierarchy[parent].append(label)

    def stop(self, label: str):
        now_s = time.time_ns() / 1e9
        """Stop the timer for `label`. This also stops any active children."""
        if label not in self.start_times:
            # Timer not started or already stopped
            return

        # Stop active children first
        if label in self.hierarchy:
            for child in self.hierarchy[label]:
                if child in self.start_times:
                    self.stop(child)

        elapsed_s = now_s - self.start_times.pop(label)

        if label in self.timings:
            self.timings[label] += elapsed_s
            self.counts[label] += 1
        else:
            self.timings[label] = elapsed_s
            self.counts[label] = 1

    def remove_recursive(self, label: str):
        """Remove `label` and all children from the hierarchy and internal records."""
        if label in self.hierarchy:
            for child in self.hierarchy[label]:
                self.remove_recursive(child)
            del self.hierarchy[label]

        if label in self.start_times:
            del self.start_times[label]

        if label in self.timings:
            del self.timings[label]

        if label in self.counts:
            del self.counts[label]

    def pause_all(self):
        """Pause (stop) all currently-active timers, but remember how long they were active."""
        if not self.start_times:
            print("No active timers to pause.")
            return

        now_s = time.time_ns() / 1e9
        for label in list(self.start_times.keys()):
            elapsed_s = now_s - self.start_times.pop(label)
            self.paused_times[label] = elapsed_s

            if label not in self.timings:
                # First time this label is 'stopped'
                self.timings[label] = elapsed_s
                self.counts[label] = 1
            else:
                self.timings[label] += elapsed_s
                self.counts[label] += 1

    def restart_all(self):
        """Restart all timers that were paused by `pause_all()`."""
        if not self.paused_times:
            print("No timers to restart.")
            return

        now_s = time.time_ns() / 1e9
        for label in list(self.paused_times.keys()):
            paused_duration_s = self.paused_times.pop(label)
            # We start it as if it started `paused_duration_s` seconds ago
            self.start_times[label] = now_s - paused_duration_s

            # Adjust the previous cumulative timing so that the partial is 'undone'
            if self.counts[label] == 1:
                # If we've only accounted for it once, remove that record
                del self.timings[label]
            else:
                self.timings[label] -= paused_duration_s
                self.counts[label] -= 1

    def append_hierarchy(self, other: "TimingInfo", parent_label_of_other: str = None):
        """
        Append `other`'s hierarchy under `parent_label_of_other` in this `TimingInfo`.
        Combine timings, counts, start_times, and paused_times.
        """
        if parent_label_of_other is None:
            parent_label_of_other = self.root_label  # Default to the current root

        # Ensure the parent label exists
        if (
            parent_label_of_other not in self.hierarchy
            and parent_label_of_other != self.root_label
        ):
            print(
                f"Label '{parent_label_of_other}' does not exist in the current hierarchy."
            )
            raise ValueError(
                f"Label '{parent_label_of_other}' does not exist in the current hierarchy."
            )

        # Ensure `other` has a root label
        if other.root_label is None:
            print("The other TimingInfo has no root label.")
            raise ValueError("The other TimingInfo has no root label.")

        # Add other's root under this parent's hierarchy (if not already present)
        if self.root_label != other.root_label:
            if parent_label_of_other not in self.hierarchy:
                self.hierarchy[parent_label_of_other] = []
            if other.root_label not in self.hierarchy[parent_label_of_other]:
                self.hierarchy[parent_label_of_other].append(other.root_label)

        # Recursively merge the entire hierarchy
        def merge_hierarchy(src_hierarchy, dst_hierarchy, src_label):
            if src_label in src_hierarchy:
                # Ensure this label is in the destination
                if src_label not in dst_hierarchy:
                    dst_hierarchy[src_label] = []
                # Merge children
                for child in src_hierarchy[src_label]:
                    if child not in dst_hierarchy[src_label]:
                        dst_hierarchy[src_label].append(child)
                    merge_hierarchy(src_hierarchy, dst_hierarchy, child)

        merge_hierarchy(other.hierarchy, self.hierarchy, other.root_label)

        # Merge timings and counts
        for key, value in other.timings.items():
            if key in self.timings:
                self.timings[key] += value
                self.counts[key] += other.counts[key]
            else:
                self.timings[key] = value
                self.counts[key] = other.counts[key]

        # Merge active timers and paused timers
        self.start_times.update(other.start_times)
        self.paused_times.update(other.paused_times)


if __name__ == "__main__":
    # Create the first TimingInfo object
    timing1 = TimingInfo()
    timing1.start("A")  # Root
    timing1.start("B", "A")
    timing1.start("C", "B")
    timing1.stop("C")
    timing1.stop("B")
    timing1.stop("A")

    # Create the second TimingInfo object
    timing2 = TimingInfo()
    timing2.start("X")  # Root of the second hierarchy
    timing2.start("Y", "X")
    timing2.start("Z", "Y")
    time.sleep(0.1)
    timing2.stop("Z")
    time.sleep(0.2)
    timing2.stop("Y")
    time.sleep(0.3)
    timing2.stop("X")

    # Create the third TimingInfo object
    timing3 = TimingInfo()
    timing3.start("M")  # Root of the third hierarchy
    timing3.start("M.Y", "M")
    timing3.start("M.Z", "M.Y")
    time.sleep(0.4)
    timing3.stop("M.Z")
    time.sleep(0.7)
    timing3.stop("M.Y")
    time.sleep(0.1)
    timing3.stop("M")

    # Append timing2's hierarchy to timing1 under 'B'
    timing1.append_hierarchy(timing2, "B")

    print("Timing1 Hierarchy after appending Timing2:")
    print(timing1)

    # Append timing2's hierarchy to timing1 under 'B' again
    timing1.append_hierarchy(timing2, "B")
    print("Timing1 Hierarchy after appending Timing2 again:")
    print(timing1)

    # Append timing3's hierarchy to timing1 under 'B'
    timing1.append_hierarchy(timing3, "B")
    print("Timing1 Hierarchy after appending Timing3:")
    print(timing1)