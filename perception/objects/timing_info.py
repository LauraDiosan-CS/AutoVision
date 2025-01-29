import time


class TimingInfo:
    def __init__(self):
        """
        !!! DO NOT REPEAT LABEL NAMES ACROSS DIFFERENT BRANCHES/HIERARCHIES !!!
        """
        self.start_times = {}
        self.paused_times = {}
        self.timings = {}
        self.counts = {}
        self.hierarchy = {}
        self.root_label = None

    def __str__(self):
        def format_time(time_value):
            units = [("s", 1), ("min", 60), ("h", 3600)]
            time_str = f"{time_value * 1000:.0f} ms"  # Default to milliseconds if time is very small
            for unit, limit in reversed(units):
                if time_value >= limit:
                    time_str = f"{time_value / limit:.2f} {unit}"
                    break
            return time_str

        def get_total_time(label):
            total_time = self.timings.get(label, 0)
            if label in self.start_times:  # Add time since start if still active
                total_time += time.perf_counter() - self.start_times[label]
            return total_time

        def build_hierarchy(label, indent=0):
            lines = []
            total_time = get_total_time(label)
            count = self.counts.get(label, 0)
            avg_time = total_time / count if count > 0 else 0
            formatted_total_time = format_time(total_time)
            formatted_avg_time = format_time(avg_time)

            status = "(active)" if label in self.start_times else "(inactive)"
            lines.append(
                f"{'  ' * indent}{label}: total {formatted_total_time}, avg {formatted_avg_time}, count {count} {status}"
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

    def start(self, label, parent=None, extra_time_seconds=0):
        if label in self.start_times:
            print(f"Timer '{label}' is already started.")
            return

        if parent is not None and parent not in self.start_times:
            raise ValueError(
                f"Parent timer '{parent}' is not started. Cannot start '{label}'."
            )

        self.start_times[label] = time.perf_counter() - extra_time_seconds

        if parent is None:
            if self.root_label == label:
                return

            if self.root_label:
                raise ValueError(
                    f"Root label is already set to '{self.root_label}'. Cannot set to '{label}'."
                )
            self.root_label = label
            return

        if parent not in self.hierarchy:
            self.hierarchy[parent] = [label]
        if label not in self.hierarchy[parent]:  # Avoid duplicates
            self.hierarchy[parent].append(label)

    def stop(self, label):
        stopped_at = time.perf_counter()
        if label not in self.start_times:
            # print(f"Timer for '{label}' was not started.")
            return
        if label in self.hierarchy:
            for child in self.hierarchy[label]:
                if child in self.start_times:
                    # print(f"Stopping child: {child} for parent: {label}")
                    self.stop(child)

        elapsed = stopped_at - self.start_times.pop(label)
        if label in self.timings:
            self.timings[label] += elapsed
            self.counts[label] += 1
        else:
            self.timings[label] = elapsed
            self.counts[label] = 1

    def remove_recursive(self, label):
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
        if not self.start_times:
            print("No active timers to pause.")
            return
        current_time = time.perf_counter()
        for label in list(self.start_times.keys()):
            self.paused_times[label] = current_time - self.start_times.pop(label)
            if (
                label not in self.timings
            ):  # Check if this would be the first time this label is stopped
                self.timings[label] = self.paused_times[label]
                self.counts[label] = 1
            else:
                self.timings[label] += self.paused_times[label]
                self.counts[label] += 1

    def restart_all(self):
        if not self.paused_times:
            print("No timers to restart.")
            return
        current_time = time.perf_counter()
        for label in list(self.paused_times.keys()):
            value = self.paused_times.pop(label)
            self.start_times[label] = current_time - value

            if (
                self.counts[label] == 1
            ):  # Check if this label was only stopped by pause_all
                del self.timings[label]
            else:
                self.timings[label] -= value
                self.counts[label] -= 1

    def append_hierarchy(self, other, parent_label_of_other: str = None):
        if parent_label_of_other is None:
            parent_label_of_other = (
                self.root_label
            )  # Use the root label of the current TimingInfo
        # Ensure the label exists in the current hierarchy
        elif parent_label_of_other not in self.hierarchy:
            print(
                f"Label '{parent_label_of_other}' does not exist in the current hierarchy."
            )
            raise ValueError(
                f"Label '{parent_label_of_other}' does not exist in the current hierarchy."
            )

        # Ensure the other TimingInfo has a root
        if other.root_label is None:
            print("The other TimingInfo has no root label.")
            raise ValueError("The other TimingInfo has no root label.")

        # Append the other root label under the current label, only if not already present
        if self.root_label != other.root_label:
            if parent_label_of_other not in self.hierarchy:
                self.hierarchy[parent_label_of_other] = []
            if other.root_label not in self.hierarchy[parent_label_of_other]:
                self.hierarchy[parent_label_of_other].append(other.root_label)

        # Recursively merge the hierarchy of the other TimingInfo to self
        def merge_hierarchy(src_hierarchy, dst_hierarchy, src_label):
            if src_label in src_hierarchy:
                if src_label not in dst_hierarchy:
                    dst_hierarchy[src_label] = []
                for child in src_hierarchy[src_label]:
                    if child not in dst_hierarchy[src_label]:
                        dst_hierarchy[src_label].append(child)
                    merge_hierarchy(src_hierarchy, dst_hierarchy, child)

        # Merge the hierarchy
        merge_hierarchy(other.hierarchy, self.hierarchy, other.root_label)

        # Merge timings and counts by accumulating them if labels are already present
        for key, value in other.timings.items():
            if key in self.timings:
                self.timings[key] += value
                self.counts[key] += other.counts[key]
            else:
                self.timings[key] = value
                self.counts[key] = other.counts[key]

        # Merge start_times and paused_times, respecting existing timers
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

    # Append timing2's hierarchy to timing1 under 'B' for the first time
    timing1.append_hierarchy(timing2, "B")

    print("Timing1 Hierarchy after appending Timing2")
    print(timing1)
    # Repeat appending timing2's hierarchy to timing1 under 'B' again
    timing1.append_hierarchy(timing2, "B")
    print("Timing1 Hierarchy after appending Timing2 again:")
    print(timing1)

    # Append timing3's hierarchy to timing1 under 'B' for the first time
    timing1.append_hierarchy(timing3, "B")
    print("Timing1 Hierarchy after appending Timing3")
    print(timing1)