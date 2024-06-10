import colorsys
import itertools
import time

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def complex_calculation():
    # time.sleep(np.random.rand())  # Simulate time-consuming computation
    time.sleep(1)

def load_resources():
    # time.sleep(np.random.rand() * 0.5)  # Simulate loading time
    time.sleep(0.5)


def data_processing():
    # time.sleep(np.random.rand() * 0.3)  # Simulate data processing
    time.sleep(0.3)


class Timer:
    def __init__(self):
        self.axs = None
        self.fig = None
        self.total_charts = None
        self.start_times = {}
        self.timings = {}
        self.counts = {}
        self.hierarchy = {}
        self.root_label = None

        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': (0.2, 0.3, 0.3),  # Dark background for the figure
            'axes.edgecolor': 'white',  # White edges for the axes
            'axes.labelcolor': 'white',  # White labels
            'xtick.color': 'white',  # White x-tick labels
            'ytick.color': 'white',  # White y-tick labels
            'text.color': 'white',  # White text
            'grid.color': 'gray',  # Gray grid lines
            'grid.alpha': 0.5,  # Slightly transparent grid lines
            'axes.prop_cycle': plt.cycler('color', plt.cm.tab10.colors)  # Bright color cycle
        })

    def start(self, label, parent=None):
        if label in self.start_times:
            print(f"Timer '{label}' is already started.")
            return

        self.start_times[label] = time.time()

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
            print(f"Timer for '{label}' was not started.")
            return
        if label in self.hierarchy:
            for child in self.hierarchy[label]:
                if child in self.start_times:
                    print(f"Stopping child: {child} for parent: {label}")
                    self.stop(child)

        end_time = time.time()
        elapsed = end_time - self.start_times.pop(label)
        if label in self.timings:
            self.timings[label] += elapsed
            self.counts[label] += 1
        else:
            self.timings[label] = elapsed
            self.counts[label] = 1

    def calculate_averages(self):
        return {label: self.timings[label] / self.counts[label] for label in self.timings}

    def stop_and_store_running_timers(self):
        running_timers = {}

        if self.root_label in self.start_times:
            running_timers[self.root_label] = time.time() - self.start_times.pop(self.root_label)
            if self.root_label not in self.timings:
                self.timings[self.root_label] = running_timers[self.root_label]
                self.counts[self.root_label] = 1
            else:
                self.timings[self.root_label] += running_timers[self.root_label]
                self.counts[self.root_label] += 1

        for label in list(self.start_times.keys()):
            running_timers[label] = time.time() - self.start_times.pop(label)
        return running_timers

    def restart_timers(self, running_timers):
        if self.root_label in running_timers:
            self.start_times[self.root_label] = time.time() - running_timers[self.root_label]
            self.timings[self.root_label] -= running_timers[self.root_label]
            self.counts[self.root_label] -= 1
            running_timers.pop(self.root_label)
        for label, elapsed in running_timers.items():
            self.start_times[label] = time.time() - elapsed
    def plot_pie_charts(self):
        running_timers = self.stop_and_store_running_timers()
        # for parent in self.hierarchy:
        #     print(f"Parent: {parent}")
        #     for child in self.hierarchy[parent]:
        #         print(f"Child: {child} - {self.timings[child]:.3f} s")

        total_charts = sum(1 for children in self.hierarchy.values() if children)

        if self.axs is not None:
            for ax in self.axs:
                ax.clear()
        if self.total_charts != total_charts or self.fig is None or self.axs is None:
            cols = min(total_charts, 3)  # Up to 3 columns of charts
            rows = (total_charts + cols - 1) // cols  # Compute rows needed

            self.fig, self.axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            self.axs = self.axs.flatten() if total_charts > 1 else [self.axs]
            self.total_charts = total_charts

        color_cycle = itertools.cycle(plt.get_cmap('tab10').colors)
        unique_labels = set(self.hierarchy[self.root_label])
        self.colors = {label: next(color_cycle) for label in unique_labels}
        self.colors[self.root_label] = next(color_cycle)

        node_queue = [(self.root_label, 0)]  # Start with the root node
        plotted_indices = set()  # Keep track of the indices that have been plotted
        while node_queue:
            parent, idx = node_queue.pop(0)
            plotted_indices.add(idx)

            if parent not in self.timings:
                print(f"Skipping '{parent}' as no timing data is available. This is likely a bug.")
                continue

            children = self.hierarchy.get(parent, [])
            labels = [child for child in children if child in self.timings]
            times = [self.timings[child] for child in labels]
            averages = self.calculate_averages()
            formatted_labels = [self.get_formatted_label(label, self.timings[label], averages[label]) for label in
                                labels]

            parent_color = self.colors[parent]
            if parent == self.root_label:
                child_colors = [self.colors[label] for label in labels]
            else:
                child_colors = self.get_color_variations(parent_color, len(labels))
                for vairation, label in zip(child_colors, labels):
                    self.colors[label] = vairation

            children_total_time = sum(times)
            parent_time = self.timings[parent]

            # If there's a difference, add "other" to the pie chart
            if parent_time > children_total_time:
                labels.append("Other")
                times.append(parent_time - children_total_time)
                formatted_labels.append(self.get_formatted_label("Other", parent_time - children_total_time,
                                                                 (parent_time - children_total_time) / self.counts[
                                                                     parent]))
                child_colors.append(parent_color)  # Use parent color for "Other"


            filtered_labels = [label if time / parent_time >= 0.03 else '' for label, time in
                                zip(formatted_labels, times)]
            times = [time + 0.0 for time in times]
            self.axs[idx].pie(times, labels=filtered_labels, autopct='%1.1f%%',
                         startangle=180, shadow=True, colors=child_colors, explode=[0.05] * len(times))
            title = self.get_formatted_title(parent, parent_time, averages[parent])
            self.axs[idx].set_title(title)
            self.axs[idx].axis('equal')
            self.axs[idx].legend(labels=formatted_labels, fontsize=8, bbox_to_anchor=(1, 1))
            plt.subplots_adjust(left=0.0, right=0.75, top=1.0, bottom=0.0)

            offset = 1
            for child in children:
                if child in self.hierarchy:  # Only if the child has further children
                    node_queue.append((child, idx + offset))
                    offset += 1

        for i in range(len(self.axs)):
            if i not in plotted_indices:
                self.axs[i].axis('off')

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        self.restart_timers(running_timers)

    def get_color_variations(self, base_color, num_variations):
        if isinstance(base_color, str):
            base_rgb = to_rgb(base_color)
        else:
            base_rgb = base_color[:3]

        base_hls = colorsys.rgb_to_hls(*base_rgb)
        variations = []
        for i in range(num_variations):
            # Slightly vary the lightness and saturation
            lightness = base_hls[1] * (0.6 + 0.4 * (i / num_variations))
            saturation = base_hls[2] * (0.8 + 0.2 * (i / num_variations))
            # lightness = min(max(base_hls[1] * (0.9 + 0.1 * (i / num_variations)), 0), 1)
            # saturation = min(max(base_hls[2] * (0.9 + 0.1 * (i / num_variations)), 0), 1)
            variation = colorsys.hls_to_rgb(base_hls[0], lightness, saturation)
            variations.append(variation)
        return variations

    def format_time(self, time_value):
        units = [("s", 1), ("min", 60), ("h", 3600)]
        time_str = f"{time_value * 1000:.0f} ms"  # Default to milliseconds if time is very small
        for unit, limit in reversed(units):
            if time_value >= limit:
                time_str = f"{time_value / limit:.2f} {unit}"
                break
        return time_str

    def get_formatted_label(self,label, total_time, average_time):
        total_time_str = self.format_time(total_time)
        average_time_str = self.format_time(average_time)
        return f"{label}:\n({average_time_str}, {total_time_str})"

    def get_formatted_title(self, label, total_time, average_time):
        total_time_str = self.format_time(total_time)
        average_time_str = self.format_time(average_time)
        return f"{label} - Avg: {average_time_str}, Total: {total_time_str}"


if __name__ == "__main__":
    timer = Timer()
    timer.start('Overall Task')
    time.sleep(0.1)
    for i in range(5):
        timer.start('Setup', parent='Overall Task')
        timer.start('Load Resources', parent='Setup')
        load_resources()
        timer.stop('Load Resources')
        timer.start('Initial Processing', parent='Setup')
        data_processing()
        timer.stop('Initial Processing')
        timer.stop('Setup')

        timer.start('Main Computation', parent='Overall Task')
        timer.start('Complex Calculation', parent='Main Computation')
        complex_calculation()
        timer.stop('Complex Calculation')
        timer.start('Data Processing', parent='Main Computation')
        timer.start('Subtask 1', parent='Data Processing')
        time.sleep(0.1)
        data_processing()
        timer.stop('Data Processing')
        timer.stop('Main Computation')

        timer.start('Finalization', parent='Overall Task')
        time.sleep(0.5)
        timer.start('Save Results', parent='Finalization')
        time.sleep(0.01)
        timer.stop('Save Results')
        timer.start('Cleanup', parent='Finalization')
        time.sleep(0.01)
        timer.stop('Cleanup')
        time.sleep(0.05)
        timer.stop('Finalization')
        timer.plot_pie_charts()
    timer.stop('Overall Task')
    timer.plot_pie_charts()
    plt.show()
timer = Timer()