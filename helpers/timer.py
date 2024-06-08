import time
import numpy as np
import matplotlib.pyplot as plt


class Timer:
    def __init__(self):
        self.start_times = {}
        self.timings = {}
        self.counts = {}

    def start(self, label):
        self.start_times[label] = time.time()

    def stop(self, label):
        if label not in self.start_times:
            print(f"Timer for '{label}' was not started.")
            return
        end_time = time.time()
        elapsed = end_time - self.start_times[label]
        if label in self.timings:
            self.timings[label] += elapsed
            self.counts[label] += 1
        else:
            self.timings[label] = elapsed
            self.counts[label] = 1
        del self.start_times[label]  # Clean up the start time

    def calculate_averages(self):
        return {label: self.timings[label] / self.counts[label] for label in self.timings}

    def get_formatted_label(self, label, total_time, average_time):
        units = [("s", 1), ("min", 60), ("h", 3600)]
        total_time_str = f"{total_time * 1000:.0f} ms"  # Default to milliseconds if time is very small
        average_time_str = f"{average_time * 1000:.0f} ms"  # Default to milliseconds if time is very small
        for unit, limit in reversed(units):
            if total_time >= limit:
                total_time_str = f"{total_time / limit:.2f} {unit}"
                average_time_str = f"{average_time / limit:.2f} {unit}"
                break
        return f"{label}\n(Total: {total_time_str}, Avg: {average_time_str})"

    def plot_pie_chart(self):
        if not self.timings:
            print("No timings available to plot.")
            return

        averages = self.calculate_averages()
        labels = [self.get_formatted_label(label, self.timings[label], averages[label]) for label in self.timings]
        times = list(self.timings.values())

        # Define the color palette
        colors = plt.cm.inferno(np.linspace(0, 1, len(labels)))

        fig, ax = plt.subplots(figsize=(12, 7))
        wedges, texts, autotexts = ax.pie(times, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True, explode=[0.00] * len(labels))
        # Position the pie chart on the left side of the ax



        plt.title('Time Distribution among Tasks (Total and Average)')
        plt.legend(wedges, labels, title="Tasks", loc="center left", bbox_to_anchor=(0.9, 0.5, 0.5, 1))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

timer = Timer()