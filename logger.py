import numpy as np
import torch

from plotter import PlotConfig, plot_data


class Logger:
    def __init__(self, dict: dict = None):
        self.dict = dict if dict else {}

    def log(self, key, time, value):
        if key not in self.dict:
            self.dict[key] = []
        self.dict[key].append((time, value))

    def get_data(self, key):
        return self.dict[key]

    def clear(self):
        self.dict = {}

    def save_plot(self, path: str, keys: list[str] = None):
        # Prepare data for plotting
        data = {}
        times = None

        # Determine which keys to plot
        plot_keys = keys if keys is not None else self.dict.keys()

        # Extract data from the logger
        for key in plot_keys:
            if key not in self.dict:
                print(f"Warning: Key '{key}' not found in logger")
                continue
            if not self.dict[key]:  # Skip empty series
                continue

            # Unzip the time-value pairs
            times_series, values = zip(*self.dict[key])

            # Convert values to numpy array, ensuring tensors are moved to CPU first and detached from computation graph
            values = [
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for v in values
            ]
            data[key] = np.array(values)

            # Use the first available time series as our x-axis
            times = np.array(times_series)

        if not data or times is None:
            return  # Nothing to plot

        # Create plot configuration
        config = PlotConfig(
            title="Training Progress",
            xlabel="Time Steps",
            ylabel="Values",
            filepath=path,
            running_avg=True,
            window_size=100,
        )

        # Generate the plot
        plot_data(times, data, config)
