from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class PlotConfig:
    """Configuration class for plot settings."""

    title: str = "Plot"
    subtitle: str = ""
    xlabel: str = "X"
    ylabel: str = "Y"
    figsize: Tuple[int, int] = (10, 6)
    style: str = "darkgrid"
    running_avg: bool = True
    window_size: int = 100
    filepath: str = "plots/plot.png"
    line_colors: Dict[str, str] = None
    avg_colors: Dict[str, str] = None
    # Dictionary mapping metric names to y-axis index (0 for primary, 1 for secondary, 2 for tertiary)
    axis_mapping: Dict[str, int] = None

    def __post_init__(self):
        if self.line_colors is None:
            self.line_colors = {}
        if self.avg_colors is None:
            self.avg_colors = {}
        if self.axis_mapping is None:
            self.axis_mapping = {}


def setup_plot(config: PlotConfig) -> Tuple[plt.Figure, plt.Axes]:
    """Set up the plot with basic styling and configuration."""
    sns.set_style(config.style)
    fig, ax = plt.subplots(figsize=config.figsize)

    if config.subtitle:
        fig.suptitle(config.subtitle, fontsize=16)

    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)

    return fig, ax


def calculate_running_average(y: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate the running average of the data."""
    running_avg = np.zeros_like(y)
    for i in range(len(y)):
        window_start = max(0, i - window_size + 1)
        running_avg[i] = np.mean(y[window_start : i + 1])
    return running_avg


def plot_data(
    x: np.ndarray,
    data: Dict[str, Union[np.ndarray, List[float]]],
    config: Optional[PlotConfig] = None,
) -> None:
    """
    Plot multiple data series with running averages and multiple y-axes for different scales.

    Args:
        x: X-axis data
        data: Dictionary containing data series to plot
        config: Plot configuration settings
    """
    if config is None:
        config = PlotConfig()

    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype=np.float32)
    data = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}

    # Setup plot
    fig, ax = setup_plot(config)

    # If axis_mapping is not provided, create a default mapping
    if not config.axis_mapping:
        config.axis_mapping = {name: idx for idx, name in enumerate(data.keys())}

    # If colors are not provided, generate them
    if not config.avg_colors:
        # Use a color map to generate distinct colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
        config.avg_colors = {name: colors[i] for i, name in enumerate(data.keys())}

    # Determine which axes are actually needed based on the data being plotted
    needed_axes = set(config.axis_mapping[series] for series in data.keys())
    max_axis_idx = max(needed_axes)

    # Create all necessary axes
    axes = [ax]  # Primary axis
    if max_axis_idx >= 1:
        ax2 = ax.twinx()  # Secondary axis
        axes.append(ax2)
    if max_axis_idx >= 2:
        # Create tertiary axis by offsetting the secondary axis
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        axes.append(ax3)

    # Store lines for legend
    lines = []
    labels = []

    # Sort data series to ensure consistent plotting order
    sorted_data = dict(sorted(data.items(), key=lambda x: config.axis_mapping[x[0]]))

    # Plot each data series
    for series_name, y in sorted_data.items():
        axis_idx = config.axis_mapping.get(series_name, 0)
        current_ax = axes[axis_idx]

        # Plot running average
        running_avg = calculate_running_average(y, config.window_size)
        avg_line = current_ax.plot(
            x,
            running_avg,
            label=f"{series_name.replace('_', ' ').title()}",
            color=config.avg_colors.get(series_name),
        )[0]
        lines.append(avg_line)
        labels.append(f"{series_name.replace('_', ' ').title()}")

        # Set axis label and color based on the series
        current_ax.set_ylabel(series_name.replace("_", " ").title())
        if axis_idx == 0:
            current_ax.spines["left"].set_color(config.avg_colors[series_name])
            current_ax.yaxis.label.set_color(config.avg_colors[series_name])
            current_ax.tick_params(axis="y", colors=config.avg_colors[series_name])
        else:
            current_ax.spines["right"].set_color(config.avg_colors[series_name])
            current_ax.yaxis.label.set_color(config.avg_colors[series_name])
            current_ax.tick_params(axis="y", colors=config.avg_colors[series_name])

    # Add legend
    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1.35, 0.5))

    # Ensure the directory exists
    save_path = Path(config.filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Finalize and save plot
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)  # Clean up resources
