from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    line_color: str = "blue"
    avg_color: str = "red"


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
    x: np.ndarray, y: np.ndarray, config: Optional[PlotConfig] = None
) -> None:
    """
    Plot data with optional running average.

    Args:
        x: X-axis data
        y: Y-axis data
        config: Plot configuration settings
    """
    if config is None:
        config = PlotConfig()

    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Setup plot
    fig, ax = setup_plot(config)

    # Plot main data
    sns.lineplot(
        x=x,
        y=y,
        ax=ax,
        label=config.ylabel,
        color=config.line_color,
    )

    # Plot running average if enabled
    if config.running_avg:
        running_avg = calculate_running_average(y, config.window_size)
        sns.lineplot(
            x=x,
            y=running_avg,
            ax=ax,
            label="Average",
            color=config.avg_color,
        )

    # Ensure the directory exists
    save_path = Path(config.filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Finalize and save plot
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)  # Clean up resources
