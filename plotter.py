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

    def __post_init__(self):
        if self.line_colors is None:
            self.line_colors = {
                "rewards": "blue",
                "q_values": "green",
                "losses": "orange",
            }
        if self.avg_colors is None:
            self.avg_colors = {
                "rewards": "red",
                "q_values": "darkgreen",
                "losses": "darkorange",
            }


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
    Plot multiple data series with optional running averages.

    Args:
        x: X-axis data
        data: Dictionary containing data series to plot (e.g., {"rewards": [...], "q_values": [...], "losses": [...]})
        config: Plot configuration settings
    """
    if config is None:
        config = PlotConfig()

    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype=np.float32)
    data = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}

    # Setup plot
    fig, ax = setup_plot(config)

    # Plot each data series
    for series_name, y in data.items():
        # Plot main data
        sns.lineplot(
            x=x,
            y=y,
            ax=ax,
            label=series_name.replace("_", " ").title(),
            color=config.line_colors.get(series_name, "blue"),
        )

        # Plot running average if enabled
        if config.running_avg:
            running_avg = calculate_running_average(y, config.window_size)
            sns.lineplot(
                x=x,
                y=running_avg,
                ax=ax,
                label=f"Average {series_name.replace('_', ' ').title()}",
                color=config.avg_colors.get(series_name, "red"),
            )

    # Ensure the directory exists
    save_path = Path(config.filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Finalize and save plot
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)  # Clean up resources
