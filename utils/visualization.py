"""
Visualization helpers for benchmark results.
"""

from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_latency_comparison(
    names: list[str],
    mean_times_ms: list[float],
    std_times_ms: Optional[list[float]] = None,
    title: str = "Latency Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing latencies across configurations."""
    if not HAS_MPL:
        print("matplotlib not installed, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, mean_times_ms, yerr=std_times_ms, capsize=4,
                  color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_memory_comparison(
    names: list[str],
    memory_mb: list[float],
    title: str = "Peak GPU Memory",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing peak memory usage."""
    if not HAS_MPL:
        print("matplotlib not installed, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, memory_mb, color="coral", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Memory (MB)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_scaling(
    x_values: list,
    y_values_dict: dict[str, list[float]],
    xlabel: str = "Sequence Length",
    ylabel: str = "Latency (ms)",
    title: str = "Scaling Behavior",
    logx: bool = False,
    logy: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Line plot showing how a metric scales with a parameter."""
    if not HAS_MPL:
        print("matplotlib not installed, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, y_values in y_values_dict.items():
        ax.plot(x_values[:len(y_values)], y_values, marker="o", label=label)

    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
