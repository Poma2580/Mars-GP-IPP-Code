"""Draw the two paper-style magnetic field figures from saved GPy models."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from GPy_true_field import DEFAULT_OUTPUT, DEFAULT_WORKBOOK, ROVERS, load_training_data


TITLES = ("Br", "Bφ", "Bψ")
ZHURONG_LEVELS = (
    np.arange(-36, 19, 6),
    np.arange(-24, 25, 6),
    np.arange(-30, 11, 5),
)
PERSEVERANCE_LEVELS = (
    np.arange(-60, 181, 30),
    np.arange(-60, 46, 15),
    np.arange(-15, 106, 15),
)


def grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(0.0, 31.0, 1.0)
    x, y = np.meshgrid(axis, axis)
    locations = np.column_stack((x.ravel(), y.ravel()))
    return x, y, locations


def plot_rover(rover: str, model, X: np.ndarray, output_path: Path) -> None:
    x_local, y_local, locations = grid()
    mean, _ = model.predict(locations)
    if mean.shape != (locations.shape[0], 3):
        raise ValueError(f"{rover} 模型预测形状不正确: {mean.shape}")

    if rover == "zhurong":
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        x_plot, y_plot = x_local, y_local
        points = X
        levels = ZHURONG_LEVELS
        x_label, y_label = "Longitude offset (°)", "Latitude offset (°)"
        ticks = np.arange(0, 31, 5)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(5, 12))
        offset = np.asarray(ROVERS[rover]["origin"])
        x_plot, y_plot = x_local + offset[0], y_local + offset[1]
        points = X + offset
        levels = PERSEVERANCE_LEVELS
        x_label, y_label = "Longitude (°)", "Latitude (°)"
        ticks = np.arange(65, 96, 5)

    for index, ax in enumerate(axes):
        surface = ax.contourf(
            x_plot,
            y_plot,
            mean[:, index].reshape(x_plot.shape),
            levels=levels[index],
            cmap="hot",
            extend="both",
        )
        if rover == "zhurong":
            ax.scatter(
                points[:, 0], points[:, 1],
                color="green", marker="*", s=22, linewidths=0.4,
                label="observation position" if index == 0 else None,
                zorder=3,
            )
        ax.set_title(TITLES[index] if rover == "zhurong" else f"{TITLES[index]}(GPR)")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(ticks)
        ax.set_yticks(np.arange(0 if rover == "zhurong" else 5, 31 if rover == "zhurong" else 36, 5))
        ax.set_xlim(float(x_plot.min()), float(x_plot.max()))
        ax.set_ylim(float(y_plot.min()), float(y_plot.max()))
        ax.grid(color="gray", linestyle="--", linewidth=1.2, alpha=0.75)
        fig.colorbar(surface, ax=ax, ticks=levels[index], fraction=0.048, pad=0.04, extendrect=True)

    if rover == "zhurong":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=True)
        fig.subplots_adjust(left=0.05, right=0.97, top=0.89, bottom=0.13, wspace=0.38)
    else:
        fig.subplots_adjust(left=0.18, right=0.9, top=0.97, bottom=0.05, hspace=0.48)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"{rover}: 图片保存到 {output_path}")


def draw_all(workbook_path: Path = DEFAULT_WORKBOOK, output_dir: Path = DEFAULT_OUTPUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for rover in ROVERS:
        with (output_dir / f"{rover}.pkl").open("rb") as source:
            model = pickle.load(source)
        X, _ = load_training_data(workbook_path, rover)
        plot_rover(rover, model, X, output_dir / f"{rover}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    draw_all(args.workbook, args.output_dir)


if __name__ == "__main__":
    main()
