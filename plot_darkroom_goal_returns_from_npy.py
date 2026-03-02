"""
Load darkroom_goal_returns.npy and plot a heatmap with fixed color scaling [0, 100].
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot a darkroom goal return heatmap from a .npy file"
    )
    parser.add_argument(
        "--input_npy",
        type=str,
        required=True,
        help="Path to darkroom_goal_returns.npy (expected shape: 10x10).",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="Path to output image. Defaults to <input_dir>/darkroom_goal_returns_heatmap_norm_0_100.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Darkroom Goal Returns (Color Scale 0-100)",
        help="Plot title.",
    )
    parser.add_argument(
        "--annotate_cells",
        action="store_true",
        help="If set, write values in each heatmap cell.",
    )
    args = parser.parse_args()

    heatmap = np.load(args.input_npy)
    print(f"Avg Return: {np.mean(heatmap)}")
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {heatmap.shape}")

    if args.output_png is None:
        input_dir = os.path.dirname(args.input_npy) or "."
        output_png = os.path.join(
            input_dir, "darkroom_goal_returns_heatmap_norm_0_100.png"
        )
    else:
        output_png = args.output_png

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(heatmap, cmap="viridis", origin="lower", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Return (normalized color scale: 0 to 100)")

    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_yticks(np.arange(heatmap.shape[0]))
    ax.set_xlabel("Goal x")
    ax.set_ylabel("Goal y")
    ax.set_title(args.title)

    if args.annotate_cells:
        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                ax.text(
                    x,
                    y,
                    f"{heatmap[y, x]:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6,
                )

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded heatmap shape: {heatmap.shape}")
    print(f"Saved heatmap image: {output_png}")
    print("Color scale fixed to [0, 100].")


if __name__ == "__main__":
    main()
