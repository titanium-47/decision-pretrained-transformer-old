"""
Fourier feature embedding for 2D (x, y) coordinates.
Embeds scalar inputs with sin/cos at multiple frequencies before passing to MLPs,
improving ability to represent high-frequency spatial structure.
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


class FourierFeatures2D(nn.Module):
    """
    Embed 2D (x, y) coordinates using sinusoidal Fourier features.
    For each dimension we use frequencies 2^0, 2^1, ..., 2^(num_freqs-1),
    outputting [sin(2^k * pi * x), cos(2^k * pi * x)] for k in 0..num_freqs-1.
    Output dim: 2 * 2 * num_freqs = 4 * num_freqs (for 2D input).
    """

    def __init__(self, num_freqs: int = 2, scale: float = 1.0):
        """
        Args:
            num_freqs: Number of frequency bands (2^0, ..., 2^(num_freqs-1)).
            scale: Scale input coords before embedding (e.g. 1/9 to normalize [0,9] grid to ~[0,1]).
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.scale = scale
        # 2D input, each dim gets 2*num_freqs (sin/cos per freq) -> 4*num_freqs total
        self.out_dim = 4 * num_freqs

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: (..., 2) tensor of x,y coordinates (any dtype; cast to float internally).

        Returns:
            features: (..., 4*num_freqs) tensor of Fourier features.
        """
        xy = xy.float() * self.scale
        freqs = 2.0 ** torch.arange(self.num_freqs, device=xy.device, dtype=xy.dtype)
        # (..., 2) @ (2, num_freqs) style: (..., 2, 1) * (num_freqs,) -> (..., 2, num_freqs)
        angles = xy.unsqueeze(-1) * freqs * np.pi  # pi
        sin_f = torch.sin(angles)
        cos_f = torch.cos(angles)
        # (..., 2, num_freqs) -> stack -> (..., 2, num_freqs, 2); flatten to (..., 4*num_freqs)
        out = torch.stack([sin_f, cos_f], dim=-1).flatten(start_dim=-3)
        return out


def plot_fourier_features_goal_grid(
    output_path: str,
    grid_size: int = 10,
    num_freqs: int = 2,
    scale: float = 1.0 / 9.0,
):
    """
    Plot each Fourier feature dimension over a 2D goal grid.

    With num_freqs=2, output dim is 8 and this produces 8 subplots.
    """
    embedder = FourierFeatures2D(num_freqs=num_freqs, scale=scale)
    out_dim = embedder.out_dim

    goals = torch.tensor(
        [[x, y] for y in range(grid_size) for x in range(grid_size)],
        dtype=torch.float32,
    )
    with torch.no_grad():
        features = embedder(goals).cpu().numpy()  # (grid_size^2, out_dim)

    cols = 4
    rows = int(np.ceil(out_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    vmin = float(np.min(features))
    vmax = float(np.max(features))

    for dim_idx in range(rows * cols):
        ax = axes[dim_idx // cols, dim_idx % cols]
        if dim_idx < out_dim:
            feature_grid = features[:, dim_idx].reshape(grid_size, grid_size)
            image = ax.imshow(
                feature_grid,
                origin="lower",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"Feature dim {dim_idx}")
            ax.set_xlabel("Goal x")
            ax.set_ylabel("Goal y")
            ax.set_xticks(np.arange(grid_size))
            ax.set_yticks(np.arange(grid_size))
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    fig.suptitle(
        f"FourierFeatures2D on {grid_size}x{grid_size} goal grid "
        f"(num_freqs={num_freqs}, out_dim={out_dim}, scale={scale})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fourier feature grid plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot FourierFeatures2D dimensions over a goal-position grid."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="fourier_goal_features_10x10.png",
        help="Path to save the output plot.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Grid side length for goal positions.",
    )
    parser.add_argument(
        "--num_freqs",
        type=int,
        default=2,
        help="Number of Fourier frequency bands. num_freqs=2 gives 8 dims.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0 / 9.0,
        help="Coordinate scale before Fourier embedding.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    plot_fourier_features_goal_grid(
        output_path=args.output_path,
        grid_size=args.grid_size,
        num_freqs=args.num_freqs,
        scale=args.scale,
    )
