"""
Fourier feature embedding for 2D (x, y) coordinates.
Embeds scalar inputs with sin/cos at multiple frequencies before passing to MLPs,
improving ability to represent high-frequency spatial structure.
"""
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

    def __init__(self, num_freqs: int = 4, scale: float = 1.0):
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
