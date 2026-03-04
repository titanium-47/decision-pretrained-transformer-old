import torch
import torch.nn as nn
from typing import abstractmethod
from fourier import FourierFeatures2D

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, x, eps=None):
        raise NotImplementedError

    @abstractmethod
    def compute_encoder_loss(self, x):
        raise NotImplementedError

class GoalInformationBottleneckEncoder(Encoder):
    """
    Encoder with information bottleneck: x -> z
    Regularizes z to follow a standard Gaussian prior N(0, I)
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=128, num_layers=3):
        super().__init__(input_dim, latent_dim)
        
        layers = []
        layers.append(nn.Linear(input_dim*8, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.fourier = FourierFeatures2D(num_freqs=4, scale=1.0/9.0)
        self.shared = nn.Sequential(*layers)
        
        # Output mean and log-variance for latent z
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, eps=None):
        """
        Args:
            x: Input features, shape (batch_size, input_dim)
            use_mean: If True, skip sampling and return mean as z
        
        Returns:
            z: Latent representation, shape (batch_size, latent_dim)
            mean: Mean of latent distribution, shape (batch_size, latent_dim)
            logvar: Log-variance of latent distribution, shape (batch_size, latent_dim)
        """
        B, T, _ = x.shape
        x = x.reshape(-1, self.input_dim)
        x = self.fourier(x)
        x = x.reshape(B, T, -1)
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z
    
    def compute_encoder_loss(self, x):
        """
        Compute KL divergence KL(q(z|x) || N(0, I))
        
        Args:
            x: Input features, shape (batch_size, input_dim)
        
        Returns:
            kl_loss: Scalar KL divergence loss
        """
        B, T, _ = x.shape
        x = x.reshape(-1, self.input_dim)
        x = self.fourier(x)
        x = x.reshape(B, T, -1)
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        # KL(q(z|x) || N(0, I)) = 0.5 * sum(exp(logvar) + mean^2 - 1 - logvar)
        # For each dimension: KL_i = 0.5 * [exp(logvar_i) + mean_i^2 - 1 - logvar_i]
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

class GoalDeterministicEncoder(Encoder):
    """
    Encoder with information bottleneck: x -> z
    Regularizes z to follow a standard Gaussian prior N(0, I)
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=128, num_layers=3):
        super().__init__(input_dim, latent_dim)
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        self.fourier = FourierFeatures2D(num_freqs=4, scale=1.0/9.0)
            
    def forward(self, x, eps=None):
        x = x.reshape(-1, self.input_dim)
        x = self.fourier(x)
        return self.encoder(x)
    
    def compute_encoder_loss(self, x):
        return torch.tensor(0.0, device=x.device)

class NullEncoder(Encoder):
    def __init__(self, input_dim, latent_dim):
        super().__init__(input_dim, latent_dim)
    
    def forward(self, x, eps=None):
        return torch.zeros(x.shape[0], self.latent_dim, device=x.device)
    
    def compute_encoder_loss(self, x):
        return torch.tensor(0.0, device=x.device)

class DiffusionForwardNoiseEncoder(Encoder):
    def __init__(self, input_dim, latent_dim, alpha=0.8):
        super().__init__(input_dim, latent_dim)
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, latent_dim)
        self.bn = nn.BatchNorm1d(input_dim, affine=False)

    def forward(self, x, eps=None):
        B, T, _ = x.shape
        x = x.reshape(-1, self.input_dim)
        x = self.bn(x)
        x = x.reshape(B, T, -1)
        return self._diffusion_forward_noise(self.embedding(x))

    def _diffusion_forward_noise(self, x):
        if self.training:
            return self.alpha * x + (1 - self.alpha) * torch.randn_like(x, device=x.device)
        else:
            return self.alpha * x

    def compute_encoder_loss(self, x):
        return torch.tensor(0.0, device=x.device)

class NoOpEncoder(Encoder):
    def __init__(self, input_dim, latent_dim):
        super().__init__(input_dim, latent_dim)
    
    def forward(self, x, eps=None):
        return x
    
    def compute_encoder_loss(self, x):
        return torch.tensor(0.0, device=x.device)

class BatchNormEncoder(Encoder):
    def __init__(self, input_dim, latent_dim):
        super().__init__(input_dim, latent_dim)
        self.bn = nn.BatchNorm1d(input_dim, affine=False)

    def forward(self, x, eps=None):
        B, T, _ = x.shape
        x = x.reshape(-1, self.input_dim)
        x = self.bn(x)
        x = x.reshape(B, T, -1)
        return x
    
    def compute_encoder_loss(self, x):
        return torch.tensor(0.0, device=x.device)
