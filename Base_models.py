import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, LeakyReLU, Sigmoid, Tanh
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class FeedForwardPredictor(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size = 1):
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU(),
            nn.BatchNorm2d(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.BatchNorm2d(),
            nn.Linear(hidden_layer_size, output_size)
        )
        
    def forward(self,x):
        return(self.network(x))
    
class AutoEncoder(nn.Module):
    """Simple autoencoder for dimensionality reduction"""
    def __init__(self, input_dim: int, encoding_dim: int):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
class ElasticNetLoss(nn.Module):
    """
    Adds an Elastic-Net penalty (α · ( l1_ratio · ‖θ‖₁ + (1-l1_ratio) · ‖θ‖₂² )) to any base loss.
    """
    def __init__(self, model, alpha: float = 1e-3, l1_ratio: float = 0.5,
                 base_loss: nn.Module = nn.MSELoss(), reg_bias: bool = False):
        super().__init__()
        self.model, self.alpha, self.l1_ratio = model, alpha, l1_ratio
        self.base_loss, self.reg_bias = base_loss, reg_bias

    def _flat_params(self):
        for p in self.model.parameters():
            if self.reg_bias or p.ndim != 1:
                yield p.view(-1)

    def forward(self, y_hat, y):
        loss = self.base_loss(y_hat, y)
        if self.alpha == 0:
            return loss                      # shortcut

        vec = torch.cat(list(self._flat_params()))
        l1 = vec.abs().sum()
        l2 = vec.pow(2).sum()               # squared ℓ₂, same as weight-decay
        return loss + self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)