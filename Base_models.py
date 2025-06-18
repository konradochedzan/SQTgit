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
        Elastic Net regularisation loss function.
        This loss function combines L1 and L2 regularisation, controlled by the
        parameters `alpha` (overall strength) and `l1_ratio` (mixing factor).
        Parameters
        ----------
        model       : the network whose parameters are regularised
        alpha       : overall strength of the regularisation term
        l1_ratio    : mixing factor (0 = L2, 1 = L1, 0.5 = Elastic Net)
        base_loss   : any point-wise loss (defaults to nn.MSELoss())
        reg_bias    : if False, skips parameters with 1 dimension (bias vectors)
        """
        def __init__(self, model,
                    alpha: float = 1e-3,
                    l1_ratio: float = 0.5,
                    base_loss: nn.Module = nn.MSELoss(),
                    reg_bias: bool = False):
            super().__init__()
            self.model = model
            self.alpha = float(alpha)
            self.l1_ratio = float(l1_ratio)
            self.base_loss = base_loss
            self.reg_bias = reg_bias

        def forward(self, y_pred, y_true):
            # data-fitting term
            loss = self.base_loss(y_pred, y_true)

            # accumulate parameter norms
            l1_norm, l2_norm = 0.0, 0.0
            for p in self.model.parameters():
                if not self.reg_bias and p.ndim == 1:      # skip bias by default
                    continue
                l1_norm += p.abs().sum()
                l2_norm += p.pow(2).sum()

            # combine
            loss += self.alpha * (
                self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm
            )
            return loss