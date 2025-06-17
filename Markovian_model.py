import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, LeakyReLU, Sigmoid, Tanh
import torch.nn as nn
from tqdm import tqdm

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
    def __init__(self, input_size, hidden_layer_size, compressed_size):
        super(AutoEncoder, self).__init__()
        self.encoder()
