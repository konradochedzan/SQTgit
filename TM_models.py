import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for time series prediction.
    Uses dilated causal convolutions to capture long-range dependencies.
    """
    def __init__(self, input_dim, num_channels=[64, 128, 64], kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation, padding=(kernel_size-1) * dilation,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        # TCN expects (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)
        # Take the last time step
        x = x[:, :, -1]
        return self.output_layer(x)

class TemporalBlock(nn.Module):
    """Individual temporal block for TCN"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.group_norm1 = nn.GroupNorm(1, n_outputs)  # GroupNorm for better stability
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.group_norm2 = nn.GroupNorm(1, n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.group_norm1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.group_norm2, self.dropout2)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """Remove padding from the end to maintain causality"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer implementation.
    Focuses on key components: attention mechanisms and variable selection.
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_layers=2, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Variable selection network
        self.variable_selection = VariableSelectionNetwork(input_dim, hidden_dim, dropout)
        
        # Static enrichment (simplified)
        self.static_enrichment = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        
        # Multi-head attention for temporal fusion
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Position-wise feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Variable selection
        selected_features = self.variable_selection(x)
        
        # Static enrichment (simplified - using mean as static context)
        static_context = torch.mean(selected_features, dim=1, keepdim=True)
        static_enriched = self.static_enrichment(static_context)
        
        # Add static context to all time steps
        enriched_features = selected_features + static_enriched.expand(-1, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(enriched_features)
        
        # Self-attention
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm1(attn_out + lstm_out)
        
        # Feed forward
        ff_out = self.feed_forward(attn_out)
        ff_out = self.norm2(ff_out + attn_out)
        
        # Output (use last time step)
        output = self.output_layer(ff_out[:, -1, :])
        
        return output

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        
        # Flatten and project
        self.flatten_projection = nn.Linear(input_dim, hidden_dim)
        
        # Variable selection weights
        self.variable_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Selected variable processing
        self.selected_processing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Flatten for weight computation
        x_flat = x.view(-1, input_dim)
        
        # Compute variable selection weights
        projected = self.flatten_projection(x_flat)
        weights = self.variable_weights(projected)
        weights = weights.view(batch_size, seq_len, input_dim)
        
        # Apply weights to original features
        selected = x * weights
        
        # Process selected variables
        selected_flat = selected.view(-1, input_dim)
        processed = self.selected_processing(selected_flat)
        processed = processed.view(batch_size, seq_len, -1)
        
        return processed