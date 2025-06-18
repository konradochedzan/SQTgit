import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TemporalMixtureGate(nn.Module):
    """
    Temporal mixture model for gating volatility and interest rate components
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, seq_len: int = 30):
        super().__init__()
        self.seq_len = seq_len
        
        # Attention mechanism for temporal dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True
        )
        
        # Gate networks
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # 2 gates: volatility and interest rate
            nn.Softmax(dim=-1)
        )
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, input_dim) * 0.1) 

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            gates: (batch_size, seq_len, 2) - [vol_gate, interest_gate]
        """
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Apply attention
        attended_x, _ = self.attention(x, x, x)
        
        # Compute gates
        gates = self.gate_network(attended_x)
        
        return gates

class VolatilityPredictor(nn.Module):
    """
    Complex architecture for volatility prediction using LSTM + additional features
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # GRU for additional temporal processing
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Autoregressive component
        self.ar_weights = nn.Parameter(torch.randn(5) * 0.1)  # AR(5) model
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 1, hidden_dim // 4),  # +1 for AR component
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Mean and log_std for volatility distribution
        )
        
    def forward(self, x: torch.Tensor, vol_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - volatility signals
            vol_history: (batch_size, seq_len) - historical volatility for AR
        Returns:
            Dict with volatility predictions and uncertainty
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # GRU processing
        gru_out, _ = self.gru(lstm_out)
        
        # Autoregressive component (using last 5 observations)
        if seq_len >= 5:
            ar_component = torch.sum(
                self.ar_weights.unsqueeze(0).unsqueeze(0) * vol_history[:, -5:], dim=-1
            ).unsqueeze(-1)
        else:
            ar_component = torch.zeros(batch_size, 1, device=x.device)
        
        # Combine features
        combined = torch.cat([gru_out[:, -1, :], ar_component], dim=-1)
        
        # Output prediction
        vol_params = self.output_layers(combined)
        vol_mean = torch.exp(vol_params[:, 0])  # Ensure positive volatility
        vol_log_std = vol_params[:, 1]
        
        return {
            'vol_mean': vol_mean,
            'vol_log_std': vol_log_std,
            'vol_std': torch.exp(vol_log_std)
        }

class InterestRatePredictor(nn.Module):
    """
    Simple FFNN for interest rate prediction
    """
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) - interest rate features
        Returns:
            interest_rate: (batch_size, 1)
        """
        return self.network(x)

class RiskAdjustedReturnCalculator(nn.Module):
    """
    Calculate risk-adjusted returns using stochastic volatility model
    """
    def __init__(self):
        super().__init__()
        # Learnable parameters for the stochastic process
        self.risk_premium = nn.Parameter(torch.tensor(0.05))
        self.vol_scaling = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, volatility: torch.Tensor, interest_rate: torch.Tensor, 
                dt: float = 1/252) -> torch.Tensor:
        """
        Calculate risk-adjusted return using stochastic volatility formula
        Args:
            volatility: (batch_size,) - predicted volatility
            interest_rate: (batch_size,) - predicted interest rate
            dt: time step (default: daily = 1/252)
        Returns:
            risk_adjusted_return: (batch_size,)
        """
        # Stochastic volatility model: dS/S = (r + λσ)dt + σdW
        # Risk-adjusted return = r + risk_premium * volatility
        risk_adjusted_return = (
            interest_rate.squeeze() + 
            self.risk_premium * self.vol_scaling * volatility
        ) * dt
        
        return risk_adjusted_return

class RealReturnTransformer(nn.Module):
    """
    Transform risk-adjusted returns to real returns using macro signals
    """
    def __init__(self, macro_features_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.feature_processor = nn.Sequential(
            nn.Linear(macro_features_dim + 1, hidden_dim),  # +1 for risk-adjusted return
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.8))
        
        # Final transformation
        self.output_layer = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, risk_adjusted_return: torch.Tensor, 
                macro_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            risk_adjusted_return: (batch_size,) - risk-adjusted predictions
            macro_features: (batch_size, macro_features_dim) - unemployment, inflation, etc.
        Returns:
            real_return: (batch_size,) - final return prediction
        """
        # Combine inputs
        combined_input = torch.cat([
            risk_adjusted_return.unsqueeze(-1), 
            macro_features
        ], dim=-1)
        
        # Process features
        processed = self.feature_processor(combined_input)
        adjustment = self.output_layer(processed).squeeze()
        
        # Apply residual connection
        real_return = (
            self.residual_weight * risk_adjusted_return + 
            (1 - self.residual_weight) * adjustment
        )
        
        return real_return

class StochasticVolatilityModel(nn.Module):
    """
    Main model combining all components
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract dimensions from config
        vol_signals_dim = config['vol_signals_dim']
        interest_signals_dim = config['interest_signals_dim']
        macro_features_dim = config['macro_features_dim']
        seq_len = config['seq_len']
        
        # Initialize components
        self.temporal_gate = TemporalMixtureGate(
            input_dim=vol_signals_dim + interest_signals_dim,
            seq_len=seq_len
        )
        
        self.volatility_predictor = VolatilityPredictor(
            input_dim=vol_signals_dim
        )
        
        self.interest_predictor = InterestRatePredictor(
            input_dim=interest_signals_dim
        )
        
        self.risk_calculator = RiskAdjustedReturnCalculator()
        
        self.return_transformer = RealReturnTransformer(
            macro_features_dim=macro_features_dim
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire model
        """
        vol_signals = batch['vol_signals']  # (batch_size, seq_len, vol_dim)
        interest_signals = batch['interest_signals']  # (batch_size, seq_len, int_dim)
        vol_history = batch['vol_history']  # (batch_size, seq_len)
        macro_features = batch['macro_features']  # (batch_size, macro_dim)
        
        batch_size, seq_len, _ = vol_signals.shape
        
        # Combine signals for gating
        combined_signals = torch.cat([vol_signals, interest_signals], dim=-1)
        
        # Compute temporal gates
        gates = self.temporal_gate(combined_signals)  # (batch_size, seq_len, 2)
        
        # Get predictions from individual components
        vol_pred = self.volatility_predictor(vol_signals, vol_history)
        interest_pred = self.interest_predictor(interest_signals[:, -1, :])  # Use last timestep
        
        # Apply temporal weighting (use last timestep gates)
        vol_weight = gates[:, -1, 0]  # (batch_size,)
        int_weight = gates[:, -1, 1]  # (batch_size,)
        
        # Weighted volatility and interest rate
        weighted_vol = vol_weight * vol_pred['vol_mean']
        weighted_interest = int_weight * interest_pred.squeeze()
        
        # Calculate risk-adjusted return
        risk_adjusted_return = self.risk_calculator(weighted_vol, weighted_interest)
        
        # Transform to real return
        real_return = self.return_transformer(risk_adjusted_return, macro_features)
        
        return {
            'real_return': real_return,
            'risk_adjusted_return': risk_adjusted_return,
            'volatility': vol_pred,
            'interest_rate': interest_pred,
            'gates': gates,
            'vol_weight': vol_weight,
            'int_weight': int_weight
        }

class FinancialLoss(nn.Module):
    """
    Custom loss function combining multiple financial objectives
    """
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {
            'return_mse': 1.0,
            'vol_nll': 0.5,
            'interest_mse': 0.3,
            'sharpe_penalty': 0.2
        }
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component loss
        """
        losses = {}
        
        # Main return prediction loss
        return_loss = F.mse_loss(predictions['real_return'], targets['actual_returns'])
        losses['return_mse'] = return_loss
        
        # Volatility negative log-likelihood (if available)
        if 'actual_volatility' in targets:
            vol_mean = predictions['volatility']['vol_mean']
            vol_std = predictions['volatility']['vol_std']
            vol_nll = -torch.distributions.Normal(vol_mean, vol_std).log_prob(
                targets['actual_volatility']
            ).mean()
            losses['vol_nll'] = vol_nll
        
        # Interest rate loss (if available)
        if 'actual_interest_rates' in targets:
            int_loss = F.mse_loss(predictions['interest_rate'], targets['actual_interest_rates'])
            losses['interest_mse'] = int_loss
        
        # Sharpe ratio penalty (encourage consistent returns)
        if len(predictions['real_return']) > 1:
            pred_returns = predictions['real_return']
            sharpe_penalty = -torch.mean(pred_returns) / (torch.std(pred_returns) + 1e-8)
            losses['sharpe_penalty'] = sharpe_penalty
        
        # Combined loss
        total_loss = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        losses['total'] = total_loss
        
        return losses

# Training utilities
class FinancialDataProcessor:
    """
    Utility class for processing financial data
    """
    def __init__(self):
        self.scalers = {}
    
    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fit scalers and transform data
        """
        processed_data = {}
        
        for key, values in data.items():
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            
            scaler = StandardScaler()
            processed_data[key] = scaler.fit_transform(values)
            self.scalers[key] = scaler
            
            if processed_data[key].shape[1] == 1:
                processed_data[key] = processed_data[key].flatten()
        
        return processed_data
    
    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform data using fitted scalers
        """
        processed_data = {}
        
        for key, values in data.items():
            if key in self.scalers:
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                
                processed_data[key] = self.scalers[key].transform(values)
                
                if processed_data[key].shape[1] == 1:
                    processed_data[key] = processed_data[key].flatten()
            else:
                processed_data[key] = values
        
        return processed_data

def create_model_config(vol_signals_dim: int = 10, 
                       interest_signals_dim: int = 5,
                       macro_features_dim: int = 8,
                       seq_len: int = 30) -> Dict:
    """
    Create model configuration
    """
    return {
        'vol_signals_dim': vol_signals_dim,
        'interest_signals_dim': interest_signals_dim,
        'macro_features_dim': macro_features_dim,
        'seq_len': seq_len
    }

# Example usage and training loop
def train_model(model, train_loader, val_loader, num_epochs: int = 100):
    """
    Training loop for the stochastic volatility model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = FinancialLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch)
            
            # Compute loss
            losses = criterion(predictions, batch)
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(total_loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                predictions = model(batch)
                losses = criterion(predictions, batch)
                val_losses.append(losses['total'].item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.6f}')
            print(f'  Val Loss: {avg_val_loss:.6f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')

if __name__ == "__main__":
    # Example configuration
    config = create_model_config(
        vol_signals_dim=10,    # e.g., VIX, realized vol, volume, etc.
        interest_signals_dim=5, # e.g., Fed funds rate, yield curve, etc.
        macro_features_dim=8,   # e.g., unemployment, inflation, GDP, etc.
        seq_len=30             # 30 days lookback
    )
    
    # Initialize model
    model = StochasticVolatilityModel(config)
    
    print("Model initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nModel architecture:")
    print(model)