import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

from TM_models import TemporalConvNet, TemporalFusionTransformer

# Load data
df = pd.read_csv('data_non_std.csv', parse_dates=['Unnamed: 0']).rename(columns={'Unnamed: 0':'Date'})
features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
target = df['returns'].values.astype(np.float32)
dates = pd.to_datetime(df['Date']).values
tbill3m_data = df['tbill3m'].values.astype(np.float32)

# We compare 2 models: Temporal Convnet and Temporal Fusion Transformer. Each entry specifies the class and the grid
# of parameters to search.
model_grids = {
    'tcn': {
        'class': TemporalConvNet,
        'param_grid': {
            'input_dim': [8],
            'num_channels': [[64,128,64], [32,64,32]],
            'pool': ['last', 'avg']
        }
    },
    'tft': {
        'class': TemporalFusionTransformer,
        'param_grid': {
            'input_dim': [8],
            'hidden_dim': [64, 128],
            'num_heads': [4, 8],
            'num_layers': [1, 2],
            'dropout': [0.1, 0.2]
        }
    }
}

# This ensures that each test fold occurs strictly after the training period.
tsvc = TimeSeriesSplit(n_splits=5)
results = []

# Iterate through each model type and all possible combinations of its parameters
for model_name, cfg in model_grids.items():
    ModelClass = cfg['class']
    keys, values = zip(*cfg['param_grid'].items())

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        mses = []

        for train_idx, test_idx in tsvc.split(features):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]

            # Create a new model instance for this fold and parameter set
            model = ModelClass(**params)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            # Train for a small number of epochs (for example 5) for quick grid search
            model.train()
            for epoch in range(5):
                inp = torch.FloatTensor(X_train)
                lbl = torch.FloatTensor(y_train).unsqueeze(1)
                optimizer.zero_grad()
                preds = model(inp)
                loss = criterion(preds.squeeze(), lbl.squeeze())
                loss.backward()
                optimizer.step()

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_preds = model(torch.FloatTensor(X_test))

            # Compute mean squared error on fold
            mse = mean_squared_error(y_test, test_preds.squeeze().numpy())
            mses.append(mse)
        # Aggregate results for this set of model and parameters
        results.append({
            'model': model_name,
            **params,
            'avg_mse': np.mean(mses),
            'std_mse': np.std(mses)
        })

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('avg_mse').reset_index(drop=True)
        print('Grid search results sorted by average mean squared error:')
        print(df_results)






