import pandas as pd
import numpy as np
from environment import sp500_training_pipeline

from TM_models import TemporalConvNet, TemporalFusionTransformer, TemporalFusionTransformer2
from Simple_models import SimpleConvolutional, SimpleFeedForward, SimpleLSTM, SimpleTransformer

from saving import save_model, load_model, save_predictions, load_predictions


df = pd.read_csv('data_non_std.csv', parse_dates=['Unnamed: 0'])
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
target = df['returns'].values.astype(np.float32)
dates = pd.to_datetime(df['Date'])
tbill3m = df['tbill3m'].values.astype(np.float32)

results_tcn = sp500_training_pipeline(
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m_data,
   model_class  = TemporalConvNet,
    model_type   = 'temporalconvnet',                    # or 'simpleconvolutional'
    model_kwargs = {
        'num_channels': [32, 64,128,64, 32], 'kernel_size': 5, 'dropout': 0.1
    },
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=True,
    encoding_dim=10,
    seq_length=24,
    ar_lags=1,
    batch_size=128,
    epochs=10,
    lr=0.001,
    plot_results=True,
    alpha=0.0,
    l1_ratio=0.0,
)