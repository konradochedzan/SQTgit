import pandas as pd
import numpy as np
from environment import sp500_training_pipeline
from pathlib import Path
from TM_models import TemporalConvNet, TemporalFusionTransformer
from Simple_models import SimpleConvolutional, SimpleFeedForward, SimpleLSTM, SimpleTransformer
from saving import save_predictions, save_model, load_model, load_predictions


df = pd.read_csv('data_non_std.csv', parse_dates=['date'])
df.rename(columns={'date': 'Date'}, inplace=True)
features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
target = df['returns'].values.astype(np.float32)
dates = pd.to_datetime(df['Date'])
tbill3m = df['tbill3m'].values.astype(np.float32)


# AUTOENCODER

results_feedforward_ae = sp500_training_pipeline(
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class  = SimpleFeedForward,
    model_type   = 'feedforward', 
    model_kwargs = {
        'hidden_dim': 140,
        'dropout': 0.1
    },
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=True,
    encoding_dim=10,
    seq_length=24,
    ar_lags=30,
    batch_size=128,
    epochs=120,
    lr=0.001,
    plot_results=True,
    do_print=True,
    alpha=0.0,
    l1_ratio=0.0,
    plt_show = False
)

save_predictions(
    dates=results_feedforward_ae["all_test_dates"],
    predictions=results_feedforward_ae["all_test_predictions"],
    csv_path="results_autoencoder/feedforward_test_predictions.csv"
)

save_predictions(
    dates=results_feedforward_ae["all_train_dates"],
    predictions=results_feedforward_ae["all_train_predictions"],
    csv_path="results_autoencoder/feedforward_train_predictions.csv"
)

for k, fold_model in enumerate(results_feedforward_ae["models"], 1):
    save_model(
        fold_model,
        f"models_autoencoder/feedforward_fold_{k:02d}.joblib"
    )

results_lstm_ae = sp500_training_pipeline(    X=features,
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class  = SimpleLSTM,
    model_type   = 'lstm', 
    model_kwargs = {'hidden_dim': 200,
        'dropout': 0.1,
        'num_layers': 3
    },
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=False,
    encoding_dim=10,
    seq_length=24,
    ar_lags=30,
    batch_size=128,
    epochs=120,
    lr=0.001,
    plot_results=True,
    do_print=True,
    alpha=0.0,
    l1_ratio=0.0,
    plt_show = False
)

save_predictions(
    dates=results_lstm_ae["all_test_dates"],
    predictions=results_lstm_ae["all_test_predictions"],
    csv_path="results_autoencoder/lstm_test_predictions.csv"
) 
save_predictions(
    dates=results_lstm_ae["all_train_dates"],
    predictions=results_lstm_ae["all_train_predictions"],
    csv_path="results_autoencoder/lstm_train_predictions.csv"
)
for k, fold_model in enumerate(results_lstm_ae["models"], 1):
    save_model(
        fold_model,
        f"models_autoencoder/lstm_fold_{k:02d}.joblib"
    ) 

results_transformer_ae = sp500_training_pipeline(
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class  = SimpleTransformer,
    model_type   = 'transformer', 
    model_kwargs = {
        'model_dim': 128,
        'num_layers': 2,
        'dropout': 0.1,
        'nhead': 8,},
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=False,
    encoding_dim=10,
    seq_length=24,
    ar_lags=30,
    batch_size=128,
    epochs=120,
    lr=0.001,
    plot_results=True,
    do_print=True,
    alpha=0.0,
    l1_ratio=0.0,
    plt_show = False)


save_predictions(
    dates=results_transformer_ae["all_test_dates"],
    predictions=results_transformer_ae["all_test_predictions"],
    csv_path="results_autoencoder/transformer_test_predictions.csv"
) 
save_predictions(
    dates=results_transformer_ae["all_train_dates"],
    predictions=results_transformer_ae["all_train_predictions"],
    csv_path="results_autoencoder/transformer_train_predictions.csv"
)
for k, fold_model in enumerate(results_transformer_ae["models"], 1):
    save_model(
        fold_model,
        f"models_autoencoder/transformer_fold_{k:02d}.joblib"
    ) 


results_convolutional_ae = sp500_training_pipeline(
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class  = SimpleConvolutional,
    model_type   = 'convolutional',
    model_kwargs = {'num_channels':[32,64,32],
        'kernel_size': 5,
        'dropout': 0.1,
        'seq_length': 24
    },
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=True,
    encoding_dim=10,
    seq_length=24,
    ar_lags=30,
    batch_size=128,
    epochs=120,
    lr=0.001,
    plot_results=True,
    do_print=True,
    alpha=0.0,
    l1_ratio=0.0,
    plt_show = False
)

save_predictions(
    dates=results_convolutional_ae["all_test_dates"],
    predictions=results_transformer_ae["all_test_predictions"],
    csv_path="results_autoencoder/cnn_test_predictions.csv"
) 
save_predictions(
    dates=results_convolutional_ae["all_train_dates"],
    predictions=results_convolutional_ae["all_train_predictions"],
    csv_path="results_autoencoder/cnn_train_predictions.csv"
)
for k, fold_model in enumerate(results_convolutional_ae["models"], 1):
    save_model(
        fold_model,
        f"models_autoencoder/cnn_fold_{k:02d}.joblib"
    ) 



# NO AUTOENCODER


results_feedforward_nae = sp500_training_pipeline(
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class  = SimpleFeedForward,
    model_type   = 'feedforward', 
    model_kwargs = {
        'hidden_dim': 140,
        'dropout': 0.1
    },
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=False,
    encoding_dim=10,
    seq_length=24,
    ar_lags=30,
    batch_size=128,
    epochs=120,
    lr=0.001,
    plot_results=True,
    do_print=True,
    alpha=0.0,
    l1_ratio=0.0,
    plt_show = False
)

save_predictions(
    dates=results_feedforward_nae["all_test_dates"],
    predictions=results_feedforward_nae["all_test_predictions"],
    csv_path="results_no_autoencoder/feedforward_test_predictions.csv"
)

save_predictions(
    dates=results_feedforward_ae["all_train_dates"],
    predictions=results_feedforward_ae["all_train_predictions"],
    csv_path="results_no_autoencoder/feedforward_train_predictions.csv"
)

for k, fold_model in enumerate(results_feedforward_ae["models"], 1):
    save_model(
        fold_model,
        f"models_autoencoder/feedforward_fold_{k:02d}.joblib"
    )

results_lstm_ae = sp500_training_pipeline(    X=features,
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class  = SimpleLSTM,
    model_type   = 'lstm', 
    model_kwargs = {'hidden_dim': 200,
        'dropout': 0.1,
        'num_layers': 3
    },
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=False,
    encoding_dim=10,
    seq_length=24,
    ar_lags=30,
    batch_size=128,
    epochs=120,
    lr=0.001,
    plot_results=True,
    do_print=True,
    alpha=0.0,
    l1_ratio=0.0,
    plt_show = False
)

save_predictions(
    dates=results_lstm_ae["all_test_dates"],
    predictions=results_lstm_ae["all_test_predictions"],
    csv_path="results_autoencoder/lstm_test_predictions.csv"
) 
save_predictions(
    dates=results_lstm_ae["all_train_dates"],
    predictions=results_lstm_ae["all_train_predictions"],
    csv_path="results_autoencoder/lstm_train_predictions.csv"
)
for k, fold_model in enumerate(results_lstm_ae["models"], 1):
    save_model(
        fold_model,
        f"models_autoencoder/lstm_fold_{k:02d}.joblib"
    ) 

results_transformer_ae = sp500_training_pipeline(
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class  = SimpleTransformer,
    model_type   = 'transformer', 
    model_kwargs = {
        'model_dim': 128,
        'num_layers': 2,
        'dropout': 0.1,
        'nhead': 8,},
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=False,
    encoding_dim=10,
    seq_length=24,
    ar_lags=30,
    batch_size=128,
    epochs=120,
    lr=0.001,
    plot_results=True,
    do_print=True,
    alpha=0.0,
    l1_ratio=0.0,
    plt_show = False)


save_predictions(
    dates=results_transformer_ae["all_test_dates"],
    predictions=results_transformer_ae["all_test_predictions"],
    csv_path="results_autoencoder/transformer_test_predictions.csv"
) 
save_predictions(
    dates=results_transformer_ae["all_train_dates"],
    predictions=results_transformer_ae["all_train_predictions"],
    csv_path="results_autoencoder/transformer_train_predictions.csv"
)
for k, fold_model in enumerate(results_transformer_ae["models"], 1):
    save_model(
        fold_model,
        f"models_autoencoder/transformer_fold_{k:02d}.joblib"
    ) 


results_convolutional_ae = sp500_training_pipeline(
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class  = SimpleConvolutional,
    model_type   = 'convolutional',
    model_kwargs = {'num_channels':[32,64,32],
        'kernel_size': 5,
        'dropout': 0.1,
        'seq_length': 24
    },
    window_strategy='rolling',
    train_window_years=3,
    test_window_years=1,
    use_autoencoder=True,
    encoding_dim=10,
    seq_length=24,
    ar_lags=30,
    batch_size=128,
    epochs=120,
    lr=0.001,
    plot_results=True,
    do_print=True,
    alpha=0.0,
    l1_ratio=0.0,
    plt_show = False
)

save_predictions(
    dates=results_convolutional_ae["all_test_dates"],
    predictions=results_transformer_ae["all_test_predictions"],
    csv_path="results_autoencoder/cnn_test_predictions.csv"
) 
save_predictions(
    dates=results_convolutional_ae["all_train_dates"],
    predictions=results_convolutional_ae["all_train_predictions"],
    csv_path="results_autoencoder/cnn_train_predictions.csv"
)
for k, fold_model in enumerate(results_convolutional_ae["models"], 1):
    save_model(
        fold_model,
        f"models_autoencoder/cnn_fold_{k:02d}.joblib"
    ) 

