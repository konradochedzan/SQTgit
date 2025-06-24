import os
import pickle
import pandas as pd
import numpy as np

from environment import (select_best_architectures, prepare_for_tuning, hyperparameter_tuning, train_final_models,
                         compare_models)

RESULTS_DIR = 'results'
ARCHITECTURE_RESULTS_FILE = 'architecture_results.csv'
BEST_ARCHITECTURES_FILE = 'best_architectures.csv'
HYPERPARAM_RESULTS_FILE_CSV = 'hyperparam_results.csv'
HYPERPARAM_RESULTS_FILE_PICKLE = 'hyperparam_results.pickle'
FINAL_PICKLE = 'final_models.pickle'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


df = pd.read_csv('data_non_std.csv', parse_dates=['Unnamed: 0'])
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
target = df['returns'].values.astype(np.float32)
dates = pd.to_datetime(df['Date'])
tbill3m = df['tbill3m'].values.astype(np.float32)


print('Running architecture selection')
arch_path = os.path.join(RESULTS_DIR, ARCHITECTURE_RESULTS_FILE)
if os.path.exists(arch_path):
    dr_arch = pd.read_csv(arch_path)
    print('Loaded existing architecture results.')
else:
    print('Selecting best architectures...')
    df_arch = select_best_architectures(
        X=features,
        y=target,
        dates=dates,
        tbill3m=tbill3m,
        results_dir=RESULTS_DIR
    )
    df_arch.to_csv(arch_path, index=False)
    print(f'Saved architecture results to {arch_path}')

print('Preparing data and running hyperparameter tuning')
hyperparam_path_csv = os.path.join(RESULTS_DIR, HYPERPARAM_RESULTS_FILE_CSV)
hyperparam_path_pickle = os.path.join(RESULTS_DIR, HYPERPARAM_RESULTS_FILE_PICKLE)
if os.path.exists(hyperparam_path_csv):
    with open(hyperparam_path_pickle, 'rb') as f: tuning_results = pickle.load(f)
    df_tuned = pd.read_csv(os.path.join(RESULTS_DIR,HYPERPARAM_RESULTS_FILE_CSV))
    print('Loaded existing hyperparameter tuning results.')
else:
    to_tune = prepare_for_tuning(df_arch)
    tuning_results = []
    for model_class, model_type, arch_params in to_tune:
        print(f'Tuning {model_type} with architecture {arch_params}')
        res = hyperparameter_tuning(
            X=features,
            y=target,
            dates=dates,
            tbill3m=tbill3m,
            model_class=model_class,
            model_type=model_type,
            **arch_params
        )
        res.update({'model_type': model_type, 'architecture': arch_params})
        tuning_results.append(res)

    df_tuned = pd.DataFrame(tuning_results)
    df_tuned.to_csv(hyperparam_path_csv, index=False)
    with open(hyperparam_path_pickle, 'wb') as f:
        pickle.dump(tuning_results, f)
    print(f'Saved hyperparameter tuning results to {HYPERPARAM_RESULTS_FILE_CSV} and {HYPERPARAM_RESULTS_FILE_PICKLE}.')

final_pickle_path = os.path.join(RESULTS_DIR, FINAL_PICKLE)
if os.path.exists(final_pickle_path):
    with open(final_pickle_path, 'rb') as f: trained_models = pickle.load(f)
    print('Loaded existing final trained models.')
else:
    to_tune = prepare_for_tuning(df_arch)
    trained_models = train_final_models(
        X=features,
        y=target,
        dates=dates,
        tbill3m=tbill3m,
        to_tune=to_tune,
        tuning_results=tuning_results,
        final_pickle_path=final_pickle_path,
        models_dir = MODELS_DIR
    )
    print(f'Trained and saved final models and metrics')

summary_df, test_results = compare_models(
    trained_models=trained_models,
    results_dir=RESULTS_DIR
)
print(summary_df)
print(test_results)
