import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_real_vs_predictions(
    real_csv_path,
    model_names,
    predictions_dir=".",
    output_dir="plots",
    real_col="returns",
    date_col="date",
    plot_title="Real vs Predictions",
    y_label="Value"
):

    real_csv_path = Path(real_csv_path)
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real data
    real_df = pd.read_csv(real_csv_path, parse_dates=[date_col])
    real_df = real_df[[date_col, real_col]].rename(columns={real_col: "real"})
    real_df = real_df.sort_values(date_col)

    # For each model, plot and save
    for model in model_names:
        pred_path = predictions_dir / f"{model}_predictions.csv"
        if not pred_path.exists():
            print(f"Prediction file for model '{model}' not found at {pred_path}. Skipping.")
            continue

        pred_df = pd.read_csv(pred_path, parse_dates=[date_col])
        pred_df = pred_df[[date_col, "prediction"]].rename(columns={"prediction": model})
        pred_df = pred_df.sort_values(date_col)

        # Merge on date
        merged = pd.merge(real_df, pred_df, on=date_col, how="inner")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(merged[date_col], merged["real"], label="Real", color="black", linewidth=2)
        plt.plot(merged[date_col], merged[model], label=f"Prediction ({model})", linestyle="--")

        # X-axis: only mark the beginning of each year
        years = merged[date_col].dt.year.unique()
        year_start_dates = [merged[merged[date_col].dt.year == y][date_col].iloc[0] for y in years]
        plt.xticks(year_start_dates, [str(y) for y in years], rotation=45)

        plt.title(f"{plot_title} - {model}")
        plt.xlabel("Year")
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()

        # Save
        out_path = output_dir / f"{model}_vs_real.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot for model '{model}' to {out_path.resolve()}")


plot_real_vs_predictions(
    real_csv_path="data_non_std.csv",
    model_names=['temporalfusiontransformer_nae', 'lstm_nae', 'feedforward_nae', 'transformer_nae', 'temporalconvnet_nae', 'cnn_nae'],
    predictions_dir="results_no_autoencoder",
    output_dir="plots"
)