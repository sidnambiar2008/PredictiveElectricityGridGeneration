import torch
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.ingestion.data_loader import GridDataLoader
from src.models.GridPulseLSTM import GridPulseLSTM
from src.models.baseline import DiurnalRollingMeanBaseline
from torch.utils.data import DataLoader, Subset


BATCH_SIZE = 32

def evaluate_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset = GridDataLoader("../../grid_data/raw/grid_history_pjm_v3.csv")

    train_size = int(len(dataset) * 0.85)

    validation_indices = list(range(train_size, len(dataset)))

    validation_subset = Subset(dataset, validation_indices)

    validation_loader = DataLoader(validation_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = GridPulseLSTM(input_size=9, hidden_size=64).to(device)
    model.load_state_dict(torch.load("../../saved_models/lstm_grid_pulse_pjm_v2.pt",
                                     map_location = device))

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch in validation_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)

            predictions.append(preds.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

        predictions = np.vstack(predictions)
        targets = np.vstack(targets)

        predictions = dataset.scaler.inverse_transform(predictions)
        targets = dataset.scaler.inverse_transform(targets)

    baseline = DiurnalRollingMeanBaseline()

    baseline_predictions = baseline.predict(dataset.df)
    baseline_predictions = baseline_predictions.iloc[train_size+ dataset.lookback_steps:]
    baseline_predictions = baseline_predictions[dataset.feature_cols].values

    print("LSTM predictions:", predictions.shape)
    print("Baseline predictions:", baseline_predictions.shape)
    print("Targets:", targets.shape)

    fuels = []
    baseline_maes = []
    baseline_rmses = []
    baseline_r2s = []

    lstm_maes = []
    lstm_rmses = []
    lstm_r2s = []

    for i, fuel in enumerate(dataset.feature_cols):
        lstm_mae = mean_absolute_error(
            targets[:, i],
            predictions[:, i]
        )

        lstm_rmse = np.sqrt(
            mean_squared_error(
            targets[:, i],
            predictions[:, i]
            )
        )

        lstm_r2 = r2_score(
            targets[:, i],
            predictions[:, i]
        )

        baseline_mae = mean_absolute_error(
            targets[:, i],
            baseline_predictions[:, i]
        )

        baseline_rmse = np.sqrt(
            mean_squared_error(
                targets[:, i],
                baseline_predictions[:, i]
            )
        )

        baseline_r2 = r2_score(
            targets[:, i],
            baseline_predictions[:, i]
        )

        fuels.append(fuel)

        baseline_maes.append(baseline_mae)
        baseline_rmses.append(baseline_rmse)
        baseline_r2s.append(baseline_r2)

        lstm_maes.append(lstm_mae)
        lstm_rmses.append(lstm_rmse)
        lstm_r2s.append(lstm_r2)

    results = pd.DataFrame({
        "Fuel": fuels,
        "Baseline MAE": baseline_maes,
        "LSTM MAE": lstm_maes,
        "Baseline RMSE": baseline_rmses,
        "LSTM RMSE": lstm_rmses,
        "Baseline R2": baseline_r2s,
        "LSTM R2": lstm_r2s
    })
    print(results)
    print("\nAverage Performance")
    print(results.mean(numeric_only=True))

    results["MAE Improvement %"] = np.where(
        results["Baseline MAE"] != 0,
        ((results["Baseline MAE"] - results["LSTM MAE"])
         / results["Baseline MAE"]) * 100,
        0
    )

    results["RMSE Improvement %"] = np.where(
        results["Baseline RMSE"] != 0,
        ((results["Baseline RMSE"] - results["LSTM RMSE"])
         / results["Baseline RMSE"]) * 100,
        0
    )

    results.to_csv(
        "evaluation_metrics_pjm.csv",
        index=False
    )

if __name__ == "__main__":
    evaluate_models()

