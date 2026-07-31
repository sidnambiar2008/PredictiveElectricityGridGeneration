import torch
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from preprocessing.data_loader import GridDataLoader
from src.models.GridPulseLSTM import GridPulseLSTM
from src.models.baseline import DiurnalRollingMeanBaseline
from torch.utils.data import DataLoader, Subset


BATCH_SIZE = 32

def evaluate_models(region_id = "PJM"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset = GridDataLoader(f"../../grid_data/raw/grid_history_{region_id.lower()}_v4.csv")

    # Ensures the same breakdown of the dataset, now focusion on validation
    train_size = int(len(dataset) * 0.85)
    validation_indices = list(range(train_size, len(dataset)))
    validation_subset = Subset(dataset, validation_indices)
    validation_loader = DataLoader(validation_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = GridPulseLSTM(input_size=9, hidden_size=64).to(device)
    model.load_state_dict(torch.load(f"../../saved_models/lstm_grid_pulse_24h_{region_id.lower()}_v2.pt",
                                     map_location = device))
    model.eval()

    # Loads the scaler this specific model was trained against, instead of trusting
    # dataset.scaler (which would silently recompute from whatever grid_history_{region}_v4.csv
    # contains today, and drift out of sync with the model if that file ever changes)
    scaler = joblib.load(f"../../saved_models/scaler_{region_id.lower()}_v2.pkl")

    predictions = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch in validation_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)

            predictions.append(preds.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

        predictions = np.vstack(predictions)  # (n_samples, 24, 9)
        targets = np.vstack(targets)          # (n_samples, 24, 9)

        # scaler was fit on individual hourly rows (n_rows, 9) and only accepts
        # up to 2D input. Flatten the 24-hour forecast windows into individual
        # hourly rows before inverse-transforming — this doesn't reorder or
        # lose anything, since the scaler applies the same per-column math to
        # every row regardless of which forecast window it came from.
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

        predictions = scaler.inverse_transform(predictions)
        targets = scaler.inverse_transform(targets)

    baseline = DiurnalRollingMeanBaseline()

    # Generate one 24-hour baseline forecast per validation window, using the
    # exact same 168-hour lookback the LSTM used for that sample — so the
    # baseline is judged on the identical held-out hours as the LSTM, instead
    # of a single forecast made once at the very end of history
    baseline_forecasts = []
    for idx in validation_indices:
        historical_slice = dataset.df.iloc[idx: idx + dataset.lookback_steps]
        forecast = baseline.predict(historical_slice)
        baseline_forecasts.append(forecast[dataset.feature_cols].values)

    baseline_predictions = np.vstack(baseline_forecasts)

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

    # Calculates various statistics
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
        f"evaluation_metrics_{region_id.lower()}.csv",
        index=False
    )

if __name__ == "__main__":
    regions = ["CISO", "SWPP", "ERCO", "MISO", "ISNE", "NYIS", "PJM"]
    for region in regions:
        try:
            evaluate_models(region)
        except Exception as error:
            print(f"Evaluation failed for {region}: {error}")

