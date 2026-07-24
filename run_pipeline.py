import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ingestion.data_loader import GridDataLoader
from src.ingestion.api_wrapper import fetch_latest_eia_data
from src.models.baseline import DiurnalRollingMeanBaseline
from src.evaluation.visualizer import plot_fuel_forecast
from src.models.GridPulseLSTM import GridPulseLSTM

if __name__ == "__main__":
    print("--- STARTING GRIDPULSE FORECASTING PIPELINE ---")

    try:
        # Pull 14 days of dynamic fuel-mix data for the California grid
        raw_grid_data = fetch_latest_eia_data(region_id="PJM", days_back=30)
        baseline_model = DiurnalRollingMeanBaseline(window_days=7)
        forecast_matrix = baseline_model.predict(raw_grid_data)
        forecast_matrix.index = raw_grid_data.index


        lstm_base_data = GridDataLoader("grid_data/raw/grid_history_pjm_v3.csv")



        feature_cols = lstm_base_data.feature_cols



        # Reindex based on the same feature to ensure consistency among the models dataset
        live_raw_matrix = raw_grid_data.reindex(columns=feature_cols, fill_value=0).ffill().bfill().values

        # Ensures the data is looked through the same range, (-1,1), as in training
        live_scaled_matrix = lstm_base_data.scaler.transform(live_raw_matrix)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = GridPulseLSTM(input_size=9, hidden_size=64).to(device)
        model.load_state_dict(torch.load("saved_models/lstm_grid_pulse_pjm_v2.pt", map_location=device))

        model.eval()

        lstm_scaled_preds = []
        lookback = 168



        with torch.no_grad():
            for i in range(lookback, len(live_scaled_matrix)):
                memory_cell_window = live_scaled_matrix[i-lookback:i]
                memory_cell_tensor = torch.tensor(memory_cell_window, dtype=torch.float32).unsqueeze(0).to(device)



                prediction = model(memory_cell_tensor)

                lstm_scaled_preds.append(prediction.squeeze(0).cpu().numpy())

            lstm_unscaled_matrix = lstm_base_data.scaler.inverse_transform(lstm_scaled_preds)

           # Ensures that there is no prediction volatility when a feature input is 0
            for idx, col in enumerate(feature_cols):
                if live_raw_matrix[:, idx].max() == 0:
                    lstm_unscaled_matrix[:, idx] = 0

            lstm_df = pd.DataFrame(data = lstm_unscaled_matrix, columns = feature_cols, index = raw_grid_data.index[-len(lstm_unscaled_matrix):])

        # Make the plot so models are comparable
        plot_fuel_forecast(actual_df=raw_grid_data, lstm_df= lstm_df, baseline_df=forecast_matrix, fuel_name="Solar", region_id = "PJM")


    except Exception as error:
        print(f"\n End-to-end pipeline failed: {error}")

