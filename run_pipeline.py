import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ingestion.data_loader import GridDataLoader
from src.ingestion.api_wrapper import fetch_latest_eia_data
from src.models.baseline import DiurnalRollingMeanBaseline
from src.evaluation.future_visualizer import plot_fuel_forecast
from src.models.GridPulseLSTM import GridPulseLSTM

if __name__ == "__main__":
    print("--- STARTING GRIDPULSE FORECASTING PIPELINE ---")

    try:
        # Pull 14 days of dynamic fuel-mix data for the California grid
        raw_grid_data = fetch_latest_eia_data(region_id="PJM", days_back=11)
        baseline_model = DiurnalRollingMeanBaseline(window_days=7)

        backtest_cutoff = raw_grid_data.index[-1]
        historical_df = raw_grid_data.loc[:backtest_cutoff].iloc[-168:]

        forecast_matrix = baseline_model.predict(historical_df)

        lstm_base_data = GridDataLoader("grid_data/raw/grid_history_pjm_v3.csv")

        feature_cols = lstm_base_data.feature_cols

        # Reindex based on the same feature to ensure consistency among the models dataset
        live_raw_matrix = historical_df.reindex(columns=feature_cols, fill_value=0).ffill().bfill().values

        # Ensures the data is looked through the same range, (-1,1), as in training
        live_scaled_matrix = lstm_base_data.scaler.transform(live_raw_matrix)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = GridPulseLSTM(input_size=9, hidden_size=64, forecast_horizon=24).to(device)
        model.load_state_dict(torch.load("saved_models/lstm_grid_pulse_24h_pjm_v1.pt", map_location=device))

        model.eval()

        lookback = 168

        with torch.no_grad():
            memory_cell_window = live_scaled_matrix[-lookback:]
            memory_cell_tensor = torch.tensor(memory_cell_window, dtype=torch.float32).unsqueeze(0).to(device)

            prediction = model(memory_cell_tensor)

            lstm_scaled_matrix = prediction.squeeze(0).cpu().numpy()

            lstm_unscaled_matrix = lstm_base_data.scaler.inverse_transform(lstm_scaled_matrix)

            forecast_index = pd.date_range(start = raw_grid_data.index[-1] + pd.Timedelta(hours=1), periods = 24, freq = "h")

            lstm_df = pd.DataFrame(data = lstm_unscaled_matrix, columns = feature_cols, index=forecast_index)

        # Make the plot so models are comparable
        plot_fuel_forecast(actual_df=historical_df, lstm_df= lstm_df, baseline_df=forecast_matrix, fuel_name="Wind", region_id = "PJM")


    except Exception as error:
        print(f"\n End-to-end pipeline failed: {error}")

