import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.ingestion.api_wrapper import fetch_latest_eia_data
from src.models.baseline import DiurnalRollingMeanBaseline
from src.evaluation.visualizer import plot_fuel_forecast

if __name__ == "__main__":
    print("--- STARTING GRIDPULSE FORECASTING PIPELINE ---")
    try:
        # Pull 14 days of dynamic fuel-mix data for the California grid
        raw_grid_data = fetch_latest_eia_data(region_id="CISO", days_back=30)
        baseline_model = DiurnalRollingMeanBaseline(window_days=7)
        forecast_matrix = baseline_model.predict(raw_grid_data)
        plot_fuel_forecast(actual_df=raw_grid_data, forecast_df=forecast_matrix, fuel_name="Hydro")

        # Extract exactly 6:00 PM for the 7 days leading up to and including 2026-06-07
        hour_df = raw_grid_data[raw_grid_data.index.hour == 18]

        # Filters the data until the prediction day.
        # We then take the last 7 rows to get the 7 days prior to and including the 7th.
        #snapshot_df = hour_df.loc[:"2026-06-15"]
        #final_snapshot_df = snapshot_df.tail(7)
        #final_snapshot_df.to_csv("6PMForPastSevenDays.csv")

        print(f" Forecasting Complete. Prediction Matrix Shape: {forecast_matrix.shape}")
        print(forecast_matrix.tail())
    except Exception as error:
        print(f"\n End-to-end pipeline failed: {error}")

