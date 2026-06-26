import pandas as pd
from src.ingestion.api_wrapper import fetch_latest_eia_data
from src.models.baseline import DiurnalRollingMeanBaseline

if __name__ == "__main__":
    print("--- STARTING GRIDPULSE FORECASTING PIPELINE ---")
    try:
        # Pull 14 days of dynamic fuel-mix data for the California grid
        raw_grid_data = fetch_latest_eia_data(region_id="CISO", days_back=14)
        baseline_model = DiurnalRollingMeanBaseline(window_days=7)
        forecast_matrix = baseline_model.predict(raw_grid_data)
        print(f" Forecasting Complete. Prediction Matrix Shape: {forecast_matrix.shape}")
        print(forecast_matrix.tail())
    except Exception as error:
        print(f"\n End-to-end pipeline failed: {error}")

