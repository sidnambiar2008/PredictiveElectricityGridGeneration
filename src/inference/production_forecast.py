import pandas as pd
import torch
import joblib

from src.preprocessing.data_loader import GridDataLoader
from src.ingestion.api_wrapper import fetch_latest_eia_data
from src.models.baseline import DiurnalRollingMeanBaseline
from src.visualization.future_visualizer import plot_fuel_forecast
from src.models.GridPulseLSTM import GridPulseLSTM

def predict_24_hours_hours_ahead(region_id = "PJM"):
    # Pull 14 days of dynamic fuel-mix data for the California grid
    raw_grid_data = fetch_latest_eia_data(region_id, days_back=11)
    baseline_model = DiurnalRollingMeanBaseline(window_days=7)

    historical_df = raw_grid_data.iloc[-168:]

    forecast_matrix = baseline_model.predict(historical_df)

    lstm_base_data = GridDataLoader(f"grid_data/raw/grid_history_{region_id.lower()}_v4.csv")

    feature_cols = lstm_base_data.feature_cols

    # Loads the scaler this model was trained against, rather than trusting
    # lstm_base_data.scaler (which would silently recompute from whatever
    # grid_history_{region}_v4.csv contains today, and drift out of sync with the
    # model if that file is ever regenerated)
    scaler = joblib.load(f"saved_models/scaler_{region_id.lower()}_v2.pkl")

    # Reindex based on the same feature to ensure consistency among the models dataset
    live_raw_matrix = historical_df.reindex(columns=feature_cols, fill_value=0).ffill().bfill().values

    # Ensures the data is looked through the same range, (-1,1), as in training
    live_scaled_matrix = scaler.transform(live_raw_matrix)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GridPulseLSTM(input_size=9, hidden_size=64, forecast_horizon=24).to(device)
    model.load_state_dict(torch.load(f"saved_models/lstm_grid_pulse_24h_{region_id.lower()}_v2.pt", map_location=device))

    model.eval()

    lookback = 168

    with torch.no_grad():
        memory_cell_window = live_scaled_matrix[-lookback:]
        memory_cell_tensor = torch.tensor(memory_cell_window, dtype=torch.float32).unsqueeze(0).to(device)

        prediction = model(memory_cell_tensor)

        lstm_scaled_matrix = prediction.squeeze(0).cpu().numpy()

        lstm_unscaled_matrix = scaler.inverse_transform(lstm_scaled_matrix)

        forecast_index = pd.date_range(start=raw_grid_data.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")

        lstm_df = pd.DataFrame(data=lstm_unscaled_matrix, columns=feature_cols, index=forecast_index)

    clean_assets = ["Solar", "Wind", "Hydro", "Nuclear"]

    # Answers how clean the energy grid is now
    clean_now = lstm_df[clean_assets].iloc[0].sum()
    total_now = lstm_df.iloc[0].sum()
    cleanliness_now = (clean_now / total_now * 100) if total_now > 0 else 0

    #  Answers which hour is forecasted to be the cleanest
    clean_hourly = lstm_df[clean_assets].sum(axis=1)
    total_hourly = lstm_df.sum(axis=1)
    clean_grid_percentage = (clean_hourly / total_hourly * 100).fillna(0)


    # EIA Standard Emissions Coefficients (Pounds of CO2 emitted per MWh)
    co2_factors = {
        "Coal": 2210.0,
        "Natural Gas": 920.0,
        "Petroleum": 2110.0,
        "Other": 920.0,  # EIA standard proxy assumption
        "Solar": 0.0,
        "Wind": 0.0,
        "Hydro": 0.0,
        "Nuclear": 0.0
    }

    # 1. Calculate hourly CO2 pounds across your 24-hour forecast
    hourly_co2_lbs = pd.Series(0.0, index=lstm_df.index)
    for fuel, factor in co2_factors.items():
        if fuel in lstm_df.columns:
            hourly_co2_lbs += lstm_df[fuel] * factor

    # 2. Convert to Metric Tons (1 Metric Ton = 2204.62 lbs)
    hourly_co2_tons = hourly_co2_lbs / 2204.62

    # 3. Calculate "Right Now" and "Tomorrow Cumulative Total"
    co2_now = hourly_co2_tons.iloc[0]
    total_co2_tomorrow = hourly_co2_tons.sum()

    # 4. Calculate Grid Carbon Intensity (Lbs of CO2 emitted per Total MWh generated)
    total_generation_hourly = lstm_df.sum(axis=1)
    grid_intensity_hourly = (hourly_co2_lbs / total_generation_hourly).fillna(0)

    print("=======================================================")
    print(f"CURRENT STATUS (Right Now):")
    print(f"  ├─ Grid Cleanliness:     {cleanliness_now:.1f}% Clean Energy")
    print(f"  ├─ Clean Generation:     {clean_now:,.0f} MW")
    print(f"  ├─ Total Grid Demand:    {total_now:,.0f} MW")
    print(f"  └─ Real-Time Carbon Box: {co2_now:.2f} Metric Tons CO2/hr")
    print("-------------------------------------------------------")
    print("DAY-AHEAD HORIZON FORECAST (Tomorrow's Peak Windows):")
    print(
        f"  ├─ Cleanest Scheduled Hour: {pd.to_datetime(clean_grid_percentage.idxmax()).strftime('%H:%M')} UTC ({clean_grid_percentage.max():.1f}% Clean)")
    print(
        f"  ├─ Dirtiest Scheduled Hour:  {pd.to_datetime(clean_grid_percentage.idxmin()).strftime('%H:%M')} UTC ({clean_grid_percentage.min():.1f}% Clean)")
    print(f"  ├─ 24-Hour Carbon Intensity: {grid_intensity_hourly.mean():.1f} lbs CO2/MWh")
    print(f"  └─ Total Expected Tomorrow:  {total_co2_tomorrow:,.1f} Metric Tons of CO2")
    print("=======================================================\n")

    # Make the plot so models are comparable
    plot_fuel_forecast(actual_df=historical_df, lstm_df=lstm_df, baseline_df=forecast_matrix, fuel_name="Wind",
                       region_id = region_id)


if __name__ == "__main__":
    print("--- STARTING GRIDPULSE FORECASTING PIPELINE ---")

    try:
        predict_24_hours_hours_ahead("PJM")

    except Exception as error:
        print(f"\n End-to-end pipeline failed: {error}")

