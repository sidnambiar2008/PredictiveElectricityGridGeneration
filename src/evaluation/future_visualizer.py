import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_fuel_forecast(actual_df: pd.DataFrame, lstm_df: pd.DataFrame, baseline_df: pd.DataFrame, fuel_name: str = "Solar",region_id: str = "CISO") -> None:
    """
        Plots a small slice of actual vs forecasted generation for simplicity.
        Automatically skips initial NaN values caused by the 7-day lookback window to ensure predictions
        are based with enough data points.
    """

    # 1. Dynamically locate where the forecast data actually begins (skipping the initial NaNs)
    first_valid_date = baseline_df[fuel_name].first_valid_index()

    if first_valid_date is None:
        raise ValueError(f"The forecast dataframe contains only NaN values for {fuel_name}.")

    lookback_window = 168
    forecast_hours = 24

    # Graphs 7 days of data starting from the earliest data point for simplicity
    actual_slice = actual_df.iloc[-lookback_window:][fuel_name]
    #forecast_index = lstm_df.index

    #actual_future = actual_df.loc[actual_df.index.intersection(forecast_index)]
    baseline_forecast = baseline_df[fuel_name]
    lstm_forecast = lstm_df[fuel_name]

    plt.figure(figsize = [12,6])

    last_actual_value = actual_slice.values[-1]
    last_actual_index = actual_slice.index[-1]

    baseline_forecast_list = baseline_forecast.tolist()
    lstm_forecast_list = lstm_forecast.tolist()

    extended_index = [last_actual_index] + list(lstm_forecast.index)

    extended_baseline = pd.Series(data=[last_actual_value] + baseline_forecast_list, index = extended_index)
    extended_lstm = pd.Series(data=[last_actual_value] + lstm_forecast_list, index = extended_index)

    # Make two graphs with the actual energy generation and forecasted energy generation
    plt.plot(actual_slice.index, actual_slice.values, label=f"Actual {fuel_name} Generation", color="black", linewidth=2)
    plt.plot(extended_baseline.index, extended_baseline.values, label = "Diurnal Mean Prediction", color="orange", linestyle="--", linewidth=2)
    plt.plot(extended_lstm.index, extended_lstm.values, label="LSTM Horizon Forecast", color="red", linestyle="solid", linewidth=2)
    #plt.plot(ground_truth_df.index, ground_truth_df[fuel_name].values, label="Ground Truth", color="green", linestyle="-", linewidth=2)

    # Customize the graph to make it more readable and fits all the data in the graph
    plt.title(f"GridPulse Analysis: {region_id} {fuel_name} Generation & Forecast", fontsize=14, fontweight="bold")
    plt.xlabel("Date & Time (UTC)", fontsize=12)
    plt.ylabel("Generation Output (Megawatts)", fontsize=12)

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    plt.xticks(rotation=25)
    plt.tight_layout()

    os.makedirs("visuals", exist_ok=True)
    plt.savefig(f"visuals/{fuel_name.lower()}_baseline_lstm_comparison_{region_id.lower()}_{lookback_window}.png", dpi=300)
    plt.show()

    plt.close()