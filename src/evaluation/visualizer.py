import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_fuel_forecast(actual_df: pd.DataFrame, forecast_df: pd.DataFrame, fuel_name: str = "Solar",):
    """
        Plots a small slice of actual vs forecasted generation for simplicity.
        Automatically skips initial NaN values caused by the 7-day lookback window to ensure predictions
        are based with enough data points.
    """

    # 1. Dynamically locate where the forecast data actually begins (skipping the initial NaNs)
    first_valid_date = forecast_df[fuel_name].first_valid_index()

    if first_valid_date is None:
        raise ValueError(f"The forecast dataframe contains only NaN values for {fuel_name}.")


    # Graphs just 3 days of data starting from the earliest data point for simplicity
    actual_slice = actual_df.loc[first_valid_date:][fuel_name].iloc[141:165]
    forecast_slice = forecast_df.loc[first_valid_date:][fuel_name].iloc[141:165]

    plt.figure(figsize = [12,6])

    # Make two graphs with the actual energy generation and forecasted energy generation
    plt.plot(actual_slice.index, actual_slice.values, label=f"Actual {fuel_name} Generation", color="black", linewidth=2)
    plt.plot(forecast_slice.index, forecast_slice.values, label = "Diurnal Mean Prediction", color="orange", linestyle="--", linewidth=2)

    # Customize the graph to make it more readable and fits all the data in the graph
    plt.title(f"GridPulse Analysis: June 16th {fuel_name} Generation & Forecast (CISO)", fontsize=14, fontweight="bold")
    plt.xlabel("Date & Time (UTC)", fontsize=12)
    plt.ylabel("Generation Output (Megawatts)", fontsize=12)

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    plt.xticks(rotation=25)
    plt.tight_layout()

    plt.show()
