import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_fuel_forecast(actual_df: pd.DataFrame, forecast_df: pd.DataFrame, fuel_name: str = "Solar",):
    plt.figure(figsize = [12,6])

    # Graphs just 3 days of data starting from eh earliest data point for simplicity
    actual_slice = actual_df.iloc[0:72][fuel_name]
    forecast_slice = forecast_df.iloc[0:72][fuel_name]

    # Make two graphs with the actual energy generation and forecasted energy generation
    plt.plot(actual_slice.index, actual_slice.values, label="Actual Grid Load", color="black", linewidth=2)
    plt.plot(forecast_slice.index, forecast_slice.values, label = "Diurnal Mean Prediction", color="orange", linestyle="--", linewidth=2)

    # Customize the graph to make it more readable and fits all the data in the graph
    plt.title(f"GridPulse Analysis: 3-Day {fuel_name} Generation & Forecast (CISO)", fontsize=14, fontweight="bold")
    plt.xlabel("Date & Time (UTC)", fontsize=12)
    plt.ylabel("Generation Output (Megawatts)", fontsize=12)

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    plt.xticks(rotation=25)
    plt.tight_layout()

    plt.show()
