import os
import torch
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.preprocessing.data_loader import GridDataLoader
from src.ingestion.api_wrapper import fetch_latest_eia_data
from src.models.baseline import DiurnalRollingMeanBaseline
from src.models.GridPulseLSTM import GridPulseLSTM
from sklearn.metrics import mean_absolute_error
from src.model_config import HIDDEN_SIZE, NUM_LAYERS

def evaluate_model_performance(region_id = "PJM", fuel_name = "Solar", days_back = 15, custom_end_date = None):
    """
    Compares the accuracy of the baseline and LSTM to the historical data of the specific region/fuel

    Args:
        region_id (string): EIA region id
        fuel_name(string) : EIA fuel name
        days_back(int): Days back specified in EIA request
        custom_end_date(Datetime): Specified date to acquire historical data from

    Returns:
        dict: A dictionary containing model evaluation metrics and series data.
            - region_id (str): EIA region ID.
            - fuel_name (str): Name of the fuel type analyzed.
            - end_date (pd.Timestamp): The final date in the grid dataset index.
            - lstm_mae (float): Mean Absolute Error of the LSTM model.
            - base_mae (float): Mean Absolute Error of the baseline model.
            - improvement (float): Net improvement of LSTM over the baseline.
            - series_data (tuple): A 4-element tuple saved for graphing:
                - eval_index (pd.Series): The evaluation timeline index.
                - actuals (pd.Series): Observed historical values.
                - plot_base (pd.Series): Baseline model predictions.
                - plot_lstm (pd.Series): LSTM model predictions.
    """


    raw_grid_data = fetch_latest_eia_data(region_id = region_id, days_back = days_back, custom_end_date = custom_end_date)
    baseline_model = DiurnalRollingMeanBaseline(window_days=7)
    lstm_base_data = GridDataLoader(f"grid_data/raw/grid_history_{region_id.lower()}_v4.csv")

    # Column order must match what the model was trained on (GridDataLoader.feature_cols)
    feature_cols = lstm_base_data.feature_cols
    fuel_idx = feature_cols.index(fuel_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = GridPulseLSTM(input_size=len(feature_cols), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, forecast_horizon=lstm_base_data.forecast_horizon).to(device)
    model.load_state_dict(torch.load(f"saved_models/lstm_grid_pulse_24h_{region_id.lower()}_v2.pt", map_location=device))

    model.eval()

    # Loads the scaler this model was trained against, rather than trusting
    # lstm_base_data.scaler (which would silently recompute from whatever
    # grid_history_{region}_v4.csv contains today, and drift out of sync with the
    # model if that file is ever regenerated)
    scaler = joblib.load(f"saved_models/scaler_{region_id.lower()}_v2.pkl")

    eval_hours = lstm_base_data.lookback_steps
    forecast_hours = lstm_base_data.forecast_horizon

    baseline_predictions = []
    lstm_predictions = []

    historical_df = raw_grid_data.iloc[-(eval_hours+forecast_hours):-forecast_hours]

    baseline_forecast = baseline_model.predict(historical_df)
    baseline_predictions = baseline_forecast[fuel_name].iloc[0:24].values

    live_raw_matrix = historical_df.reindex(columns=feature_cols, fill_value=0).ffill().bfill().values
    live_scaled_matrix = scaler.transform(live_raw_matrix)

    torch_input = torch.tensor(live_scaled_matrix, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(torch_input).squeeze(0).cpu().numpy()
        unscaled_pred = scaler.inverse_transform(prediction)

        lstm_predictions = unscaled_pred[0:24, fuel_idx]


    eval_index = raw_grid_data.index[-forecast_hours:]
    actuals = raw_grid_data[fuel_name].iloc[-forecast_hours:]

    plot_lstm = pd.Series(lstm_predictions, index=eval_index)
    plot_base = pd.Series(baseline_predictions, index=eval_index)

    plt.figure(figsize=[14, 6])
    plt.plot(eval_index, actuals.values, label="True Actual Generation", color="black", linewidth=2)
    plt.plot(eval_index, plot_base.values, label="Diurnal Mean Baseline", color="orange", linestyle="--",
             linewidth=1.5)
    plt.plot(eval_index, plot_lstm.values, label="LSTM Rolling Forecast", color="red", linestyle="solid",
             linewidth=1.5)

    plt.title(f"{region_id} Continuous Backtest Performance Analysis ({fuel_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Timeline Date & Time (UTC)", fontsize=12)
    plt.ylabel("Generation Output (Megawatts)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs("visuals", exist_ok=True)
    plt.savefig(f"visuals/{fuel_name.lower()}_continuous_backtest_comparison_{region_id.lower()}.png", dpi=300)
    plt.show()
    plt.close()

    lstm_mae = mean_absolute_error(actuals, plot_lstm)
    base_mae = mean_absolute_error(actuals, plot_base)
    net_improvement = ((base_mae - lstm_mae) / base_mae) * 100

    return {
        "region_id": region_id,
        "fuel_name": fuel_name,
        "end_date": raw_grid_data.index[-1].date(),
        "lstm_mae": lstm_mae,
        "base_mae": base_mae,
        "improvement": net_improvement,
        "series_data": (eval_index, actuals, plot_base, plot_lstm)  # Saved for graphing later
    }

if __name__ == "__main__":
    print("Testing model on historical data!!!")

    summer_test_date = pd.Timestamp("2026-07-28")
    winter_test_date = pd.Timestamp("2026-01-15 14:00:00")
    spring_storm_date = pd.Timestamp("2026-04-15")
    winter_test_date_two = pd.Timestamp("2025-12-22")

    results = evaluate_model_performance(region_id = "PJM", fuel_name = "Natural Gas", days_back=15, custom_end_date=winter_test_date_two)

    print("\n--- QUICK RESULTS SUMMARY ---")
    print(f"Target Fuel Source: {results['fuel_name']}")
    print(f"Evaluation End Date: {results['end_date']}")
    print(f"LSTM Average Deviation (MAE): {results['lstm_mae']:.2f} MW")
    print(f"Baseline Average Deviation (MAE): {results['base_mae']:.2f} MW")
    print(f"Total Neural Network Advantage: {results['improvement']:.1f}% lower error")

