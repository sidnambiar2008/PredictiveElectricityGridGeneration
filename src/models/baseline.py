import pandas as pd

class DiurnalRollingMeanBaseline:
    def __init__(self, window_days: int = 7):
        self.window_days = window_days

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None):
        """
            Maintains standard ML architecture compatibility.
            No training phase is required, as this is a naive math baseline.
        """
        return self

    def predict(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
            Calculates a 7-day rolling average for the next 24 hours

           Args:
               historical_data (DataFrame): Hourly generation history, one column per fuel type

           Returns:
               DataFrame: 24-hour forecast, same columns as historical_data
        """
        # 1. Forward-fill then backward-fill missing data to handle API gaps or network dropouts
        df_clean = historical_data.ffill().bfill()

        forecasts = []

        last_timestamp = df_clean.index[-1]
        forecast_times = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=24,
            freq="h"
        )

        for timestamp in forecast_times:
            hour = timestamp.hour
            hour_values = df_clean[df_clean.index.hour == hour]
            prediction = hour_values.tail(self.window_days).mean()
            forecasts.append(prediction)

        forecast_df = pd.DataFrame(forecasts, index=forecast_times, columns = df_clean.columns)


        return forecast_df
