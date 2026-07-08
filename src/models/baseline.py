import pandas as pd
import numpy as np

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
            Calculates a 7-day rolling average for each hour of the day individually.

             This function takes historical_data dataframe to allow for fitting
             and predictions

           Args:
               region_id (string): The region code that is current being fetched
               days_back (int): The number of days back to allow

           Returns:
               dataframe:
        """
        # 1. Forward-fill then backward-fill missing data to handle API gaps or network dropouts
        df_clean = historical_data.ffill().bfill()

        # 2. Group the continuous time-series rows by the hour of the day (0 through 23)
        hourly_groups = df_clean.groupby(df_clean.index.hour)

        # 3. For each isolated hour bucket, calculate a rolling mean across the lookback window.
        # 'closed="left"' excludes the target hour itself, preventing target data leakage.
        rolling_averages = hourly_groups.rolling(window=self.window_days, min_periods=1, closed="left").mean()

        # 4. Remove the temporary hour column (index level 0) from the table
        clean_forecasts = rolling_averages.reset_index(level=0, drop=True)

        # 5. Sort all rows chronologically from the oldest hour to the newest hour
        clean_forecasts = clean_forecasts.sort_index()

        return clean_forecasts
