import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class GridDataLoader(Dataset):
    def __init__(self, grid_history_csv, lookback_steps: int=168):
        self.lookback_steps = lookback_steps
        self.feature_cols = ["Coal", "Geothermal", "Hydro", "Natural Gas", "Nuclear", "Petroleum", "Wind", "Solar", "Other"]

        # Make the data a pandas recognized datatype to allow for chronological sorting
        df_raw = pd.read_csv(grid_history_csv, index_col=0)
        df_raw.index = pd.to_datetime(df_raw.index)

        # Sort the data and remove duplicates to allow the memory cell to traverse chronologically
        df_sorted = df_raw.sort_index()
        df_unique = df_sorted[~df_sorted.index.duplicated(keep='first')]

        # Resample to hourly intervals for time series forecasting.
        # Fill gaps with ffill (previous hour) and bfill (start of series).
        # Crucial to prevent PyTorch from crashing on NaN values during training.
        df_continuous = df_unique.asfreq('h')
        self.df = df_continuous.ffill().bfill()
        df_continuous = df_unique.asfreq('h')
        self.df = df_continuous.ffill().bfill()

        # Make the data in each energy type a numpy array to scale the data
        raw_matrix = self.df[self.feature_cols].values

        # Scales the data to prevent large numbers from returning flat output values
        # For example, 58 MWh and 720 MWh return 1 due to the tanh activation, causing overfitting
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.scaled_data = self.scaler.fit_transform(raw_matrix)


    def __len__(self):
        # Ensures that we do not look ahead past the most recent data
        return len(self.scaled_data) - self.lookback_steps

    def __getitem__(self, idx):
        # Ensures the window of the memory cell is 168 hours(7 days) for all
        x_sequence = self.scaled_data[idx : idx + self.lookback_steps]
        y_target = self.scaled_data[idx + self.lookback_steps]

        # Creates a torch readable dataset to train the network successfully
        return (torch.tensor(x_sequence, dtype=torch.float32),
                torch.tensor(y_target, dtype=torch.float32))