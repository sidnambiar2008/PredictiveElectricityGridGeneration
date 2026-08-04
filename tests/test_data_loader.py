import pandas as pd
from src.preprocessing.data_loader import GridDataLoader

def test_data_loader(tmp_path):
    dates = pd.date_range(start = "2024-1-1", periods = 20, freq = "h")
    df = pd.DataFrame({"Coal": range(20), "Geothermal": 0, "Hydro": 0, "Natural Gas": 0,
                       "Nuclear" : 0, "Petroleum":0, "Solar": 0, "Wind":0, "Other":0}, index = dates)

    df.index.name = "Period"
    csv_path = tmp_path / "fake_history.csv"
    df.to_csv(csv_path)

    dataset = GridDataLoader(str(csv_path))

    assert (dataset.scaler.data_max_[0] == 16)