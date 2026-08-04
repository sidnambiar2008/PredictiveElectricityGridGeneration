import pandas as pd
from src.models.baseline import DiurnalRollingMeanBaseline

def test_baseline_predicts_hour_of_day_average():
    dates = pd.date_range("2024-01-01", periods = 24 * 10, freq = "h")
    values = [ts.hour for ts in dates]
    df = pd.DataFrame({"Solar": values}, index = dates)

    day_index = (df.index - df.index[0]).days
    df["Solar"] = df.index.hour + day_index * 100

    baseline = DiurnalRollingMeanBaseline(window_days = 7)
    forecast = baseline.predict(df)

    assert (forecast["Solar"].iloc[0] == forecast.index[0].hour + 600)
