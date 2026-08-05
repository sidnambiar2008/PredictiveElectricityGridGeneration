import pandas as pd
import src.ingestion.collect_history as collect_history


def test_main_appends_across_runs_instead_of_overwriting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Simulated "week 1" fetch: 3 hourly rows, 00:00 through 02:00
    week1 = pd.DataFrame(
        {"Coal": [1, 2, 3], "Geothermal": 0, "Hydro": 0, "Natural Gas": 0,
         "Nuclear": 0, "Petroleum": 0, "Wind": 0, "Solar": 0, "Other": 0},
        index=pd.date_range("2024-01-01 00:00", periods=3, freq="h"),
    )

    monkeypatch.setattr(collect_history, "fetch_historical_slice", lambda **kwargs: week1)
    collect_history.main()

    # Simulated "week 2" fetch: 3 hourly rows, 02:00 through 04:00 — note 02:00
    # overlaps with week1, and this time Coal at 02:00 is a different value (99)
    week2 = pd.DataFrame(
        {"Coal": [99, 4, 5], "Geothermal": 0, "Hydro": 0, "Natural Gas": 0,
         "Nuclear": 0, "Petroleum": 0, "Wind": 0, "Solar": 0, "Other": 0},
        index=pd.date_range("2024-01-01 02:00", periods=3, freq="h"),
    )

    monkeypatch.setattr(collect_history, "fetch_historical_slice", lambda **kwargs: week2)
    collect_history.main()

    result = pd.read_csv(tmp_path / "grid_data/raw/grid_history_ciso_v4.csv",
                          index_col="period", parse_dates=True)

    assert len(result) == 5
    assert result.loc["2024-01-01 02:00:00", "Coal"] == 99
    assert result.loc["2024-01-01 00:00:00", "Coal"] == 1
    