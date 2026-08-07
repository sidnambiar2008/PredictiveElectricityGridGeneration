import pandas as pd
import requests
import src.ingestion.collect_history as collect_history
from src.ingestion.collect_history import fetch_historical_slice


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

def test_geothermal_merge_does_not_wipe_others(monkeypatch):
    fake_records = [
        # An early hour: only "Other" was reported, no Geothermal category existed yet
        {"period": "2024-01-01T00", "type-name": "Other", "value": "100"},
        # A later hour: both categories now exist
        {"period": "2024-01-01T01", "type-name": "Other", "value": "50"},
        {"period": "2024-01-01T01", "type-name": "Geothermal", "value": "10"},
    ]

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"response": {"data": fake_records, "total": len(fake_records)}}

    monkeypatch.setattr(requests, "get", lambda url, params: FakeResponse())
    monkeypatch.setenv("EIA_API_KEY", "fake-key-for-test")

    result = fetch_historical_slice(start_date="2024-01-01T00", end_date="2024-01-01T01", region_id="CISO")

    assert (result.loc["2024-01-01 00:00:00", "Other"] == 100)
    assert (result.loc["2024-01-01 01:00:00", "Geothermal"] == 0)


def test_successful_response_does_not_retry(monkeypatch):
    # Guards against a real bug: the retry loop's success condition was once
    # inverted (checking status_code != 200 instead of == 200), which caused
    # even a genuinely successful first response to loop unnecessarily.
    call_count = {"n": 0}
    fake_records = [
        {"period": "2024-01-01T00", "type-name": "Other", "value": "100"},
    ]

    class FakeResponse:
        status_code = 200
        def json(self):
            call_count["n"] += 1
            return {"response": {"data": fake_records, "total": len(fake_records)}}

    monkeypatch.setattr(requests, "get", lambda url, params: FakeResponse())
    monkeypatch.setenv("EIA_API_KEY", "fake-key-for-test")

    fetch_historical_slice(start_date="2024-01-01T00", end_date="2024-01-01T01", region_id="CISO")

    assert call_count["n"] == 1


def test_recovers_from_network_error(monkeypatch):
    # Distinct from the Geothermal test above: this checks recovery from the
    # request itself failing to complete (e.g. a dropped connection), not a
    # bad HTTP status. The retry loop originally only handled the latter.
    call_count = {"n": 0}
    fake_records = [
        {"period": "2024-01-01T00", "type-name": "Other", "value": "100"},
    ]

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"response": {"data": fake_records, "total": len(fake_records)}}

    def flaky_get(url, params=None):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise requests.exceptions.ConnectionError("simulated network blip")
        return FakeResponse()

    monkeypatch.setattr(requests, "get", flaky_get)
    monkeypatch.setenv("EIA_API_KEY", "fake-key-for-test")

    result = fetch_historical_slice(start_date="2024-01-01T00", end_date="2024-01-01T01", region_id="CISO")

    assert call_count["n"] == 2
    assert result.loc["2024-01-01 00:00:00", "Other"] == 100.0
