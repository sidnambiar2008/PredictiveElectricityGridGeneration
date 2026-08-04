import requests
from src.ingestion.api_wrapper import fetch_latest_eia_data

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

    result = fetch_latest_eia_data(region_id = "CISO", days_back=1)

    assert (result.loc["2024-01-01 00:00:00", "Other"] == 100)
    assert (result.loc["2024-01-01 01:00:00", "Geothermal"] == 0)

