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

def test_successful_response_does_not_retry(monkeypatch):
    # Guards against a real bug: the retry loop's success condition was once
    # inverted (checking status_code != 200 instead of == 200), which caused
    # even a genuinely successful first response to loop unnecessarily.
    call_count = {"n": 0}

    class FakeResponse:
        status_code = 200
        def json(self):
            call_count["n"] += 1
            return {"response": {"data": [
                {"period": "2024-01-01T00", "type-name": "Other", "value": "100"},
            ]}}

    monkeypatch.setattr(requests, "get", lambda url, params: FakeResponse())
    monkeypatch.setenv("EIA_API_KEY", "fake-key-for-test")

    fetch_latest_eia_data(region_id="CISO", days_back=1)

    assert call_count["n"] == 1

def test_recovers_from_network_error(monkeypatch):
    # Distinct from the Geothermal test above: this checks recovery from the
    # request itself failing to complete (e.g. a dropped connection), not a
    # bad HTTP status. The retry loop originally only handled the latter.
    call_count = {"n": 0}

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"response": {"data": [
                {"period": "2024-01-01T00", "type-name": "Other", "value": "100"},
            ]}}

    def flaky_get(url, params=None):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise requests.exceptions.ConnectionError("simulated network blip")
        return FakeResponse()

    monkeypatch.setattr(requests, "get", flaky_get)
    monkeypatch.setenv("EIA_API_KEY", "fake-key-for-test")

    result = fetch_latest_eia_data(region_id="CISO", days_back=1)

    assert call_count["n"] == 2
    assert result.loc["2024-01-01 00:00:00", "Other"] == 100.0