import os
import pandas as pd
import requests

def fetch_historical_slice(start_date, end_date, region_id: str = "CISO"):
        api_key = os.getenv("EIA_API_KEY")

        if not api_key:
            raise ValueError("Missing EIA_API_KEY environment variable. Please export it in your terminal.")

        BASE_URL = "https://api.eia.gov"
        url = f"{BASE_URL}/v2/electricity/rto/fuel-type-data/data/"
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "start": start_date,
            "end": end_date,
            "data[0]": "value",
            "facets[respondent][]": region_id,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
            "offset": 0
        }
        all_records = []
        total_records = None

        while True:
            request = requests.get(url, params=params)
            if request.status_code != 200:
                print(f"Error fetching data from EIA API. Status code: {request.status_code}")
                break

            request_data = request.json()
            data_records = request_data.get("response", {}).get("data", [])
            total_records = int(request_data["response"]["total"])

            all_records.extend(data_records)

            if not data_records:
                print(f"No data available for this balancing authority, {region_id}")
                break

            params["offset"] += params["length"]
            if params["offset"] >= total_records:
                break

        df = pd.DataFrame(all_records)
        df["period"] = pd.to_datetime(df["period"])

        df = df.pivot_table(index = ["period"], columns = ["type-name"], values = "value", aggfunc = "first")
        df = df.sort_index()
        return df


def main():
    os.makedirs("grid_data/raw", exist_ok=True)

    clean_history_df = fetch_historical_slice(
        start_date="2023-01-01",
        end_date="2026-01-01",
        region_id="CISO"
    )
    clean_history_df.index.name = "period"
    clean_history_df.to_csv("grid_data/raw/grid_history.csv", index=True, mode="w")


if __name__ == "__main__":
    main()
