import os
import pandas as pd
import requests

def fetch_historical_slice(start_date, end_date, region_id: str = "CISO"):
        api_key = os.getenv("EIA_API_KEY")

        if not api_key:
            raise ValueError("Missing EIA_API_KEY environment variable. Please export it in your terminal.")

        print("Connection to the EIA API...")

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
    os.makedirs("data/raw", exist_ok=True)
    save_path = "../../grid_data/raw/grid_history.csv"

    clean_history_df = fetch_historical_slice(
        start_date="2023-01-01",
        end_date="2026-01-01",
        region_id="CISO"
    )

    # Diagnostic Telemetry Checkpoints:
    print("\n Checking dataframe dimensions before saving:")
    print(f"   -> Dataframe Row Count: {len(clean_history_df)}")
    print(f"   -> Columns Extracted:   {list(clean_history_df.columns)}")

    if clean_history_df.empty:
        print(" WARNING: The dataframe is empty! The pivot key might be mismatched.")
    else:
        # 2. Explicitly name the index for your data loader tracking constraints
        clean_history_df.index.name = "period"

        # 3. Save the file cleanly
        clean_history_df.to_csv(save_path, index=True, mode="w")
        print(f"\n SUCCESS! 3-Year baseline successfully saved at: {save_path}")

    clean_history_df.index.name = "period"
    clean_history_df.to_csv(save_path, index=True, mode="w")


if __name__ == "__main__":
    main()
