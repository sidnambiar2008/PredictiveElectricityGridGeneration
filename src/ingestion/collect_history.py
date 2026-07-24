import os
import pandas as pd
import requests

EXPECTED_COLS = ["Coal", "Geothermal", "Hydro", "Natural Gas", "Nuclear", "Petroleum", "Wind", "Solar", "Other"]


def fetch_historical_slice(start_date, end_date, region_id: str = "CISO"):
        api_key = os.getenv("EIA_API_KEY")

        if not api_key:
            raise ValueError("Missing EIA_API_KEY environment variable. Please export it in your terminal.")

        print(f"Connecting to the EIA API for {region_id}")

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
                raise RuntimeError(
                    f"EIA request failed for {region_id}: {request.status_code}"
                )

            request_data = request.json()

            data_records = request_data.get("response", {}).get("data", [])
            total_records = int(request_data.get("response", {}).get("total", 0))

            all_records.extend(data_records)

            if not data_records:
                print(f"No data available for this balancing authority, {region_id}")
                break

            params["offset"] += params["length"]
            if params["offset"] >= total_records:
                break

        df = pd.DataFrame(all_records)

        if df.empty:
            raise ValueError(f"No data returned for {region_id}")

        df["period"] = pd.to_datetime(df["period"])

        df = df.pivot_table(index = "period", columns = "type-name", values = "value", aggfunc = "first")

        print("\nMissing values before filling:")
        print(df.isna().sum()[df.isna().sum() > 0])

        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = 0



        df = df[EXPECTED_COLS]
        df = df.fillna(0)

        print("\nFinal missing values:")
        print(df[EXPECTED_COLS].isna().sum())

        df = df.sort_index()
        return df


def main():
    os.makedirs("../../grid_data/raw", exist_ok=True)
    regions = ["CISO", "PJM", "SWPP", "ERCO", "MISO", "ISNE", "NYIS"]
    for region in regions:
        save_path = f"../../grid_data/raw/grid_history_{region.lower()}_v3.csv"

        clean_history_df = fetch_historical_slice(
            start_date="2023-01-01",
            end_date="2026-01-01",
            region_id=region
        )

        # Diagnostic Telemetry Checkpoints:
        print("\n Checking dataframe dimensions before saving:")
        print("Expected hourly rows:", 3 * 365 * 24)
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

if __name__ == "__main__":
    main()
