import os
import pandas as pd
import requests
from datetime import datetime, timedelta


EXPECTED_COLS = ["Coal", "Geothermal", "Hydro", "Natural Gas", "Nuclear", "Petroleum", "Wind", "Solar", "Other"]


def fetch_historical_slice(start_date, end_date, region_id: str = "CISO"):
    """
    Fetch a specific region of the EIA data to add to the region's historical csv

    Args:
        start_date (datetime): Start date of the historical data
        end_date (datetime): End date of the historical data
        region_id (string): EIA region code

    Returns:
        dataframe: Sliver of the EIA data
    """


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

    # The EIA API returns "value" as a string, so pivoted columns come out as
    # object dtype rather than numeric. That was harmless before (nothing did
    # arithmetic on them in memory — writing to CSV and reading it back with
    # pd.read_csv silently re-inferred the correct numeric dtype), but the
    # Geothermal/Other merge below does real arithmetic, so it needs to happen
    # on actual floats.
    df = df.astype(float)

    print("\nMissing values before filling:")
    print(df.isna().sum()[df.isna().sum() > 0])

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0

    df = df[EXPECTED_COLS]
    df = df.fillna(0)

    # EIA started reporting this region's geothermal generation as its own
    # category partway through the collection window (it was previously
    # folded into "Other"). Fold it back in so the feature is consistent
    # across the whole history instead of jumping from a constant zero to
    # a real value partway through. This must run after fillna(0): pivot_table
    # leaves timestamps with no Geothermal record at all as NaN (not 0), and
    # NaN + real_value = NaN, which would silently wipe out genuine Other
    # data for every row before Geothermal existed as a reported category.
    df["Other"] = df["Other"] + df["Geothermal"]
    df["Geothermal"] = 0.0

    print("\nFinal missing values:")
    print(df[EXPECTED_COLS].isna().sum())

    df = df.sort_index()
    return df


def main():
    end_date_dt = datetime.utcnow()
    start_date_dt = end_date_dt - timedelta(days=7)

    os.makedirs("grid_data/raw", exist_ok=True)
    regions = ["CISO", "PJM", "SWPP", "ERCO", "MISO", "ISNE", "NYIS"]

    for region in regions:
        save_path = f"grid_data/raw/grid_history_{region.lower()}_v4.csv"

        clean_history_df = fetch_historical_slice(
            start_date=start_date_dt.strftime("%Y-%m-%dT%H"),
            end_date=end_date_dt.strftime("%Y-%m-%dT%H"),
            region_id=region
        )

        print("\n Checking dataframe dimensions before saving:")
        print("Expected hourly rows:", 7 * 24)
        print(f"   -> Dataframe Row Count: {len(clean_history_df)}")
        print(f"   -> Columns Extracted:   {list(clean_history_df.columns)}")

        if clean_history_df.empty:
            print(" WARNING: The dataframe is empty! The pivot key might be mismatched.")
        else:
            # Must match the index_col="period" read below, and what
            # GridDataLoader expects when it reads this CSV back in.
            clean_history_df.index.name = "period"

            if os.path.exists(save_path):
                real_df = pd.read_csv(save_path, index_col="period", parse_dates=True)
                updated_df = pd.concat([real_df, clean_history_df], axis=0)
            else:
                updated_df = clean_history_df

            # Keep the newer of any overlapping hours instead of appending
            # duplicates, since this week's slice always overlaps the tail
            # of what's already on disk.
            updated_df = updated_df[~updated_df.index.duplicated(keep="last")]
            updated_df = updated_df.sort_index()

            # mode="w" is safe here: updated_df already contains the full
            # merged history (existing + new), not just this week's slice.
            updated_df.to_csv(save_path, index=True, mode="w")


if __name__ == "__main__":
    main()
