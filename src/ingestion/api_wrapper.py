import os
import requests
import pandas as pd
import time

def fetch_latest_eia_data(region_id = "CISO", days_back = 14):
    """
       Fetches the latest data for each fuel in the electricity grid
       to ensure the model has 7 days of historical data and 7 days to evaluate
       the predictions

       This function takes region_id and number of days back. Setting this allows for
       flexibility with the historical timeframe and location

       Args:
           region_id (string): The region code that is current being fetched
           days_back (int): The number of days back to allow

       Returns:
           dataframe: Historical dat

       Raises:
           ValueError: If no data is fetched, indicating a wrong path.
           RuntimeError: If the fetch is unsuccessful, indicating a request error.
       """

    api_key = os.getenv("EIA_API_KEY")

    if not api_key:
        raise ValueError("Missing EIA_API_KEY environment variable. Please export it in your terminal.")

    BASE_URL = "https://api.eia.gov"
    url = f"{BASE_URL}/v2/electricity/rto/fuel-type-data/data/"

    # Calculate true rolling boundaries based on the current calendar day
    end_date: pd.Timestamp = pd.Timestamp.now()
    start_date: pd.Timestamp = pd.Timestamp(end_date - pd.Timedelta(days=days_back))

    # Format the timestamps to match the EIA API expectations (YYYY-MM-DDTHH)
    start_str = start_date.strftime("%Y-%m-%dT%H")
    end_str = end_date.strftime("%Y-%m-%dT%H")

    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region_id,
        "start": start_str,
        "end": end_str,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000
    }

    max_retries = 3
    retry_delay = 5  # Seconds to wait

    print(f"Connecting to EIA API .. Fetching past {days_back} days for region: {region_id}")

    for attempt in range(max_retries):
        response = requests.get(url, params= params)

        if (response.status_code == 200):
            break

        elif attempt < max_retries - 1:
            print(print(f" EIA Server timeout (Status: {response.status_code}). Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"))
        else:
            raise Exception(f"Failed to connect to EIA API after {max_retries} attempts. Last status code: {response.status_code}")


    raw_json = response.json()

    data_records = raw_json.get("response",{}).get("data", [])

    if not data_records:
        raise ValueError(f"No grid metrics returned by the API for region {region_id}. Check parameters.")

    df = pd.DataFrame(data_records)

    # Converts the data type of to pandas datetime to make it easier to sort by period (an index)
    df["period"] = pd.to_datetime(df["period"])

    # Summarizes the data in the dataframe to make the distinction between fuel type by hour easier
    clean_matrix = df.pivot_table(
        values="value",
        index="period",
        columns="type-name",
        aggfunc="last"
    )

    clean_matrix = clean_matrix.sort_index().astype(float)

    return clean_matrix