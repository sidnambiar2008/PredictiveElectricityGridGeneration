import os
import requests
import pandas as pd

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
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region_id,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000
    }

    print(f"Connecting to EIA API .. Fetching past {days_back} days for region: {region_id}")
    response = requests.get(url, params= params)

    if (response.status_code != 200):
        raise RuntimeError(f"Error fetching data from EIA API. Status code: {response.status_code}")

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