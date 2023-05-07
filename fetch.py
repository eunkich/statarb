import os
import time

import requests
import polygon
import sqlalchemy
import pandas as pd

from datetime import datetime, timedelta
from typing import Iterable
from pathlib import Path
from io import StringIO

from dotenv import load_dotenv
from polygon import RESTClient

# Does not override system environment variable.
# POLYGON_API_KEY should be stored in .env or as system environment variable.
load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
MAX_REQUEST = 50000
MAX_PAST = 2 * timedelta(days=365)


def get_largest_US_stock_tickers(num_assets: int = 100,
                                 save_summary: bool = False):
    # Get tickers of the largest companies by market cap. in the US
    url = "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/?download=csv"
    r = requests.get(url)

    summary = pd.read_csv(StringIO(r.text))[:num_assets]
    today = datetime.now().strftime("%Y%m%d")
    if save_summary:
        summary.to_csv(f"US_Stock_MarketCap_summary_{today}")
    tickers = summary[:num_assets].Symbol

    return tickers


def fetch(tickers: Iterable, date_from: datetime, date_to: datetime, timespan: str,
          adjusted: bool = True, save_dir="data/db", verbose: bool = True):
    """
    Fetch historical data from https://polygon.io
    POLYGON_API_KEY should be provided in constants.py

    tickers: list of tickers to fetch data
    date_*: datetime objects for time window
    adjusted: Whether or not the results are adjusted for splits
    timespan: resolution of time series; {"minute", "hour", "day"}
    """
    assert timespan in {"minute", "hour", "day"}, \
        "Invalid resolution, timespan should be one of the following: {'minute', 'hour', 'day'}"

    denom = {"minute": 60,
             "hour": 60 * 60,
             "day": 60 * 60 * 12}

    # Handle Path to db
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    engine = sqlalchemy.create_engine(f"sqlite:///{path}/freq-{timespan}.db")

    curr_tos = [date_from for _ in range(len(tickers))]
    # Retrive last checkpoint
    for i, ticker in enumerate(tickers):
        try:
            last_stmp = pd.read_sql_query(
                f'SELECT MAX(timestamp) FROM {ticker}', engine).values.item()
            last_to = datetime.fromtimestamp(last_stmp/1000.0)
            if verbose:
                print(
                    f"Found existing table for {ticker} in the database. Resume fetching from the last point {last_to}")
            curr_tos[i] = max(last_to, curr_tos[i])

        except sqlalchemy.exc.OperationalError:
            pass

    client = RESTClient(POLYGON_API_KEY)
    partition = timedelta(seconds=1) * denom[timespan] * MAX_REQUEST

    for i, ticker in enumerate(tickers):
        curr_to = curr_tos[i]
        while curr_to < date_to:
            curr_from = curr_to
            curr_to = min(date_to, curr_to + partition)
            if verbose:
                print(f"Fetching {ticker} from {curr_from} to {curr_to}")
            while True:
                try:
                    out = client.get_aggs(ticker=ticker,
                                          multiplier=1,
                                          timespan=timespan,
                                          from_=curr_from,
                                          to=curr_to,
                                          adjusted=adjusted,
                                          limit=MAX_REQUEST,
                                          )
                    break
                except polygon.exceptions.BadResponse:
                    time.sleep(20)
                    continue

            pd.DataFrame(out).drop_duplicates().to_sql(name=ticker,
                                                       con=engine,
                                                       if_exists="append",
                                                       index=False)

        # Remove Duplicates
        with engine.connect() as conn:
            query = (
                f"CREATE TABLE temp_table as SELECT DISTINCT * FROM {ticker}",
                f"DROP TABLE {ticker}",
                f"ALTER TABLE temp_table RENAME TO {ticker}"
            )
            for q in query:
                conn.execute(sqlalchemy.text(q))
        if verbose:
            print(f"Fetch complete for {ticker} from {date_from} to {date_to}")


if __name__ == "__main__":
    tickers = get_largest_US_stock_tickers(100, save_summary=False)
    print(tickers)

    today = datetime.today().replace(microsecond=0, second=0)
    today -= timedelta(days=1)
    fetch(tickers,
          date_from=today - MAX_PAST,
          date_to=today,
          timespan="minute")
