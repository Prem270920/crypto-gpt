from __future__ import annotations
import _asyncio
import pandas as pd
from datetime import datetime
from typing import Tuple
from prophet import Prophet #forecasting lib from facebook
from services.coingecko import CoinGeckoClient

async def fetch_history(coin_id, vs_currency, days):
    """
    Asynchronously fetches historical price data for a given cryptocurrency from CoinGecko.

    Args:
        coin_id (str): The ID of the cryptocurrency (e.g., "solana").
        vs_currency (str): The currency to compare against (e.g., "usd").
        days (int): The number of past days for which to fetch data. Defaults to 365.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'ds': timestamp (ISO date format, suitable for Prophet)
            - 'y': price (the target variable for Prophet)
    """
    cg = CoinGeckoClient(api_key="CG-vNmyQcAqALXnY8XB677BVA72")

    try:
        raw_data = await cg.market_chart(coin_id, vs_currency, days)
    finally:
        await cg.close()

    df = pd.dataFrame(raw_data, columns = ["ts", "price"])
    # Convert Unix timestamps (milliseconds) to datetime objects
    df["ds"] = pd.to_datetime(df["ts"], unit="ms")
    # Rename the 'price' column to 'y' as required by Prophet
    df["y"] = df["price"]

    return df[["ds", "y"]]

async def predict_price(coin_id: str, vs_currency: str, target_date: str):
    """
    Asynchronously forecasts the price of a cryptocurrency on a specific target date.

    Args:
        coin_id (str): The ID of the cryptocurrency (e.g., "solana").
        vs_currency (str): The currency to compare against (e.g., "usd").
        target_date (str): The date for which to predict the price, in 'YYYY-MM-DD' format.

    Returns:
        Tuple[float, float, float]: A tuple containing the predicted price (yhat),
                                     the lower bound of the prediction interval (yhat_lower),
                                     and the upper bound of the prediction interval (yhat_upper).
    """
    # Fetch historical data for the past 2 years (365 * 2 days) to train the model

    df = await fetch_history(coin_id, vs_currency, days=365 * 2)

    model = Prophet(daily_seasonality=False, yearly_seasonality=True)

    model.fit(df)

    future_date = pd.DataFrame({"ds": [pd.to_datetime(target_date)]})

    forecast = model.predict(future_date).iloc[0]

    return float(forecast.yhat), float(forecast.yhat_lower), float(forecast.yhat_upper)

if __name__ == "__main__":
    async def beta():
        price, lo, hi = await predict_price("solana", "usd", "2024-07-15")

        print(f"SOL on 15-Jul-2024: ${price:,.2f}  (range {lo:,.0f} - {hi:,.0f})")

    _asyncio.run(beta())