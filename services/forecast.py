from __future__ import annotations
import asyncio
import pandas as pd
from datetime import datetime
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
    cg = CoinGeckoClient(api_key="CG-zZrSVxFuf91Yv3JH4t4KjKcy")

    try:
        raw_data = await cg.market_chart(coin_id, vs_currency, days)
    finally:
        await cg.close()

    if not raw_data:
        return pd.DataFrame() # Return an empty DataFrame if no data

    # Corrected 'pd.dataFrame' to 'pd.DataFrame'
    df = pd.DataFrame(raw_data, columns=["ts", "price"])
    df["ds"] = pd.to_datetime(df["ts"], unit="ms")
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
    # Fetch historical data for the past 1 years (365 days) to train the model

    df = await fetch_history(coin_id, vs_currency, days=365)

    if df.empty:
        print(f"Warning: No historical data received for {coin_id}. Cannot forecast.")
        return None

    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
    model.fit(df)

    future_date = pd.DataFrame({"ds": [pd.to_datetime(target_date)]})
    forecast = model.predict(future_date).iloc[0]
    return float(forecast.yhat), float(forecast.yhat_lower), float(forecast.yhat_upper)

if __name__ == "__main__":
    async def _demo():
        print("Testing forecast for Solana...")
        # Note: The date here is in the past for this example to work with current data.
        # Change "2024-07-15" to a future date for a real prediction.
        prediction = await predict_price("solana", "usd", "2024-07-15")
        if prediction:
            price, lo, hi = prediction
            print(f"SOL on 15-Jul-2024: ${price:,.2f} (range {lo:,.0f} - {hi:,.0f})")
        else:
            print("Prediction failed.")

    asyncio.run(_demo())