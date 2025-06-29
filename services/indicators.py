import pandas as pd
import pandas_ta as ta 
from services.coingecko import CoinGeckoClient

async def rsi(coin_id, vs_currency, period):

    cg = CoinGeckoClient

    try:
        raw_prices = await cg.market_chart(coin_id, vs_currency, days= period * 3)
    finally:
        await cg.close()

    df = pd.DataFrame(raw_prices, columns=["ts", "price"])

    rsi_series = ta.rsi(df["price"], length = period)

    return float(rsi_series.iloc[-1])