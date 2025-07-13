import pandas as pd
import pandas_ta as ta
from services.coingecko import CoinGeckoClient

async def rsi(coin_id: str, vs_currency: str = "usd", period: int = 14) -> float:
    # 1. Create an INSTANCE of the client (the "house")
    cg = CoinGeckoClient(api_key="ac300b17ba474705b0fc958e6b8fd212") 
    
    try:
        # 2. Call methods on the instance variable 'cg'
        prices = await cg.market_chart(coin_id, vs_currency, days=period * 3)
    finally:
        # 3. Call close on the instance variable 'cg'
        await cg.close()

    if not prices:
        return 50.0 # Return a neutral RSI if no price data is found

    df = pd.DataFrame(prices, columns=["ts", "price"])
    rsi_series = ta.rsi(df["price"], length=period)
    
    # Check if the RSI calculation was successful
    if rsi_series is None or rsi_series.empty:
        return 50.0 # Return neutral if RSI calculation fails

    # Return the last RSI value in the series
    return float(rsi_series.iloc[-1])