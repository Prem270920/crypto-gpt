"""
services.advisor

Combine sentiment + RSI to rank coins each day.
"""

import asyncio
from datetime import date
from typing import List, Dict
from services.indicators import rsi
from services.news import coin_sentiment

COIN_LIST = ["bitcoin", "ethereum", "solana", "binancecoin", "ripple",
             "dogecoin", "cardano", "tron", "polkadot", "chainlink",
             "sui"]

async def coin_score(coin):
    try:
        rsi_value, senti_value = await asyncio.gather(
            rsi(coin, vs_currency="usd", period=14),
            coin_sentiment(coin)
        )
        # very naive scoring: lower RSI (oversold) and positive sentiment is good
        score = (70 - rsi_value) + (senti_value * 40)
        decision = (
            "buy" if score > 30 else
            "hold" if score > 0 else
            "sell"
        )
        return {"coin": coin, "score": score, "rsi": rsi_value, "sent": senti_value, "decision": decision}
    except Exception as e:
        print(f"Could not process {coin}: {e}")
        return None

async def top_coin():
    
    results = []
    for coin in COIN_LIST:
        score_data = await coin_score(coin)
        results.append(score_data)
        # Wait for 2 seconds before the next request to avoid being rate-limited
        await asyncio.sleep(2)

    ranked = sorted(results, key=lambda d: d['score'], reverse=True)
    return ranked 


