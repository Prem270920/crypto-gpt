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
    """
    Gets scores for all coins using a semaphore to limit concurrency.
    """
    # Create a semaphore that allows only 2 tasks to run at once
    sem = asyncio.Semaphore(2)

    async def get_score_with_semaphore(coin: str):
        async with sem:
            return await coin_score(coin)

    tasks = [get_score_with_semaphore(c) for c in COIN_LIST]
    results = await asyncio.gather(*tasks)

    # Filter out any coins that failed
    valid_results = [r for r in results if r is not None]
    
    ranked = sorted(valid_results, key=lambda d: d['score'], reverse=True)
    return ranked