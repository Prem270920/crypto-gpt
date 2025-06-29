"""
services.advisor

Combine sentiment + RSI to rank coins each day.
"""

import asyncio
from datetime import date
from typing import List, Dict
from services.indicators import rsi
from services.news import coin_sentiment

COIN_LIST = ["bitcoin", "ethereum", "solana", "bnb", "xrp",
             "dogecoin", "cardano", "tron", "polkadot", "chainlink",
             "sui"]

async def coin_score(coin):
    rsi_value = await rsi(coin)
    senti_value = await coin_sentiment(coin)
    # very naive scoring: lower RSI (oversold) and positive sentiment is good
    score = (70 - rsi_value) + (senti_value * 40)
    decision = (
        "buy" if score > 30 else
        "hold" if score > 0 else
        "sell"
    )
    return {"coin": coin, "score": score, "rsi": rsi_value, "sent": senti_value, "decision": decision}

async def top_coin():
    tasks = [asyncio.create_task(coin_score(c) for c in COIN_LIST)]
    ranked = sorted([await t for t in tasks], key = lambda d: d["score"], reverse=True)
    return ranked 


