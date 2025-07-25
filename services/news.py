import httpx
import pandas as pd
import joblib
from pathlib import Path

model_path = Path("models/traditional_sentiment_model.pkl")


try:
    sentiment_pipline = joblib.load(model_path)
except FileNotFoundError:
    print(f"Warning: Model not found at {model_path}. Cannot perform sentiment analysis.")
    sentiment_pipline = None

async def coin_sentiment(coin_name):

    """Return compound sentiment (-1, +1) from last 10 NewsAPI headlines using ML."""

    if sentiment_pipline is None:
          print("Sentiment pipeline not loaded. Returning neutral sentiment.")
          return 0.0
     
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": coin_name,
        "apiKey": "ac300b17ba474705b0fc958e6b8fd212",
        "pageSize": 10,
        "sortBy": "publishedAt",
        "language": "en",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(url, params=params)
        except httpx.HTTPStatusError as e:
            print(f"NewsAPI error for {coin_name}: {e}")
            return 0.0
    
    articles = response.json().get("articles", [])
    headlines = [art["title"]for art in articles if "title" in art and art["title"]]

    if not headlines:
        return 0.0
    
    predictions = sentiment_pipline.predict(headlines)

    #mapping: POSITIVE -> +score, NEGATIVE -> -score, NEUTRAL -> 0
    score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = [score_map.get(pred, 0.0) for pred in predictions]

    return sum(scores) / len(scores) if scores else 0.0

# --- Quick CLI Sanity Check

if __name__ == "__main__":

    import asyncio
    async def _demo():
        sentiment = await coin_sentiment("solana")
        print(f"Sentiment for Solana headlines: {sentiment:.2f}")
        sentiment_bitcoin = await coin_sentiment("bitcoin")
        print(f"Sentiment for Bitcoin headlines: {sentiment_bitcoin:.2f}")

    asyncio.run(_demo())