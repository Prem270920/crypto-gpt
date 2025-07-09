import httpx
import pandas as pd
from transformers import pipeline

# Using Hugging Face Transformers, loading sentiment-analysis pipeline 
try:
    sentiment_pipline = pipeline("sentiment-analysis")
except:
     print(f"Warning: Could not load sentiment-analysis pipeline: {e}")
     _sentiment_pipeline = None

async def coin_sentiment(coin_name):

    """Return compound sentiment (-1…+1) from last 10 NewsAPI headlines using ML."""

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
         response = await client.get(url, params=params)
    
    if response.status_code != 200:
         return 0.0
    
    articles = response.json().get("articles", [])
    headlines = [art["title"]for art in articles if "title" in art and art["title"]]

    if not headlines:
        return 0.0
    
    results = sentiment_pipline(headlines)

    #mapping: POSITIVE -> +score, NEGATIVE -> -score, NEUTRAL -> 0
    compound_scores = []
    for result in results:
        if result['label'] == "POSITIVE":
            compound_scores.append(result['score'])
        elif result['label'] == 'NEGATIVE':
            compound_scores.append(-res['score'])
        else: # Assuming 'NEUTRAL' or other categories are 0
            compound_scores.append(0.0)

    return sum(compound_scores) / len(compound_scores) if compound_scores else 0.0

# --- Quick CLI Sanity Check

if __name__ == "__main__":

    import asyncio
    async def _demo():
        sentiment = await coin_sentiment("solana")
        print(f"Sentiment for Solana headlines: {sentiment:.2f}")
        sentiment_bitcoin = await coin_sentiment("bitcoin")
        print(f"Sentiment for Bitcoin headlines: {sentiment_bitcoin:.2f}")

    asyncio.run(_demo())