import asyncio
from services.advisor import top_coin
from services.forecast import predict_price

async def main():
    print("Top-10 Coins Today:")
    ranking = await top_coin()
    for rank in ranking:
        print(f"   {rank['coin']:<10} {rank['decision'].upper():<5} (RSI={rank['rsi']:.1f}, Sentiment={rank['sent']:.2f})")

    print("\n" + "="*40)
    print("Forecasting Example:")
    try:
        price, lo, hi = await predict_price("solana", "usd", "2025-12-25")
        print(f"   Predicted SOL price on 25-Dec-2025: ${price:,.2f} (Range: ${lo:,.0f} - ${hi:,.0f})")
    except Exception as e:
        print(f"   Could not fetch forecast: {e}")

if __name__ == "__main__":
    asyncio.run(main())

