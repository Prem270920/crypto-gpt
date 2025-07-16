import httpx

url = "https://api.coingecko.com/api/v3"

class CoinGeckoClient:
    "An asynchronous client for interacting with the CoinGecko API."

    def __init__(self, api_key: str | None = None):
        headers = {"x-cg-demo-api-key": api_key} if api_key else {} 
        # Create an async client that will be used for all requests
        self._client = httpx.AsyncClient(base_url=url, headers=headers, timeout=15)

    async def get_price(self, coin_ids, vs_currencies):
        '''
        Fetch the current price of one or more coins in one or more currencies.
        '''
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": ",".join(vs_currencies),
        }
        resp = await self._client.get("/simple/price", params=params)
        resp.raise_for_status()  
        return resp.json()
    
    async def market_chart(self, coin_id: str, vs_currency: str, days: int):
        """
        Fetch historical market data for a specific coin.
        Returns a list of [timestamp, price] pairs.
        """
        params = {
            "vs_currency": vs_currency, 
            "days": str(days), 
            "interval": "daily"
        }
        resp = await self._client.get(f"/coins/{coin_id}/market_chart", params=params)
        resp.raise_for_status()

        return resp.json().get("prices", [])

    async def get_top_market_cap_coins(self, count: int = 20):
        """
        Fetches the top coins by market capitalization.
        """
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": count,
            "page": 1,
            "sparkline": "false"
        }

        resp = await self._client.get("/coins/markets", params=params)
        resp.raise_for_status()

        return [coin["id"] for coin in resp.json()]
    
    async def close(self):
        """
        Closes the underlying HTTP client. Should be called when done.
        """
        await self._client.aclose()

    