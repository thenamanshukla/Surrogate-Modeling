import os
import time
import json
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

ASSETS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY")


def _cg_headers():
    if COINGECKO_API_KEY:
        return {"x-cg-demo-api-key": COINGECKO_API_KEY}
    return {}


def fetch_market_chart(coin_id, days=365, vs_currency="usd"):
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    r = requests.get(url, params=params, headers=_cg_headers(), timeout=30)
    r.raise_for_status()
    payload = r.json()
    prices = pd.DataFrame(payload["prices"], columns=["ts", "price"])
    caps = pd.DataFrame(payload["market_caps"], columns=["ts", "market_cap"])
    vols = pd.DataFrame(payload["total_volumes"], columns=["ts", "volume"])
    df = prices.merge(caps, on="ts").merge(vols, on="ts")
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    df = df.drop(columns=["ts"]).drop_duplicates(subset="date").set_index("date")
    return df


def fetch_coin_snapshot(coin_id):
    url = f"{COINGECKO_BASE}/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "true",
        "developer_data": "true",
        "sparkline": "false",
    }
    r = requests.get(url, params=params, headers=_cg_headers(), timeout=30)
    r.raise_for_status()
    payload = r.json()
    md = payload.get("market_data", {})
    dd = payload.get("developer_data", {})
    cd = payload.get("community_data", {})
    return {
        "circulating_supply": md.get("circulating_supply"),
        "total_supply": md.get("total_supply"),
        "max_supply": md.get("max_supply"),
        "commit_count_4_weeks": dd.get("commit_count_4_weeks"),
        "forks": dd.get("forks"),
        "stars": dd.get("stars"),
        "pull_requests_merged": dd.get("pull_requests_merged"),
        "closed_issues": dd.get("closed_issues"),
        "reddit_subscribers": cd.get("reddit_subscribers"),
        "reddit_average_posts_48h": cd.get("reddit_average_posts_48h"),
        "reddit_average_comments_48h": cd.get("reddit_average_comments_48h"),
        "twitter_followers": cd.get("twitter_followers"),
        "sentiment_votes_up_percentage": payload.get("sentiment_votes_up_percentage"),
        "sentiment_votes_down_percentage": payload.get("sentiment_votes_down_percentage"),
    }


def fetch_fear_greed_index(limit=0):
    url = "https://api.alternative.me/fng/"
    params = {"limit": limit, "format": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df["value"] = df["value"].astype(int)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df = df[["date", "value", "value_classification"]].sort_values("date")
    df = df.rename(columns={"value": "fear_greed", "value_classification": "fear_greed_label"})
    return df.set_index("date")


def run():
    if not COINGECKO_API_KEY:
        print("WARNING: COINGECKO_API_KEY is not set. CoinGecko now rejects unauthenticated "
              "market_chart requests with 401. Get a free Demo key at "
              "https://www.coingecko.com/en/api/pricing and set os.environ['COINGECKO_API_KEY'] "
              "before rerunning.")

    fng = fetch_fear_greed_index(limit=0)
    fng.to_csv(os.path.join(DATA_DIR, "fear_greed_index.csv"))
    print("saved fear_greed_index.csv")

    for symbol, coin_id in ASSETS.items():
        try:
            chart = fetch_market_chart(coin_id, days=365)
            chart.to_csv(os.path.join(DATA_DIR, f"{symbol}_market_chart.csv"))
            print(f"saved {symbol}_market_chart.csv ({len(chart)} rows)")
        except requests.exceptions.HTTPError as e:
            print(f"FAILED market_chart for {symbol}: {e}")
            time.sleep(2)
            continue

        try:
            snapshot = fetch_coin_snapshot(coin_id)
            with open(os.path.join(DATA_DIR, f"{symbol}_snapshot.json"), "w") as fh:
                json.dump(snapshot, fh, indent=2)
            print(f"saved {symbol}_snapshot.json")
        except requests.exceptions.HTTPError as e:
            print(f"FAILED snapshot for {symbol}: {e}")

        time.sleep(2)


if __name__ == "__main__":
    run()
