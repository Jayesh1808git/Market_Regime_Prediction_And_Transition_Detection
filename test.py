
import pandas as pd, json

n = 320

price = pd.read_csv("data/raw_ohlcv.csv").tail(n)
vix = pd.read_csv("data/raw_vix.csv").tail(n)
usdinr = pd.read_csv("data/raw_usdinr.csv").tail(n)
crude = pd.read_csv("data/raw_crude.csv").tail(n)

payload = {
    "price": [
        {
            "date": str(r["Date"]),
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
            "volume": float(r.get("Volume", 0)),
        }
        for _, r in price.iterrows()
    ],
    "vix": [{"date": str(r["Date"]), "close": float(r["Close"])} for _, r in vix.iterrows()],
    "usdinr": [{"date": str(r["Date"]), "close": float(r["Close"])} for _, r in usdinr.iterrows()],
    "crude": [{"date": str(r["Date"]), "close": float(r["Close"])} for _, r in crude.iterrows()],
}

with open("test_payload.json", "w") as f:
    json.dump(payload, f)

print("Wrote test_payload.json")

