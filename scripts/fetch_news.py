#!/usr/bin/env python3
import os, json, time
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUT_DIR = os.environ.get("OUT_DIR","site")
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV","data/universe.csv")
REPORT_DATE = os.environ.get("REPORT_DATE")
LOOKBACK_DAYS = int(os.environ.get("NEWS_LOOKBACK_DAYS","7"))
MAX_PER_SYMBOL = int(os.environ.get("NEWS_MAX_PER_SYMBOL","20"))
SLEEP_SEC = float(os.environ.get("NEWS_SLEEP_SEC","1.0"))

RSS_SOURCES = [
  "https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
]

def load_universe(path):
    df = pd.read_csv(path)
    if "symbol" not in df.columns: raise RuntimeError("universe.csv needs 'symbol'")
    if "name" not in df.columns: df["name"]=df["symbol"]
    return df

def main():
    os.makedirs(f"{OUT_DIR}/data/news", exist_ok=True)
    u = load_universe(UNIVERSE_CSV)
    analyzer = SentimentIntensityAnalyzer()
    per_symbol_sent = {}

    for _, row in u.iterrows():
        name = row["name"]
        sym  = row["symbol"]
        articles = []
        for src in RSS_SOURCES:
            url = src.format(query=name.replace(" ","+"))
            feed = feedparser.parse(url)
            for e in feed.entries[:MAX_PER_SYMBOL]:
                title = getattr(e, "title", "")
                score = analyzer.polarity_scores(title)["compound"]
                articles.append({"title": title, "sentiment": score})
        if articles:
            # 最終スコアはタイトル感情の平均を0-1へ変換
            s = pd.Series([a["sentiment"] for a in articles])
            z = (s - s.min())/(s.max()-s.min()+1e-9)
            per_symbol_sent[name] = float(z.mean())
        time.sleep(SLEEP_SEC)

    payload = {
        "score_0_1": per_symbol_sent,
        "meta": {"source": "rss/google-news", "lookback_days": LOOKBACK_DAYS}
    }

    with open(f"{OUT_DIR}/data/news/latest.json","w") as f:
        json.dump(payload, f, indent=2)
    if REPORT_DATE:
        os.makedirs(f"{OUT_DIR}/data/{REPORT_DATE}", exist_ok=True)
        with open(f"{OUT_DIR}/data/{REPORT_DATE}/news.json","w") as f:
            json.dump(payload, f, indent=2)
    print(f"[NEWS] saved: {OUT_DIR}/data/news/latest.json and {OUT_DIR}/data/{REPORT_DATE}/news.json ({len(per_symbol_sent)} symbols)")

if __name__ == "__main__":
    main()
