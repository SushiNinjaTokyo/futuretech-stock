#!/usr/bin/env python3
import os, json

OUT_DIR = os.environ.get("OUT_DIR","site")
REPORT_DATE = os.environ.get("REPORT_DATE")
SOURCE = os.environ.get("DII_SOURCE","skip")

def main():
    os.makedirs(f"{OUT_DIR}/data/dii", exist_ok=True)
    # 実装がない/失敗時でも空のスコア辞書を返す（0件でも落ちない）
    payload = {
        "score_0_1": {},  # { CompanyName: 0~1 }
        "meta": {"source": SOURCE}
    }
    with open(f"{OUT_DIR}/data/dii/latest.json","w") as f:
        json.dump(payload, f, indent=2)
    if REPORT_DATE:
        os.makedirs(f"{OUT_DIR}/data/{REPORT_DATE}", exist_ok=True)
        with open(f"{OUT_DIR}/data/{REPORT_DATE}/dii.json","w") as f:
            json.dump(payload, f, indent=2)
    print(f"[DII] saved: {OUT_DIR}/data/dii/latest.json and {OUT_DIR}/data/{REPORT_DATE}/dii.json (symbols=0) source=latest_json")

if __name__ == "__main__":
    main()
