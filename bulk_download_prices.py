import os
import time
import ssl
import pandas as pd
import yfinance as yf

# SSL workaround (for proxy/self-signed environments)
ssl._create_default_https_context = ssl._create_unverified_context

START = "2010-01-01"
END   = "2026-01-07"  # end exclusive in yfinance
OUT_DIR = os.path.join("data", "prices")
os.makedirs(OUT_DIR, exist_ok=True)

# Load tickers from your SP500.csv
sp500_df = pd.read_csv("SP500.csv")
tickers = sp500_df["Symbol"].astype(str).str.strip().str.upper().tolist()
tickers = [t.replace(".", "-") for t in tickers if t and t != "NAN"]  # BRK.B -> BRK-B

def yf_download_retry(symbol: str, retries: int = 5, sleep_sec: float = 2.0):
    # try increase yfinance timeout (works in many versions)
    try:
        import yfinance.shared as shared
        shared._DEFAULTS["timeout"] = 60
    except Exception:
        pass

    last_err = None
    for i in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                start=START,
                end=END,
                auto_adjust=True,
                progress=False,
                threads=False,     # proxy-friendly
            )
            if df is not None and not df.empty:
                df = df[["Open","High","Low","Close","Volume"]].dropna()
                return df
            last_err = "empty"
        except Exception as e:
            last_err = str(e)

        print(f"[WARN] {symbol} attempt {i}/{retries} failed: {last_err}")
        time.sleep(sleep_sec)

    return None

failed = []
ok = 0

for idx, sym in enumerate(tickers, start=1):
    out_path = os.path.join(OUT_DIR, f"{sym}.csv")

    # skip if already downloaded
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"[SKIP] {sym} exists")
        continue

    print(f"[{idx}/{len(tickers)}] Downloading {sym} ...")
    df = yf_download_retry(sym)

    if df is None or df.empty:
        failed.append(sym)
        continue

    df.to_csv(out_path, index=True)
    ok += 1

    # rate limit (important)
    time.sleep(0.8)

print("\nDONE")
print("Downloaded:", ok)
print("Failed:", len(failed))
if failed:
    with open("failed_tickers.txt", "w") as f:
        for s in failed:
            f.write(s + "\n")
    print("Saved failed tickers to failed_tickers.txt")