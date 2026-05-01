import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from model import get_binance_data, predict_next_bar, rolling_entropy, simulate_mc_clean
import requests

def get_binance_data_full(symbol="BTCUSDT", interval="1h", total_limit=1220):
    all_data = []
    end_time = None
    while len(all_data) < total_limit:
        limit = min(1000, total_limit - len(all_data))
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if end_time:
            url += f"&endTime={end_time}"
        
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Error {r.status_code}: {r.text}")
        
        data = r.json()
        if not data:
            break
        
        all_data = data + all_data
        end_time = data[0][0] - 1
        
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['close_time'], unit='ms')
    df.set_index('date', inplace=True)
    df['close'] = df['close'].astype(float)
    df = df[~df.index.duplicated(keep='first')]
    return df['close'].sort_index().tail(total_limit)

def run_backtest():
    np.random.seed(42)
    print("Fetching data for 30-day backtest...")
    prices = get_binance_data_full("BTCUSDT", "1h", 1220)
    print(f"Got {len(prices)} bars.")
    
    train_size = 500
    test_size = 720
    
    res_li = []
    
    print("Starting simulation loop (No-Peeking)...")
    for i in tqdm(range(len(prices) - test_size, len(prices) - 1)):
        train_prices = prices.iloc[i - train_size : i + 1]
        actual = prices.iloc[i + 1]
        
        # Predict
        try:
            _, low95, high95, _ = predict_next_bar(train_prices)
        except Exception as e:
            continue
            
        width95 = high95 - low95
        alpha = 0.05
        winkler = (width95 + (2/alpha)*(low95-actual)) if actual < low95 else \
                  (width95 + (2/alpha)*(actual-high95)) if actual > high95 else \
                  width95
                  
        res_li.append({
            'timestamp': prices.index[i+1].isoformat(),
            'actual': actual,
            'low_95': low95,
            'high_95': high95,
            'width_95': width95,
            'coverage_95': int(low95 <= actual <= high95),
            'winkler': winkler
        })

    with open('backtest_results.jsonl', 'w') as f:
        for r in res_li:
            f.write(json.dumps(r) + '\n')
            
    df = pd.DataFrame(res_li)
    print("\n--- REFRESHED BACKTEST RESULTS ---")
    print(f"Coverage 95%: {df['coverage_95'].mean():.4f}")
    print(f"Average Width: {df['width_95'].mean():.2f}")
    print(f"Mean Winkler: {df['winkler'].mean():.2f}")

if __name__ == "__main__":
    run_backtest()
