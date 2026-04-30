import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm
import json
from datetime import datetime

def get_binance_data_full(symbol="BTCUSDT", interval="1h", total_limit=1500):
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
    # Remove duplicates if any
    df = df[~df.index.duplicated(keep='first')]
    return df['close'].sort_index().tail(total_limit)

def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

def simulate_mc_clean(S0, mu, sigma_fig, H, M, bar_sigma2, nu, redundancy_val, info_filter_val,
                      n_sims=10000, dt=1):
    H_max = H.max() if H.max() > 0 else 1.0
    M_max = M.max() if M.max() > 0 else 1.0
    
    α0, δ0 = 0.5, 0.3
    if α0 * H_max + δ0 * M_max >= 1:
        fac = 0.95 / (α0 * H_max + δ0 * M_max)
        α0 *= fac
        δ0 *= fac
    base_params = {'alpha': α0, 'delta': δ0, 'gamma': 0.2}
    
    current = -1
    H_val = min(H.iloc[current] / H_max, 1.0)
    M_val = min(M.iloc[current] / M_max, 1.0)
    crisis  = (H_val > 0.8) or (M_val > 0.8)
    delta_t = base_params['delta'] if crisis else 0.0
    
    sigma2_initial = sigma_fig.iloc[current] ** 2
    
    sigma2 = (
        sigma2_initial * (1 + base_params['alpha'] * H_val + delta_t * M_val)
        + base_params['gamma'] * (bar_sigma2 - sigma2_initial)
    )
    sigma2 *= max(1e-12, redundancy_val)
    sigma2 *= 1 + 0.5 * info_filter_val
    sigma2 = max(1e-6, min(sigma2, 0.5))
    
    Z = np.random.standard_t(nu, size=n_sims) * np.sqrt((nu - 2) / nu)
    S_t1 = S0 * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)
    return S_t1

def run_backtest():
    np.random.seed(42)
    # 720 bars for testing, 500 for training = 1220 total
    print("Fetching data...")
    prices = get_binance_data_full("BTCUSDT", "1h", 1220)
    print(f"Got {len(prices)} bars.")
    
    train_size = 500
    test_size = 720
    
    if len(prices) < train_size + test_size:
        raise ValueError("Not enough data fetched.")
    
    res_li = []
    
    print("Starting backtest...")
    for i in tqdm(range(len(prices) - test_size, len(prices))):
        train_prices = prices.iloc[i - train_size : i + 1]
        
        # We predict the next bar (i+1), wait, if i = len(prices) - 1, actual is not available
        if i + 1 >= len(prices):
            continue
            
        actual = prices.iloc[i + 1]
        
        log_ret = np.log(train_prices / train_prices.shift(1)).dropna()
        
        try:
            am = arch_model(log_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
            res = am.fit(disp='off', show_warning=False)
        except Exception as e:
            print(f"Error fitting arch model at {prices.index[i]}: {e}")
            continue
            
        sigma_fig = res.conditional_volatility / 100
        resid = (log_ret * 100 - res.params.get('mu', log_ret.mean()*100)) / res.conditional_volatility
        
        # Fit t-dist for nu
        try:
            nu_bt = max(4.0, stats.t.fit(resid, floc=0, fscale=1)[0])
        except:
            nu_bt = 4.0
            
        H_bt = rolling_entropy(resid, window=60)
        M_bt = log_ret.abs().rolling(60).mean()
        
        # Redundancy & Info filter
        var_5 = train_prices.rolling(5).var()
        var_20 = train_prices.rolling(20).var()
        redundancy_series = 1 + 0.1 * np.log1p(var_5 / var_20)
        
        # Fill NA for redundancy
        redundancy_series = redundancy_series.fillna(1.0)
        info_filter_series = (H_bt > H_bt.mean()).astype(float)
        
        S0_bt = train_prices.iloc[-1]
        
        S_t1 = simulate_mc_clean(
            S0_bt, log_ret.mean(),
            sigma_fig, H_bt.fillna(0), M_bt.fillna(0),
            (sigma_fig**2).mean(), nu_bt, 
            redundancy_series.iloc[-1], 
            info_filter_series.iloc[-1],
            n_sims=10000
        )
        
        low95, high95 = np.percentile(S_t1, [2.5, 97.5])
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

    # Save to jsonl
    with open('backtest_results.jsonl', 'w') as f:
        for r in res_li:
            f.write(json.dumps(r) + '\n')
            
    df = pd.DataFrame(res_li)
    print("\n--- RESULTS ---")
    print(f"Coverage 95%: {df['coverage_95'].mean():.4f}")
    print(f"Average Width: {df['width_95'].mean():.2f}")
    print(f"Mean Winkler: {df['winkler'].mean():.2f}")

if __name__ == "__main__":
    run_backtest()
