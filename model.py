import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model

def get_binance_data(symbol="BTCUSDT", interval="1h", limit=500):
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Error {r.status_code}: {r.text}")
    data = r.json()
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['close_time'], unit='ms')
    df.set_index('date', inplace=True)
    df['close'] = df['close'].astype(float)
    return df['close'].sort_index()

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

def predict_next_bar(prices):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    am = arch_model(log_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
    res = am.fit(disp='off', show_warning=False)
    
    sigma_fig = res.conditional_volatility / 100
    resid = (log_ret * 100 - res.params.get('mu', log_ret.mean()*100)) / res.conditional_volatility
    
    try:
        nu_bt = max(4.0, stats.t.fit(resid, floc=0, fscale=1)[0])
    except:
        nu_bt = 4.0
        
    H_bt = rolling_entropy(resid, window=60).fillna(0)
    M_bt = log_ret.abs().rolling(60).mean().fillna(0)
    
    var_5 = prices.rolling(5).var()
    var_20 = prices.rolling(20).var()
    redundancy_series = (1 + 0.1 * np.log1p(var_5 / var_20)).fillna(1.0)
    info_filter_series = (H_bt > H_bt.mean()).astype(float)
    
    S0_bt = prices.iloc[-1]
    
    S_t1 = simulate_mc_clean(
        S0_bt, log_ret.mean(),
        sigma_fig, H_bt, M_bt,
        (sigma_fig**2).mean(), nu_bt, 
        redundancy_series.iloc[-1], 
        info_filter_series.iloc[-1],
        n_sims=10000
    )
    
    low95, high95 = np.percentile(S_t1, [2.5, 97.5])
    return S_t1, low95, high95, S0_bt
