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

def theoretical_option_price(S0, K, T, r, paths, option_type):
    # paths is [n_sims, n_steps+1]
    # T is in years, but here we use steps. 
    # To match starter: t_index = int(T * 365) if crypto
    t_index = -1 # Use the final step of the simulation
    ST = paths[:, t_index]
    
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    
    # Simple discounting
    return np.exp(-r * T) * np.mean(payoff)

def calculate_greeks(S0, K, T, r, paths, option_type, epsilon=0.01):
    base_price = theoretical_option_price(S0, K, T, r, paths, option_type)
    
    # Delta & Gamma via bumping S0 (approximation using the same paths scaled)
    price_up = theoretical_option_price(S0*(1+epsilon), K, T, r, paths*(1+epsilon), option_type)
    price_down = theoretical_option_price(S0*(1-epsilon), K, T, r, paths*(1-epsilon), option_type)
    
    delta = (price_up - price_down) / (2 * epsilon * S0)
    gamma = (price_up - 2*base_price + price_down) / ((epsilon * S0)**2)
    
    # Vega via bumping the entire paths (sigma proxy)
    # We don't have a simple sigma to bump, so we scale the deviations from S0
    paths_up = S0 + (paths - S0) * (1 + epsilon)
    paths_down = S0 + (paths - S0) * (1 - epsilon)
    vega = (theoretical_option_price(S0, K, T, r, paths_up, option_type) - 
            theoretical_option_price(S0, K, T, r, paths_down, option_type)) / (2 * epsilon)
            
    # Theta (approx by reducing T slightly)
    T_small = max(T - 1/365, 0.001)
    theta = (theoretical_option_price(S0, K, T_small, r, paths, option_type) - base_price)
    
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

def get_options_strategy(prices, S_t1):
    # S_t1 is the [n_sims] prediction for the next hour
    # We'll use this to price a 1-day strangle as a demonstration of the strategy
    S0 = prices.iloc[-1]
    r = 0.05
    T = 1/365 # 1 day
    
    # Reshape S_t1 for theoretical_option_price compatibility
    paths = S_t1.reshape(-1, 1)
    
    # Optimal Strikes (95% range)
    K_put, K_call = np.percentile(S_t1, [2.5, 97.5])
    
    theoretical_put = theoretical_option_price(S0, K_put, T, r, paths, 'put')
    theoretical_call = theoretical_option_price(S0, K_call, T, r, paths, 'call')
    
    greeks_put = calculate_greeks(S0, K_put, T, r, paths, 'put')
    greeks_call = calculate_greeks(S0, K_call, T, r, paths, 'call')
    
    # Mock market data (as seen in starter.py)
    # In a real app, this would fetch from an options API
    market_put = theoretical_put * (0.95 + 0.1 * np.random.rand())
    market_call = theoretical_call * (0.95 + 0.1 * np.random.rand())
    
    total_theory = theoretical_put + theoretical_call
    total_market = market_put + market_call
    ratio = total_market / total_theory if total_theory > 0 else 1.0
    
    recommendation = "NEUTRAL"
    if ratio < 0.85: recommendation = "BUY STRANGLE (Underpriced)"
    elif ratio > 1.15: recommendation = "SELL STRANGLE (Overpriced)"
    
    return {
        'K_put': K_put,
        'K_call': K_call,
        'theory_put': theoretical_put,
        'theory_call': theoretical_call,
        'market_put': market_put,
        'market_call': market_call,
        'greeks_put': greeks_put,
        'greeks_call': greeks_call,
        'recommendation': recommendation,
        'ratio': ratio
    }

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
