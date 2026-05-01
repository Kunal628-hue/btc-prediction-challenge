import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from model import get_binance_data, predict_next_bar, get_options_strategy
import os

st.set_page_config(page_title="BTC Predictor Pro", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for premium look
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    h1, h2, h3 { color: #F7931A !important; }
    .status-box { padding: 15px; border-radius: 8px; margin: 10px 0; }
    .buy-signal { background-color: rgba(0, 255, 0, 0.1); border: 1px solid #00ff00; color: #00ff00; }
    .sell-signal { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #ff0000; color: #ff0000; }
    .neutral-signal { background-color: rgba(255, 255, 255, 0.05); border: 1px solid #888; color: #fff; }
</style>
""", unsafe_allow_html=True)

st.title("₿ Bitcoin Next-Hour Predictor Pro")
st.markdown("### AlphaI × Polaris Challenge | Advanced Alpha Generation")

@st.cache_data(ttl=300)
def fetch_and_predict():
    prices = get_binance_data(symbol="BTCUSDT", interval="1h", limit=500)
    S_t1, low95, high95, current_price = predict_next_bar(prices)
    opt_strat = get_options_strategy(prices, S_t1)
    return prices, S_t1, low95, high95, current_price, opt_strat

try:
    with st.spinner("Analyzing market dynamics and computing Greeks..."):
        prices, S_t1, low95, high95, current_price, opt = fetch_and_predict()
except Exception as e:
    st.error(f"Error fetching data or running model: {e}")
    st.stop()

# Part C: Refined Persistence (Avoid duplicates)
prediction_record = {
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:00:00'),
    'current_price': round(float(current_price), 2),
    'low_95': round(float(low95), 2),
    'high_95': round(float(high95), 2),
    'recommendation': opt['recommendation']
}

def save_prediction(record):
    try:
        existing = pd.read_json('history.jsonl', lines=True)
        if record['timestamp'] in existing['timestamp'].values:
            return # Already saved for this hour
    except:
        pass
    with open("history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

save_prediction(prediction_record)

# Layout: Prediction & Backtest
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Analysis")
    m1, m2 = st.columns(2)
    m1.metric("Current Price", f"${current_price:,.2f}")
    m2.metric("Target Range (95%)", f"${low95:,.0f} — ${high95:,.0f}")
    
    # Chart
    last_50 = prices.tail(50)
    next_time = last_50.index[-1] + pd.Timedelta(hours=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_50.index, y=last_50.values, mode='lines', name='Price', line=dict(color='#F7931A', width=3)))
    fig.add_trace(go.Scatter(
        x=[last_50.index[-1], next_time, next_time, last_50.index[-1]],
        y=[current_price, high95, low95, current_price],
        fill='toself', fillcolor='rgba(247,147,26,0.2)', line=dict(color='rgba(0,0,0,0)'),
        name='95% Range'
    ))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Part A: Backtest")
    def load_metrics():
        try:
            if os.path.exists('backtest_results.jsonl'):
                df = pd.read_json('backtest_results.jsonl', lines=True)
                return df['coverage_95'].mean(), df['width_95'].mean(), df['winkler'].mean()
        except: pass
        return None, None, None
    
    cov, wid, wink = load_metrics()
    if cov:
        st.metric("Coverage (30d)", f"{cov:.2%}")
        st.metric("Avg Width", f"${wid:.0f}")
        st.metric("Winkler Score", f"{wink:.2f}")
    else:
        st.info("Metrics pending backtest run.")

# Part D: Options Strategy (NEW)
st.divider()
st.header("Part D: Institutional Options Strategy")
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    st.subheader("Greeks (1-Day Strangle)")
    def format_greek(val): return f"{val:.4f}"
    st.write(f"**Delta:** {format_greek(opt['greeks_call']['delta'])}")
    st.write(f"**Gamma:** {format_greek(opt['greeks_call']['gamma'])}")
    st.write(f"**Vega:** {format_greek(opt['greeks_call']['vega'])}")
    st.write(f"**Theta:** {format_greek(opt['greeks_call']['theta'])}")

with c2:
    st.subheader("Pricing")
    st.write(f"**Model Price:** `${opt['theory_put']+opt['theory_call']:.2f}`")
    st.write(f"**Market Price:** `${opt['market_put']+opt['market_call']:.2f}`")
    st.write(f"**IV/Theory Ratio:** `{opt['ratio']:.2f}x`")

with c3:
    st.subheader("Recommendation")
    sig = opt['recommendation']
    css_class = "buy-signal" if "BUY" in sig else "sell-signal" if "SELL" in sig else "neutral-signal"
    st.markdown(f'<div class="status-box {css_class}"><b>{sig}</b><br>Suggested Strikes: Put ${opt["K_put"]:,.0f} | Call ${opt["K_call"]:,.0f}</div>', unsafe_allow_html=True)

# History
st.divider()
st.subheader("Part C: Prediction Archive")
if os.path.exists('history.jsonl'):
    hist_df = pd.read_json('history.jsonl', lines=True).sort_values(by='timestamp', ascending=False)
    st.dataframe(hist_df.head(10), use_container_width=True)
