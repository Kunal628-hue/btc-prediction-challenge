import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from model import get_binance_data, predict_next_bar
import os

st.set_page_config(page_title="BTC Predictor Dashboard", layout="wide")
st.title("Bitcoin Next-Hour Predictor")
st.markdown("AlphaI × Polaris Challenge Submission")

@st.cache_data(ttl=300)
def fetch_and_predict():
    prices = get_binance_data(symbol="BTCUSDT", interval="1h", limit=500)
    S_t1, low95, high95, S0_bt = predict_next_bar(prices)
    return prices, S_t1, low95, high95, S0_bt

try:
    with st.spinner("Fetching data and running model..."):
        prices, S_t1, low95, high95, current_price = fetch_and_predict()
except Exception as e:
    st.error(f"Error fetching data or running model: {e}")
    st.stop()

def load_metrics():
    try:
        if os.path.exists('backtest_results.jsonl'):
            df = pd.read_json('backtest_results.jsonl', lines=True)
            cov = df['coverage_95'].mean()
            wid = df['width_95'].mean()
            wink = df['winkler'].mean()
            return cov, wid, wink
    except:
        pass
    return None, None, None

cov, wid, wink = load_metrics()

if cov is not None:
    st.header("Part A Backtest Metrics (720 bars)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Coverage (Target ~0.95)", f"{cov:.4f}")
    col2.metric("Average Width", f"${wid:.2f}")
    col3.metric("Winkler Score", f"{wink:.2f}")
else:
    st.info("Run `python backtest.py` locally to generate `backtest_results.jsonl` for metrics display.")

st.header("Live Prediction")
col1, col2 = st.columns(2)
col1.metric("Current BTC Price", f"${current_price:,.2f}")
col2.metric("Predicted 95% Range (Next Hour)", f"${low95:,.2f} - ${high95:,.2f}")

# Part C: Persistence
prediction_record = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'current_price': float(current_price),
    'low_95': float(low95),
    'high_95': float(high95)
}
with open("history.jsonl", "a") as f:
    f.write(json.dumps(prediction_record) + "\n")

st.subheader("Last 50 Bars + Prediction Ribbon")
last_50 = prices.tail(50)
next_time = last_50.index[-1] + pd.Timedelta(hours=1)

fig = go.Figure()
# Historical Price
fig.add_trace(go.Scatter(x=last_50.index, y=last_50.values, mode='lines+markers', name='BTC Price', line=dict(color='#F7931A')))

# Prediction Range Polygon
fig.add_trace(go.Scatter(
    x=[last_50.index[-1], next_time, next_time, last_50.index[-1]],
    y=[current_price, high95, low95, current_price],
    fill='toself',
    fillcolor='rgba(247,147,26,0.3)',
    line=dict(color='rgba(255,255,255,0)'),
    name='95% Confidence Interval',
    hoverinfo='skip'
))

fig.update_layout(
    height=500, 
    template="plotly_dark", 
    xaxis_title="Time", 
    yaxis_title="Price (USD)",
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)

if os.path.exists('history.jsonl'):
    st.subheader("Prediction History (Part C)")
    try:
        hist_df = pd.read_json('history.jsonl', lines=True)
        hist_df = hist_df.sort_values(by='timestamp', ascending=False)
        st.dataframe(hist_df.head(20), use_container_width=True)
    except:
        pass
