# ₿ Bitcoin Next-Hour Predictor Pro
**AlphaI × Polaris Challenge Submission**

An advanced Bitcoin price prediction system that utilizes a Geometric Brownian Motion (GBM) simulator enhanced with FIGARCH volatility modeling and Student-t distribution analysis.

## 🚀 Key Features

### Part A: 30-Day Backtest
- Automated backtesting across the last 720 bars (30 days).
- Metrics tracked: Coverage 95%, Average Width, and Winkler Score.
- Strict "No-Peeking" rule enforcement.

### Part B: Live Dashboard
- Professional Streamlit interface with real-time Binance data integration.
- Dynamic Plotly visualization with 95% confidence intervals.
- Aesthetic dark-mode design optimized for institutional analysis.

### Part C: Refined Persistence
- Automatic archival of hourly predictions to `history.jsonl`.
- Smart deduplication logic to ensure one record per hour.

### Part D: Institutional Options Strategy (Bonus)
- **Greeks Computation**: Real-time Delta, Gamma, Vega, and Theta for a 1-day strangle.
- **Optimal Strike Selection**: Automated strike selection based on the 95% predicted range.
- **Trading Recommendations**: Model-driven Buy/Sell/Neutral signals based on theoretical vs. market price variance.

## 🛠️ Setup & Execution

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Backtest (Refresh Metrics)**:
   ```bash
   python backtest.py
   ```

3. **Launch Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 📈 Methodology
The system accounts for "Fat Tails" and "Volatility Clustering" by fitting a FIGARCH model to the residuals of log-returns. It also incorporates rolling entropy to detect regime shifts and adjust the prediction ribbon accordingly.