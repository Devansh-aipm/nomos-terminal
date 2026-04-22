import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- SYSTEM CONFIG ---
st.set_page_config(page_title="Nomos Terminal | v7.0", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; height: 115px; }
    .metric-container { background-color: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; height: 115px; }
    .confidence-high { color: #00ff00; font-weight: bold; font-size: 20px; }
    .confidence-low { color: #ff4b4b; font-weight: bold; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- QUANTITATIVE ENGINES ---

def get_performance_metrics(returns):
    """Calculate professional risk-adjusted metrics."""
    if returns.empty: return 0, 0, 0
    # Annualized Sharpe (Assumes 252 trading days)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    # Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    # Win Rate
    win_rate = len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
    return sharpe, max_dd, win_rate

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    for s in ["", ".NS", ".BO"]:
        try:
            df = yf.Ticker(f"{ticker}{s}").history(period="2y")
            if not df.empty: return df, f"{ticker}{s}"
        except: continue
    return pd.DataFrame(), ticker

# --- INTERFACE ---
st.title("🏛️ NOMOS TERMINAL v7.0")
st.caption("Quantitative Risk & Decision Support System | Institutional Grade")

# Sidebar Controls
st.sidebar.title("🛠️ PARAMETERS")
user_input = st.sidebar.text_input("Ticker", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity (Z-Score)", 0.5, 3.0, 1.5)
slippage = st.sidebar.slider("Slippage + Fees (%)", 0.0, 0.5, 0.1) / 100

if user_input:
    df, active_ticker = fetch_data(user_input)
    
    if not df.empty and len(df) >= 100:
        # 1. THE NOMOS MULTI-FACTOR ENGINE
        df['MA50'] = df['Close'].rolling(50).mean()
        df['SD'] = df['Close'].rolling(50).std()
        df['Z_Score'] = (df['Close'] - df['MA50']) / df['SD']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # Scoring: Z-Score (Mean Reversion) + RSI (Momentum)
        df['Nomos_Score'] = (5 + (df['Z_Score'] * 2) + (df['RSI'] - 50) / 10).clip(1, 10)
        
        # 2. VECTORIZED BACKTEST WITH REAL-WORLD FRICTION
        # Entry: Score > 8 | Exit: Score < 4 (Lagged by 1 day to prevent look-ahead bias)
        df['Signal'] = np.where(df['Nomos_Score'] > 8, 1, np.where(df['Nomos_Score'] < 4, 0, np.nan))
        df['Position'] = df['Signal'].ffill().fillna(0).shift(1) # Entry happens NEXT day
        
        # Transaction Costs Logic
        df['Trades'] = df['Position'].diff().abs().fillna(0)
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = (df['Market_Return'] * df['Position']) - (df['Trades'] * slippage)
        
        sharpe, mdd, win_rate = get_performance_metrics(df['Strategy_Return'].dropna())

        # 3. DASHBOARD
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c2.metric("Max Drawdown", f"{mdd*100:.1f}%")
        c3.metric("Win Rate", f"{win_rate*100:.1f}%")
        
        # Confidence logic based on Volatility Regime
        vol_ratio = (df['SD'].iloc[-1] / df['SD'].mean())
        conf_label = "HIGH (Stable)" if vol_ratio < 1.2 else "LOW (Chaotic)"
        conf_class = "confidence-high" if vol_ratio < 1.2 else "confidence-low"
        with c4:
            st.markdown(f'<div class="metric-container"><p style="color:#8b949e;font-size:14px;margin-bottom:5px;">Algo Confidence</p><p class="{conf_class}">{conf_label}</p></div>', unsafe_allow_html=True)

        # 4. PATH SIMULATION (1,000 ITERATIONS)
        st.markdown("### Risk Probability Cloud (Monte Carlo)")
        mc_sims = 1000
        horizon = 21
        mu, sigma = df['Market_Return'].mean(), df['Market_Return'].std()
        mc_paths = np.zeros((horizon, mc_sims))
        mc_paths[0] = df['Close'].iloc[-1]
        for i in range(1, horizon):
            mc_paths[i] = mc_paths[i-1] * (1 + np.random.normal(mu, sigma, mc_sims))
        
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(y=np.percentile(mc_paths, 95, axis=1), line=dict(color='gray', dash='dot'), name='95th %ile'))
        fig_mc.add_trace(go.Scatter(y=np.percentile(mc_paths, 5, axis=1), line=dict(color='gray', dash='dot'), name='5th %ile'))
        fig_mc.add_trace(go.Scatter(y=mc_paths.mean(axis=1), line=dict(color='#00ff00', width=3), name='Expected Mean'))
        fig_mc.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_mc, use_container_width=True)

        st.divider()
        st.info("**Quant Note:** Backtest includes a 1-day execution lag and simulated slippage. Sharpe ratio assumes a 0% risk-free rate for baseline volatility assessment.")

    else:
        st.error("Invalid ticker or insufficient history.")
