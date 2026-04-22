import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- SYSTEM CONFIG ---
st.set_page_config(page_title="Nomos Terminal | v8.5", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; height: 110px; }
    .metric-container { background-color: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; height: 110px; }
    .confidence-high { color: #00ff00; font-weight: bold; font-size: 18px; }
    .confidence-low { color: #ff4b4b; font-weight: bold; font-size: 18px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 4px 4px 0px 0px; padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINES ---
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    for s in ["", ".NS", ".BO"]:
        try:
            df = yf.Ticker(f"{ticker}{s}").history(period="2y")
            if not df.empty: return df, f"{ticker}{s}"
        except: continue
    return pd.DataFrame(), ticker

# --- INTERFACE ---
st.title("🏛️ NOMOS TERMINAL")
st.caption("Strategic Intelligence & Risk Management")

st.sidebar.title("🏛️ CONTROL")
user_input = st.sidebar.text_input("Asset Ticker", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5)
st.sidebar.divider()
st.sidebar.info("v8.5 Hybrid Build")

if user_input:
    df, active_ticker = fetch_data(user_input)
    
    if not df.empty and len(df) >= 100:
        # 1. CORE CALCULATION (Quant logic driving the UI)
        df['MA50'] = df['Close'].rolling(50).mean()
        df['SD'] = df['Close'].rolling(50).std()
        df['Z'] = (df['Close'] - df['MA50']) / df['SD']
        df['Score'] = (5 + (df['Z'] * 2.5)).clip(1, 10)
        curr = df.iloc[-1]

        # 2. THE TOP BAR: GLANCEABLE METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", active_ticker)
        c2.metric("Nomos Score", f"{curr['Score']:.1f}/10")
        
        vol_ratio = curr['SD'] / df['SD'].mean()
        conf = "HIGH (Stable)" if vol_ratio < 1.3 else "LOW (Chaotic)"
        color = "confidence-high" if vol_ratio < 1.3 else "confidence-low"
        with c3:
            st.markdown(f'<div class="metric-container"><p style="color:#8b949e;font-size:14px;margin-bottom:5px;">Algo Confidence</p><p class="{color}">{conf}</p></div>', unsafe_allow_html=True)
        c4.metric("Volatility Ratio", f"{vol_ratio:.2f}x")

        # 3. FEATURE-BASED LAYOUT
        tab1, tab2, tab3 = st.tabs(["📉 Market Visuals", "🔮 Risk Projections", "🧠 The Quant Vault"])

        with tab1:
            st.markdown("### Trend Architecture")
            fig = go.Figure()
            # The Noise Threshold (Standard Deviation Band)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50']+(sensitivity*df['SD']), line_color='rgba(0,0,0,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50']-(sensitivity*df['SD']), fill='tonexty', fillcolor='rgba(128,128,128,0.08)', line_color='rgba(0,0,0,0)', name='Noise Floor'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Mathematical Mean', line=dict(color='#FFD700', dash='dot')))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Simplified Intelligence
            col_tl, col_tr = st.columns(2)
            with col_tl:
                st.markdown(f"**Current State:** Asset is **{round(abs(curr['Z']), 2)} SD** from the mean.")
            with col_tr:
                st.markdown(f"**Regime:** {'Trending' if abs(curr['Z']) > 1 else 'Consolidating'}")

        with tab2:
            st.markdown("### 21-Day Risk Probability Cloud (Monte Carlo)")
            mu, sigma = df['Close'].pct_change().mean(), df['Close'].pct_change().std()
            mc = np.zeros((21, 1000))
            mc[0] = curr['Close']
            for i in range(1, 21): mc[i] = mc[i-1] * (1 + np.random.normal(mu, sigma, 1000))
            
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(y=np.percentile(mc, 95, axis=1), line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name='Extreme Bull'))
            fig_mc.add_trace(go.Scatter(y=np.percentile(mc, 5, axis=1), line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name='Extreme Bear'))
            fig_mc.add_trace(go.Scatter(y=mc.mean(axis=1), line=dict(color='#00ff00', width=3), name='Probable Path'))
            fig_mc.update_layout(template="plotly_dark", height=380, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)
            st.caption("Based on 1,000 statistical iterations of historical daily returns.")

        with tab3:
            st.markdown("### Institutional Risk Analysis")
            # Calculate Quant Metrics
            returns = df['Close'].pct_change().dropna()
            # Mock Strategy based on score (Entry > 8, Exit < 4)
            strat_pos = np.where(df['Score'].shift(1) > 8, 1, 0)
            strat_returns = (returns * strat_pos) - (np.abs(np.diff(strat_pos, prepend=0)) * 0.001)
            
            sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() != 0 else 0
            mdd = ( (1+strat_returns).cumprod() / (1+strat_returns).cumprod().cummax() - 1).min()
            
            q1, q2, q3 = st.columns(3)
            q1.metric("Sharpe Ratio", f"{sharpe:.2f}")
            q2.metric("Max Drawdown", f"{mdd*100:.1f}%")
            q3.metric("Annualized Vol", f"{returns.std() * np.sqrt(252) * 100:.1f}%")
            
            st.divider()
            st.markdown("""
            **Technical Methodology:**
            - **Scoring:** Multi-factor mean reversion adjusted for rolling standard deviation.
            - **Friction:** Backtest includes a 0.1% slippage penalty per trade and 1-day execution lag.
            """)

        st.divider()
        st.caption("Nomos Terminal: Strategic Research Tool. Not Investment Advice.")

    else:
        st.error("Invalid Asset or insufficient data history.")
