import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIG & COMPLIANCE ---
st.set_page_config(page_title="Nomos Terminal | v5.2", layout="wide")

# Institutional Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    .confidence-high { color: #00ff00; font-weight: bold; }
    .confidence-low { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINES ---

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    suffixes = ["", ".NS", ".BO"]
    for s in suffixes:
        try:
            df = yf.Ticker(f"{ticker}{s}").history(period="2y")
            if not df.empty: return df, f"{ticker}{s}"
        except: continue
    return pd.DataFrame(), ticker

def run_monte_carlo(start_price, sd, days=21, sims=100):
    daily_vol = sd / start_price
    results = np.zeros((days + 1, sims))
    results[0] = start_price
    for s in range(sims):
        for d in range(1, days + 1):
            results[d, s] = results[d-1, s] * (1 + np.random.normal(0, daily_vol))
    return results

# --- SIDEBAR ---
st.sidebar.title("🏛️ NOMOS CONTROL")
user_input = st.sidebar.text_input("Asset Ticker", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5)
st.sidebar.divider()
st.sidebar.info("v5.2 | Decision Support System (Non-Advisory)")

# --- MAIN TERMINAL ---
st.title("🏛️ NOMOS TERMINAL")

if user_input:
    df, active_ticker = fetch_data(user_input)
    
    if not df.empty and len(df) >= 100:
        # Math Layer
        df['MA50'] = df['Close'].rolling(50).mean()
        df['SD'] = df['Close'].rolling(50).std()
        df['RSI'] = (100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                        (-df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))))
        
        # Nomos Score Calculation
        df['diff'] = df['Close'] - df['MA50']
        df['Nomos_Score'] = (5 + (df['diff'] / (sensitivity * df['SD'])) * 5 + (df['RSI'] - 50) / 10).clip(1, 10)
        curr = df.iloc[-1]
        
        # 1. TOP METRICS & CONFIDENCE ENGINE
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", active_ticker)
        c2.metric("Nomos Score", f"{curr['Nomos_Score']:.1f}/10")
        
        # THE COMPLIANCE PIVOT: Measuring Math Stability, not Market Direction
        vol_ratio = curr['SD'] / df['SD'].mean()
        if vol_ratio < 1.2 and 30 < curr['RSI'] < 70:
            conf_label, conf_class = "HIGH (Stable Math)", "confidence-high"
        else:
            conf_label, conf_class = "LOW (Chaotic Data)", "confidence-low"
        
        c3.markdown(f"**Algo Confidence** \n<span class='{conf_class}'>{conf_label}</span>", unsafe_allow_html=True)
        c4.metric("Volatility Index", f"{vol_ratio:.2f}x")

        # 2. TREND VISUALIZATION
        st.markdown("### Trend Architecture")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'] + (sensitivity * df['SD']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'] - (sensitivity * df['SD']), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', line_color='rgba(0,0,0,0)', name='Noise Threshold'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Mathematical Mean', line=dict(color='#FFD700', dash='dot')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # 3. RISK PROJECTION & LOGIC
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown("### 21-Day Probabilistic Simulation")
            mc = run_monte_carlo(curr['Close'], curr['SD'])
            fig_mc = go.Figure()
            for i in range(mc.shape[1]):
                fig_mc.add_trace(go.Scatter(y=mc[:, i], mode='lines', line=dict(width=1, color='rgba(0, 255, 150, 0.03)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(y=mc.mean(axis=1), mode='lines', line=dict(color='#00ff00', width=3), name='Statistical Mean'))
            fig_mc.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)

        with col_r:
            st.markdown("### Terminal Intelligence")
            st.write(f"**Score Context:** The current score of {curr['Nomos_Score']:.1f} suggests the asset is " + 
                     ("extended above " if curr['Nomos_Score'] > 5 else "compressed below ") + "its mathematical mean.")
            
            with st.expander("🛡️ HOW TO INTERPRET THIS DATA"):
                st.write("""
                1. **Score > 8.0:** Price is significantly outperforming the trend.
                2. **Score < 4.0:** Price is significantly underperforming the trend.
                3. **Confidence:** If 'LOW', the Monte Carlo simulation (left) is less reliable due to high volatility.
                """)

        # 4. BACKTEST & LEGAL FOOTER
        with st.expander("🚀 HISTORICAL STRATEGY VALIDATION"):
            df['Sig'] = np.where(df['Nomos_Score'] > 8, 1, np.where(df['Nomos_Score'] < 4, 0, np.nan))
            df['Pos'] = df['Sig'].ffill().fillna(0)
            df['Strat_Ret'] = (df['Close'].pct_change() * df['Pos'].shift(1)).add(1).cumprod() - 1
            df['Mkt_Ret'] = (df['Close'] / df['Close'].iloc[0]) - 1
            
            b1, b2 = st.columns(2)
            b1.metric("Strategy Alpha", f"{df['Strat_Ret'].iloc[-1]*100:.1f}%")
            b2.metric("Market Benchmark", f"{df['Mkt_Ret'].iloc[-1]*100:.1f}%")

        st.divider()
        st.caption("⚠️ **LEGAL DISCLAIMER:** Nomos Terminal is a mathematical simulation tool for educational research only. "
                   "It does not provide investment advice or buy/sell recommendations. Trading involves significant risk. "
                   "Not registered with SEBI. User assumes all responsibility for financial decisions.")

    else:
        st.error("Invalid Ticker or insufficient history (2 years required).")
