import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- SYSTEM CONFIG & THEME ---
st.set_page_config(page_title="Nomos Terminal | v5.4", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; height: 110px; display: flex; flex-direction: column; justify-content: center; }
    .metric-container { background-color: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; height: 110px; }
    .confidence-high { color: #00ff00; font-weight: bold; font-size: 20px; margin: 0; }
    .confidence-low { color: #ff4b4b; font-weight: bold; font-size: 20px; margin: 0; }
    .metric-label { color: #8b949e; font-size: 14px; margin-bottom: 4px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE COMPUTATION ENGINES ---

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    # Seamless Global Search (US + India NSE/BSE)
    for suffix in ["", ".NS", ".BO"]:
        try:
            data = yf.Ticker(f"{ticker}{suffix}").history(period="2y")
            if not data.empty: return data, f"{ticker}{suffix}"
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

# --- SIDEBAR CONTROL ---
st.sidebar.title("🏛️ NOMOS CONTROL")
user_input = st.sidebar.text_input("Asset Search", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5)
st.sidebar.divider()
st.sidebar.info("v5.4 | Master Synchronized Build")

# --- MAIN TERMINAL ---
st.title("🏛️ NOMOS TERMINAL")

if user_input:
    df, active_ticker = fetch_data(user_input)
    
    if not df.empty and len(df) >= 100:
        # 1. MATH LAYER
        df['MA50'] = df['Close'].rolling(50).mean()
        df['SD'] = df['Close'].rolling(50).std()
        
        # Smooth RSI Logic
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # Nomos Scoring (Deterministic)
        df['diff'] = df['Close'] - df['MA50']
        df['Nomos_Score'] = (5 + (df['diff'] / (sensitivity * df['SD'])) * 5 + (df['RSI'] - 50) / 10).clip(1, 10)
        curr = df.iloc[-1]
        
        # 2. THE TOP METRIC BAR (PIXEL PERFECT LAYOUT)
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.metric("Active Ticker", active_ticker)
        
        with c2:
            st.metric("Nomos Score", f"{curr['Nomos_Score']:.1f}/10")
            
        with c3:
            vol_ratio = curr['SD'] / df['SD'].mean()
            is_stable = vol_ratio < 1.3
            conf_label = "HIGH (Stable)" if is_stable else "LOW (Chaotic)"
            conf_class = "confidence-high" if is_stable else "confidence-low"
            st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-label">Algo Confidence</p>
                    <p class="{conf_class}">{conf_label}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.metric("Volatility Ratio", f"{vol_ratio:.2f}x")

        # 3. ANALYSIS VISUALS
        st.markdown("### Mathematical Trend Architecture")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50']+(sensitivity*df['SD']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50']-(sensitivity*df['SD']), fill='tonexty', fillcolor='rgba(128,128,128,0.08)', line_color='rgba(0,0,0,0)', name='Noise Threshold'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Mathematical Mean', line=dict(color='#FFD700', dash='dot')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown("### 21-Day Risk Probability Cloud")
            mc = run_monte_carlo(curr['Close'], curr['SD'])
            fig_mc = go.Figure()
            p5, p95 = np.percentile(mc, 5, axis=1), np.percentile(mc, 95, axis=1)
            fig_mc.add_trace(go.Scatter(y=p95, line=dict(color='rgba(0,255,0,0.2)', dash='dash'), name='95th %ile'))
            fig_mc.add_trace(go.Scatter(y=p5, line=dict(color='rgba(255,0,0,0.2)', dash='dash'), name='5th %ile'))
            fig_mc.add_trace(go.Scatter(y=mc.mean(axis=1), mode='lines', line=dict(color='#00ff00', width=3), name='Statistical Path'))
            fig_mc.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)

        with col_r:
            st.markdown("### Terminal Intelligence")
            state = "Overbought" if curr['RSI'] > 70 else "Oversold" if curr['RSI'] < 30 else "Neutral"
            st.write(f"The system identifies a **{state}** momentum state. The current price is {round(abs(curr['diff']), 2)} units from the mean.")
            with st.expander("🛡️ HOW TO DECIDE"):
                st.write("""
                - **High Score (>8):** Positive momentum deviation.
                - **Low Score (<4):** Negative momentum deviation.
                - **Wide Cloud:** High statistical uncertainty. Proceed with caution.
                """)

        # 4. COMPLIANT PERFORMANCE PROOF
        with st.expander("🚀 VIEW HISTORICAL STRATEGY PERFORMANCE"):
            df['Sig'] = np.where(df['Nomos_Score'] > 8, 1, np.where(df['Nomos_Score'] < 4, 0, np.nan))
            df['Pos'] = df['Sig'].ffill().fillna(0)
            df['Strat'] = (df['Close'].pct_change() * df['Pos'].shift(1)).add(1).cumprod() - 1
            df['Mkt'] = (df['Close'] / df['Close'].iloc[0]) - 1
            
            b1, b2 = st.columns(2)
            b1.metric("Nomos Strategy", f"{df['Strat'].iloc[-1]*100:.1f}%")
            b2.metric("Market Benchmark", f"{df['Mkt'].iloc[-1]*100:.1f}%")
            st.caption("Strategy logic: Entry at Score > 8 | Exit at Score < 4. Fully synced to Sensitivity.")

        st.divider()
        st.caption("⚠️ **LEGAL:** Mathematical simulation only. No financial advice provided. Not SEBI registered. Trading involves capital risk.")

    else:
        st.error("Ticker not recognized or insufficient historical data.")
