import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- SYSTEM CONFIG & THEME ---
st.set_page_config(page_title="Nomos Terminal | v9.0 Hybrid", layout="wide")

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
    for suffix in ["", ".NS", ".BO"]:
        try:
            data = yf.Ticker(f"{ticker}{suffix}").history(period="2y")
            if not data.empty: return data, f"{ticker}{suffix}"
        except: continue
    return pd.DataFrame(), ticker

# --- MAIN TERMINAL ---
st.title("🏛️ NOMOS TERMINAL")
st.caption("Strategic Intelligence | v9.0 Hybrid Build")

# Sidebar
st.sidebar.title("🏛️ CONTROL")
user_input = st.sidebar.text_input("Asset Search", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5)
st.sidebar.divider()
st.sidebar.info("Escaping the Quant Trap: Balanced UX")

if user_input:
    df, active_ticker = fetch_data(user_input)
    
    if not df.empty and len(df) >= 100:
        # 1. CORE MATH LAYER (Under-the-hood)
        df['MA50'] = df['Close'].rolling(50).mean()
        df['SD'] = df['Close'].rolling(50).std()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df['diff'] = df['Close'] - df['MA50']
        df['Nomos_Score'] = (5 + (df['diff'] / (sensitivity * df['SD'])) * 5 + (df['RSI'] - 50) / 10).clip(1, 10)
        curr = df.iloc[-1]
        
        # 2. THE TOP METRIC BAR (Persistent)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Ticker", active_ticker)
        c2.metric("Nomos Score", f"{curr['Nomos_Score']:.1f}/10")
        
        vol_ratio = curr['SD'] / df['SD'].mean()
        conf_label = "HIGH (Stable)" if vol_ratio < 1.3 else "LOW (Chaotic)"
        conf_class = "confidence-high" if vol_ratio < 1.3 else "confidence-low"
        with c3:
            st.markdown(f'<div class="metric-container"><p class="metric-label">Algo Confidence</p><p class="{conf_class}">{conf_label}</p></div>', unsafe_allow_html=True)
        c4.metric("Volatility Ratio", f"{vol_ratio:.2f}x")

        # 3. THE HYBRID TABS (The Middle Ground)
        tab1, tab2, tab3 = st.tabs(["📉 Market Visuals", "🔮 Risk Projections", "🧠 The Quant Vault"])

        with tab1:
            st.markdown("### Mathematical Trend Architecture")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50']+(sensitivity*df['SD']), line_color='rgba(0,0,0,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50']-(sensitivity*df['SD']), fill='tonexty', fillcolor='rgba(128,128,128,0.08)', line_color='rgba(0,0,0,0)', name='Noise Threshold'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Mathematical Mean', line=dict(color='#FFD700', dash='dot')))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Re-adding Terminal Intelligence summaries here for Tab 1
            st.markdown("### Terminal Intelligence")
            col_l, col_r = st.columns(2)
            with col_l:
                state = "Overbought" if curr['RSI'] > 70 else "Oversold" if curr['RSI'] < 30 else "Neutral"
                st.write(f"The system identifies a **{state}** momentum state.")
                st.write(f"Price is **{round(abs(curr['diff']), 2)}** units from the mean.")
            with col_r:
                regime = "Trending" if abs(curr['diff'] / curr['SD']) > 1.2 else "Stable/Ranging"
                st.write(f"**Market Regime:** {regime}")
                st.info("💡 Tip: Scores above 8.0 indicate positive momentum strength.")

        with tab2:
            st.markdown("### 21-Day Risk Probability Cloud (Monte Carlo)")
            # Quant-grade simulation logic
            days, sims = 21, 1000
            daily_vol = curr['SD'] / curr['Close']
            mc_results = np.zeros((days + 1, sims))
            mc_results[0] = curr['Close']
            for s in range(sims):
                for d in range(1, days + 1):
                    mc_results[d, s] = mc_results[d-1, s] * (1 + np.random.normal(0, daily_vol))
            
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(y=np.percentile(mc_results, 95, axis=1), line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name='95th %ile (Bull Case)'))
            fig_mc.add_trace(go.Scatter(y=np.percentile(mc_results, 5, axis=1), line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name='5th %ile (Bear Case)'))
            fig_mc.add_trace(go.Scatter(y=mc_results.mean(axis=1), mode='lines', line=dict(color='#00ff00', width=3), name='Probable Path'))
            fig_mc.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)
            st.caption("Simulation based on 1,000 statistical iterations of historical standard deviation.")

        with tab3:
            st.markdown("### Institutional Quant Insights")
            # Safe vectorized performance calculation (No ValueError)
            returns = df['Close'].pct_change()
            df['Sig'] = np.where(df['Nomos_Score'] > 8, 1, np.where(df['Nomos_Score'] < 4, 0, np.nan))
            df['Pos'] = df['Sig'].ffill().fillna(0).shift(1) # Shifted to prevent look-ahead bias
            
            # Simulated 0.1% transaction friction
            trades = df['Pos'].diff().abs().fillna(0)
            strat_returns = (returns * df['Pos']) - (trades * 0.001)
            
            if not strat_returns.dropna().empty:
                sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() != 0 else 0
                cum_ret = (1 + strat_returns.dropna()).cumprod()
                mdd = (cum_ret / cum_ret.cummax() - 1).min()
                
                q1, q2, q3 = st.columns(3)
                q1.metric("Sharpe Ratio (Risk Adj.)", f"{sharpe:.2f}")
                q2.metric("Max Strategy Drawdown", f"{mdd*100:.1f}%")
                q3.metric("Annualized Volatility", f"{returns.std() * np.sqrt(252) * 100:.1f}%")
            
            st.divider()
            st.markdown("""
            **Technical Methodology:**
            - **Execution:** Backtest incorporates a 1-day lag to reflect real-world order execution.
            - **Friction:** 10 basis points (0.1%) slippage applied per trade toggle.
            - **Derivation:** Score integrates Z-Score mean reversion with RSI momentum factors.
            """)

        st.divider()
        st.caption("⚠️ **LEGAL:** Mathematical simulation only. No financial advice. Not SEBI registered.")

    else:
        st.error("Asset not recognized or insufficient data for analysis.")
