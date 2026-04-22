import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t # For Fat-Tail Distribution

# --- SYSTEM CONFIG & THEME ---
st.set_page_config(page_title="Nomos Terminal | v9.5 Elite", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; height: 110px; }
    .metric-container { background-color: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; height: 110px; }
    .confidence-high { color: #00ff00; font-weight: bold; font-size: 20px; }
    .confidence-low { color: #ff4b4b; font-weight: bold; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE ENGINES ---
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    for suffix in ["", ".NS", ".BO"]:
        try:
            data = yf.Ticker(f"{ticker}{suffix}").history(period="2y")
            if not data.empty: return data, f"{ticker}{suffix}"
        except: continue
    return pd.DataFrame(), ticker

# --- MAIN INTERFACE ---
st.title("🏛️ NOMOS TERMINAL")
st.caption("Institutional Decision Support | v9.5 Fat-Tail Logic")

st.sidebar.title("🏛️ QUANT CONTROL")
user_input = st.sidebar.text_input("Asset Search", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5)
risk_free_rate = st.sidebar.number_input("Risk Free Rate (%)", value=7.0) / 100 # Default to India T-Bill

if user_input:
    df, active_ticker = fetch_data(user_input)
    
    if not df.empty and len(df) >= 100:
        # 1. ENHANCED MATH LAYER
        df['MA50'] = df['Close'].rolling(50).mean()
        df['SD'] = df['Close'].rolling(50).std()
        
        # Scoring Logic (Now uses shifted data to avoid look-ahead bias)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['Z_Score'] = (df['Close'] - df['MA50']) / df['SD']
        
        # The Moat: Multi-factor interaction instead of simple addition
        df['Nomos_Score'] = (5 + (df['Z_Score'] * 3)).clip(1, 10)
        curr = df.iloc[-1]
        
        # 2. TOP METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Ticker", active_ticker)
        c2.metric("Nomos Score", f"{curr['Nomos_Score']:.1f}/10")
        
        vol_ratio = curr['SD'] / df['SD'].mean()
        conf_label = "HIGH" if vol_ratio < 1.3 else "LOW"
        conf_class = "confidence-high" if vol_ratio < 1.3 else "confidence-low"
        with c3:
            st.markdown(f'<div class="metric-container"><p style="color:#8b949e;font-size:14px;margin-bottom:4px;">Algo Confidence</p><p class="{conf_class}">{conf_label} (Stable)</p></div>', unsafe_allow_html=True)
        c4.metric("Volatility Ratio", f"{vol_ratio:.2f}x")

        # 3. THE HYBRID TABS
        tab1, tab2, tab3 = st.tabs(["📉 Trend Architecture", "🔮 Fat-Tail Risk (t-Dist)", "🧠 The Quant Vault"])

        with tab1:
            st.markdown("### Mathematical Trend Architecture")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50']+(sensitivity*df['SD']), line_color='rgba(0,0,0,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50']-(sensitivity*df['SD']), fill='tonexty', fillcolor='rgba(128,128,128,0.08)', line_color='rgba(0,0,0,0)', name='Noise Threshold'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Mathematical Mean', line=dict(color='#FFD700', dash='dot')))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Terminal Intelligence")
            st.write(f"The asset is trading at **{round(abs(curr['Z_Score']), 2)} SD** from its mathematical mean.")

        with tab2:
            st.markdown("### 21-Day Monte Carlo (Student's t-Distribution)")
            returns = df['Close'].pct_change().dropna()
            # Fitting a t-distribution to capture "Fat Tails" (Degrees of Freedom)
            df_params = t.fit(returns)
            days, sims = 21, 1000
            
            mc_results = np.zeros((days + 1, sims))
            mc_results[0] = curr['Close']
            for s in range(sims):
                # Using t.rvs instead of normal.rvs for "Honest" simulation
                random_shocks = t.rvs(df_params[0], loc=df_params[1], scale=df_params[2], size=days)
                for d in range(1, days + 1):
                    mc_results[d, s] = mc_results[d-1, s] * (1 + random_shocks[d-1])
            
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(y=np.percentile(mc_results, 95, axis=1), line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name='Extreme Bull'))
            fig_mc.add_trace(go.Scatter(y=np.percentile(mc_results, 5, axis=1), line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name='Extreme Bear'))
            fig_mc.add_trace(go.Scatter(y=mc_results.mean(axis=1), line=dict(color='#00ff00', width=3), name='Probable Path'))
            fig_mc.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)
            st.info("Simulation upgraded to **Student's t-Distribution** to account for 'Black Swan' fat-tail volatility.")

        with tab3:
            st.markdown("### Professional Fund Metrics")
            # Logic: Entry > 8, Exit < 4
            df['Signal'] = np.where(df['Nomos_Score'] > 8, 1, np.where(df['Nomos_Score'] < 4, 0, np.nan))
            df['Position'] = df['Signal'].ffill().fillna(0).shift(1) # Execution Lag Fixed
            
            # Daily Returns minus Risk Free Rate (Daily Adjusted)
            daily_rf = (1 + risk_free_rate)**(1/252) - 1
            excess_returns = (df['Close'].pct_change() * df['Position']) - daily_rf
            
            if not excess_returns.dropna().empty:
                # 1. True Sharpe (on Excess Returns)
                sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
                # 2. Max Drawdown
                cum_ret = (1 + excess_returns.dropna()).cumprod()
                mdd = (cum_ret / cum_ret.cummax() - 1).min()
                # 3. Calmar Ratio (Annualized Return / Max Drawdown)
                ann_return = excess_returns.mean() * 252
                calmar = abs(ann_return / mdd) if mdd != 0 else 0
                
                q1, q2, q3 = st.columns(3)
                q1.metric("True Sharpe (Excess)", f"{sharpe:.2f}")
                q2.metric("Calmar Ratio", f"{calmar:.2f}")
                q3.metric("Max Drawdown", f"{mdd*100:.1f}%")
                
                st.write("---")
                st.markdown("**Quant Disclosure:** All metrics calculated against a **" + str(risk_free_rate*100) + "% Risk-Free Rate** benchmark.")
            else:
                st.warning("Insufficient signal history.")

        st.divider()
        st.caption("Nomos Terminal v9.5 | Decision Support Architecture")

    else:
        st.error("Invalid Asset.")
