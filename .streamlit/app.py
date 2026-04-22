import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PRE-FLIGHT CONFIG ---
st.set_page_config(page_title="Nomos Terminal | v5.1", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #30363d; }
    div[data-testid="stExpander"] { border: 1px solid #30363d !important; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE ENGINES ---

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    # SMART SEARCH LOGIC: Handles the .NS problem automatically
    tickers_to_try = [ticker, f"{ticker}.NS", f"{ticker}.BO"]
    for t in tickers_to_try:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="2y")
            if not df.empty:
                return df, t # Return data and the successful ticker
        except:
            continue
    return pd.DataFrame(), ticker

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_monte_carlo(start_price, sd, days=21, sims=100):
    daily_vol = sd / start_price
    sim_results = np.zeros((days + 1, sims))
    sim_results[0] = start_price
    for s in range(sims):
        for d in range(1, days + 1):
            sim_results[d, s] = sim_results[d-1, s] * (1 + np.random.normal(0, daily_vol))
    return sim_results

# --- SIDEBAR CONTROL ---
st.sidebar.title("🏛️ NOMOS CONTROL")
user_input = st.sidebar.text_input("Search (e.g. NVDA, EDELWEISS, RELIANCE)", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5, help="Higher = stricter signals. Lower = more sensitive to noise.")

with st.sidebar.expander("🔍 Search Guide"):
    st.caption("Auto-search supports US and NSE India. For BSE, add .BO manually.")

st.sidebar.divider()
st.sidebar.caption("v5.1 | User-Centric Interface Active")

# --- MAIN TERMINAL ---
st.title("🏛️ NOMOS TERMINAL")

if user_input:
    df, active_ticker = fetch_data(user_input)
        
    if not df.empty and len(df) >= 100:
        # 1. Calculation Layer
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['SD'] = df['Close'].rolling(window=50).std()
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Scoring Logic (Price + Momentum)
        df['diff'] = df['Close'] - df['MA50']
        df['threshold'] = sensitivity * df['SD']
        df['Nomos_Score'] = (5 + (df['diff'] / df['threshold']) * 5 + (df['RSI'] - 50) / 10).clip(1, 10)

        curr = df.iloc[-1]
        score = round(curr['Nomos_Score'], 1)
        
        # 2. Top Metrics Tray
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Ticker", active_ticker)
        c2.metric("Nomos Score", f"{score}/10")
        c3.metric("RSI Momentum", f"{curr['RSI']:.1f}")
        
        # Market Regime Logic
        if curr['SD'] > (df['SD'].mean() * 1.5):
            regime = "⚠️ HIGH VOLATILITY"
        else:
            regime = "✅ STABLE TREND"
        c4.metric("Market Regime", regime)

        # 3. Analysis Chart
        st.markdown(f"### {active_ticker} Trend & Noise Floor")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50']+df['threshold'], fill=None, mode='lines', line_color='rgba(128,128,128,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50']-df['threshold'], fill='tonexty', mode='lines', line_color='rgba(128,128,128,0)', fillcolor='rgba(128,128,128,0.08)', name='Noise Floor'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Trend', line=dict(color='#FFD700', dash='dot')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # 4. Intelligence & Projection
        st.divider()
        cl, cr = st.columns([2, 1])
        
        with cl:
            st.markdown("### 21-Day Risk Projection")
            mc_res = run_monte_carlo(curr['Close'], curr['SD'])
            fig_mc = go.Figure()
            for i in range(mc_res.shape[1]):
                fig_mc.add_trace(go.Scatter(y=mc_res[:, i], mode='lines', line=dict(width=1, color='rgba(0, 255, 150, 0.04)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(y=mc_res.mean(axis=1), mode='lines', line=dict(color='#00ff00', width=3), name='Mean Path'))
            fig_mc.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)

        with cr:
            st.markdown("### Terminal Intelligence")
            
            # Plain Language Summary
            if score >= 6.5:
                status, color = "Strong Bullish Bias", "green"
            elif score <= 3.5:
                status, color = "Heavy Bearish Bias", "red"
            else:
                status, color = "Neutral / Sideways", "grey"
                
            st.markdown(f"**Current Analysis:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
            
            # Contextual Help for your friend
            with st.expander("❓ Help: What am I seeing?"):
                st.write("""
                **Score:** A 1-10 ranking of trend strength.
                **Cloud Chart:** 100 mathematical 'futures'. If the cloud is narrow, risk is low.
                **Backtest (Below):** Proof of historical profit.
                """)

        # 5. Backtester
        with st.expander("🚀 VIEW STRATEGY BACKTEST (NOMOS VS. HOLDING)"):
            df['Pos'] = np.where(df['Nomos_Score'] > 8, 1, np.where(df['Nomos_Score'] < 4, 0, np.nan))
            df['Pos'] = df['Pos'].ffill().fillna(0)
            df['Strat_Ret'] = df['Close'].pct_change() * df['Pos'].shift(1)
            cum_strat = (1 + df['Strat_Ret']).cumprod().iloc[-1] - 1
            cum_mkt = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
            
            b1, b2 = st.columns(2)
            b1.metric("Strategy Return", f"{cum_strat*100:.1f}%")
            b2.metric("Market Return", f"{cum_mkt*100:.1f}%")
    else:
        st.error("Ticker not found or insufficient data. Try adding .NS for Indian stocks.")
