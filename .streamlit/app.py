import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PRE-FLIGHT CONFIG ---
st.set_page_config(page_title="Nomos Terminal | v5.0", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINES ---

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y") # 2 years for better backtesting
        return df
    except Exception:
        return pd.DataFrame()

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

# --- SIDEBAR ---
st.sidebar.title("🏛️ NOMOS CONTROL")
ticker = st.sidebar.text_input("Global Ticker Search", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5, step=0.1)
st.sidebar.divider()
st.sidebar.caption("v5.0 | Institutional Backtester Active")

# --- MAIN TERMINAL ---
st.title("🏛️ NOMOS TERMINAL")

if ticker:
    df = fetch_data(ticker)
        
    if not df.empty and len(df) >= 100:
        # 1. Math Layer
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['SD'] = df['Close'].rolling(window=50).std()
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Scoring Logic
        df['diff'] = df['Close'] - df['MA50']
        df['threshold'] = sensitivity * df['SD']
        df['Trend_Score'] = (df['diff'] / df['threshold']) * 5
        df['Mom_Score'] = (df['RSI'] - 50) / 10
        df['Nomos_Score'] = (5 + df['Trend_Score'] + df['Mom_Score']).clip(1, 10)

        curr = df.iloc[-1]
        
        # 2. Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${curr['Close']:.2f}")
        c2.metric("Nomos Score", f"{curr['Nomos_Score']:.1f}/10")
        c3.metric("RSI (14D)", f"{curr['RSI']:.1f}")
        state = "BULLISH" if curr['Nomos_Score'] > 6 else "BEARISH" if curr['Nomos_Score'] < 4 else "NEUTRAL"
        c4.metric("Market State", state)

        # 3. Main Chart
        st.markdown("### Historical Trend & Noise Floor")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50']+df['threshold'], fill=None, mode='lines', line_color='rgba(128,128,128,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50']-df['threshold'], fill='tonexty', mode='lines', line_color='rgba(128,128,128,0)', fillcolor='rgba(128,128,128,0.1)', name='Noise Floor'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Trend', line=dict(color='#FFD700', dash='dot')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # 4. Prediction & Intelligence
        st.divider()
        cl, cr = st.columns([2, 1])
        
        with cl:
            st.markdown("### 21-Day Risk Projection")
            mc_res = run_monte_carlo(curr['Close'], curr['SD'])
            fig_mc = go.Figure()
            for i in range(mc_res.shape[1]):
                fig_mc.add_trace(go.Scatter(y=mc_res[:, i], mode='lines', line=dict(width=1, color='rgba(0, 255, 150, 0.05)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(y=mc_res.mean(axis=1), mode='lines', line=dict(color='#00ff00', width=3), name='Mean path'))
            fig_mc.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)

        with cr:
            st.markdown("### Terminal Intelligence")
            st.write(f"**Signal Strength:** {round(abs(curr['Nomos_Score'] - 5) * 20, 1)}%")
            if curr['RSI'] > 70: st.warning("⚠️ OVERBOUGHT: Pullback risk.")
            elif curr['RSI'] < 30: st.success("✨ OVERSOLD: Recovery zone.")
            else: st.info("📊 STABLE: Normal momentum.")

        # 5. STEP 3: THE INSTITUTIONAL BACKTESTER
        with st.expander("🚀 VIEW STRATEGY BACKTEST (HISTORICAL PERFORMANCE)"):
            st.markdown("#### Logic: Buy at Score > 8.0 | Sell at Score < 4.0")
            
            # Simple Backtest Logic
            df['Signal'] = 0
            df.loc[df['Nomos_Score'] > 8, 'Signal'] = 1
            df.loc[df['Nomos_Score'] < 4, 'Signal'] = 0
            df['Position'] = df['Signal'].shift(1).fillna(0)
            
            # Calculate Returns
            df['Market_Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = df['Market_Returns'] * df['Position']
            
            cum_market = (1 + df['Market_Returns']).cumprod().iloc[-1] - 1
            cum_strat = (1 + df['Strategy_Returns']).cumprod().iloc[-1] - 1
            
            b1, b2 = st.columns(2)
            b1.metric("Nomos Strategy Return", f"{cum_strat*100:.2f}%")
            b2.metric("Buy & Hold Return", f"{cum_market*100:.2f}%")
            
            if cum_strat > cum_market:
                st.success(f"✅ Nomos outperformed the market by {(cum_strat - cum_market)*100:.2f}%")
            else:
                st.warning(f"⚠️ Market outperformed the strategy. Consider adjusting Sensitivity.")

    else:
        st.error("Need more historical data for analysis.")
