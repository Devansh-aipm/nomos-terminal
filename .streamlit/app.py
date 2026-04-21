import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Nomos Terminal", layout="wide")

# 2. Styling (Professional Terminal Aesthetic)
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ NOMOS TERMINAL")
st.subheader("Deterministic Market Sentiment Engine")

# 3. Sidebar Controls (Your 0.1 Threshold Logic)
st.sidebar.header("Terminal Controls")
ticker = st.sidebar.text_input("Asset Ticker", value="TSLA").upper()
# Converting your 0.1 to a decimal (0.001) for the math
user_threshold = st.sidebar.slider("Neutral Zone (%)", 0.05, 0.50, 0.10, step=0.01)
threshold_decimal = user_threshold / 100

# 4. Data Engine
if ticker:
    with st.spinner(f'Fetching {ticker} data...'):
        df = yf.download(ticker, period="1y")
        
    if not df.empty:
        # Calculate 50-day Moving Average
        df['MA50'] = df['Close'].rolling(window=50).mean()
        curr_price = float(df['Close'].iloc[-1])
        ma50_val = float(df['MA50'].iloc[-1])
        
        # The Deterministic Logic
        diff = (curr_price - ma50_val) / ma50_val
        
        if abs(diff) < threshold_decimal:
            score, label, color = 5, "NEUTRAL", "#808080"
        elif diff > 0:
            score, label, color = 10, "BULLISH", "#00ff00"
        else:
            score, label, color = 1, "BEARISH", "#ff0000"
            
        # 5. Display Interface
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${curr_price:.2f}")
        col2.metric("50-Day MA", f"${ma50_val:.2f}")
        col3.metric("Nomos Score", f"{score}/10")

        st.markdown(f"### Sentiment: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        
        # 6. Interactive Charting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-Day MA', line=dict(color='#FFD700', dash='dot')))
        fig.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Invalid Ticker. Check spelling or market availability.")

st.sidebar.write("---")
st.sidebar.caption("Institutional-grade synthesis. No noise. Pure Nomos.")
