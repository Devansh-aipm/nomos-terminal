import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Nomos Terminal", layout="wide")

# Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; border-left: 5px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ NOMOS TERMINAL")
st.caption("Deterministic Market Sentiment Engine | v2.0")

# Sidebar
st.sidebar.header("Terminal Controls")
ticker = st.sidebar.text_input("Ticker Symbol", value="TSLA").upper()
user_threshold = st.sidebar.slider("Neutral Zone (%)", 0.05, 0.50, 0.10, step=0.01)
threshold_decimal = user_threshold / 100

if ticker:
    with st.spinner('Synchronizing Data...'):
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
    if not df.empty and len(df) >= 50:
        df['MA50'] = df['Close'].rolling(window=50).mean()
        curr_price = float(df['Close'].iloc[-1])
        ma50_val = float(df['MA50'].iloc[-1])
        diff = (curr_price - ma50_val) / ma50_val
        
        # Calculate Volume Strength
        avg_vol = df['Volume'].tail(10).mean()
        curr_vol = df['Volume'].iloc[-1]
        vol_ratio = curr_vol / avg_vol

        # Scoring Logic
        if abs(diff) < threshold_decimal:
            score, label, color = 5, "NEUTRAL", "#808080"
        elif diff > 0:
            score, label, color = 10, "BULLISH", "#00ff00"
        else:
            score, label, color = 1, "BEARISH", "#ff0000"
            
        # UI - Professional Metric Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${curr_price:.2f}")
        c2.metric("50-Day MA", f"${ma50_val:.2f}")
        c3.metric("Volume Strength", f"{vol_ratio:.1f}x")
        c4.metric("Nomos Score", f"{score}/10")

        # Signal Bar
        st.markdown(f"### Sentiment: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.progress(score * 10) # Visualizes the 1-10 score as a bar
        
        # Target Price Logic (What price flips the signal?)
        flip_price = ma50_val * (1 + threshold_decimal) if diff < 0 else ma50_val * (1 - threshold_decimal)
        st.write(f"💡 **Pivot Point:** Asset becomes Neutral at **${flip_price:.2f}**")

        # Main Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-Day MA', line=dict(color='#FFD700', dash='dot')))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # The "Tape"
        with st.expander("View Recent Tape"):
            st.table(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5).sort_index(ascending=False))
    else:
        st.error("Data Unavailable for this ticker.")

st.sidebar.info(f"Target: {ticker} | Mode: Deterministic")
