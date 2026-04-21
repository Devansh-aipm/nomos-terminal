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
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ NOMOS TERMINAL")
st.caption("Deterministic Market Sentiment Engine | v1.5")

# Sidebar
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Ticker Symbol", value="TSLA").upper()
user_threshold = st.sidebar.slider("Neutral Zone (%)", 0.05, 0.50, 0.10, step=0.01)
threshold_decimal = user_threshold / 100

if ticker:
    with st.spinner('Accessing Market Data...'):
        data = yf.Ticker(ticker)
        df = data.history(period="1y")
        
    if not df.empty and len(df) >= 50:
        df['MA50'] = df['Close'].rolling(window=50).mean()
        curr_price = float(df['Close'].iloc[-1])
        ma50_val = float(df['MA50'].iloc[-1])
        diff = (curr_price - ma50_val) / ma50_val
        
        # Logic for Score & Interpretation
        if abs(diff) < threshold_decimal:
            score, label, color, note = 5, "NEUTRAL", "#808080", "Price is consolidating within the trend zone."
        elif diff > 0:
            score, label, color, note = 10, "BULLISH", "#00ff00", f"Price is {(diff*100):.1f}% above the 50-day average."
        else:
            score, label, color, note = 1, "BEARISH", "#ff0000", f"Price is {abs(diff*100):.1f}% below the 50-day average."
            
        # UI - Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${curr_price:.2f}")
        c2.metric("50-Day MA", f"${ma50_val:.2f}")
        c3.metric("Nomos Score", f"{score}/10")
        c4.metric("52W High", f"${df['High'].max():.2f}")

        st.markdown(f"### Sentiment: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.info(f"**Analysis:** {note}")
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-Day MA', line=dict(color='#FFD700', dash='dot')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Recent Data Table
        st.subheader("Recent Tape")
        st.dataframe(df[['Open', 'High', 'Low', 'Close']].tail(5).sort_index(ascending=False), use_container_width=True)
    else:
        st.error("Invalid Ticker or Insufficient Data.")

st.sidebar.caption("v1.5 Final Stable Build")
