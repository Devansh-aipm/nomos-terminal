import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Nomos Terminal | Logic-First", layout="wide")

# Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ NOMOS TERMINAL")
st.caption("Volatility-Adjusted Deterministic Engine | v3.0")

# Sidebar
ticker = st.sidebar.text_input("Asset Ticker", value="AAPL").upper()
# 1.5 SD is a standard statistical threshold for 'Significant' moves
sensitivity = st.sidebar.slider("Signal Sensitivity (Std Dev)", 0.5, 3.0, 1.5, step=0.1)

if ticker:
    with st.spinner(f'Calculating {ticker} Volatility...'):
        data = yf.Ticker(ticker)
        df = data.history(period="1y")
        
    if not df.empty and len(df) >= 50:
        # Step 1: The Trend (50-Day Moving Average)
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Step 2: The Noise (Rolling Standard Deviation)
        # This calculates how 'wild' the stock has been over the last 50 days
        df['SD'] = df['Close'].rolling(window=50).std()
        
        curr_price = float(df['Close'].iloc[-1])
        ma50_val = float(df['MA50'].iloc[-1])
        curr_sd = float(df['SD'].iloc[-1])
        
        # Step 3: Define the Mathematical 'Neutral Zone'
        # Any price inside this range is considered 'Noise'
        upper_bound = ma50_val + (sensitivity * curr_sd)
        lower_bound = ma50_val - (sensitivity * curr_sd)
        
        # Step 4: Deterministic Scoring
        if curr_price > upper_bound:
            score, label, color = 10, "BULLISH", "#00ff00"
        elif curr_price < lower_bound:
            score, label, color = 1, "BEARISH", "#ff0000"
        else:
            score, label, color = 5, "NEUTRAL", "#808080"
            
        # Display Results
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${curr_price:.2f}")
        c2.metric("Trend (MA50)", f"${ma50_val:.2f}")
        c3.metric("Volatility (SD)", f"${curr_sd:.2f}")
        c4.metric("Nomos Score", f"{score}/10")

        st.markdown(f"### Sentiment: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        
        # Chart with Shaded 'Noise' Zone
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Trend', line=dict(color='#FFD700', dash='dot')))
        
        # Shaded area representing the 'Neutral Zone'
        fig.add_trace(go.Scatter(
            x=df.index.tolist() + df.index.tolist()[::-1],
            y=(df['MA50'] + (sensitivity * df['SD'])).tolist() + (df['MA50'] - (sensitivity * df['SD'])).tolist()[::-1],
            fill='toself', fillcolor='rgba(128,128,128,0.2)', line=dict(color='rgba(255,255,255,0)'),
            name='Noise Floor'
        ))
        
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Mathematical Logic:** Based on {ticker}'s recent volatility, the price must exceed **${upper_bound:.2f}** to be Bullish or drop below **${lower_bound:.2f}** to be Bearish. Anything in between is statistically insignificant.")
    else:
        st.error("Invalid ticker or insufficient data.")
