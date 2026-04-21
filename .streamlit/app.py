import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Nomos Terminal | Semi-Precise", layout="wide")

# Institutional Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ NOMOS TERMINAL")
st.caption("Volatility-Adjusted Deterministic Engine | v3.6")

# Sidebar
ticker = st.sidebar.text_input("Global Ticker Search", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity (Std Dev)", 0.5, 3.0, 1.5, step=0.1)

if ticker:
    with st.spinner(f'Analyzing {ticker} DNA...'):
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
    if not df.empty and len(df) >= 50:
        # Math Core
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['SD'] = df['Close'].rolling(window=50).std()
        
        curr_price = float(df['Close'].iloc[-1])
        ma50_val = float(df['MA50'].iloc[-1])
        curr_sd = float(df['SD'].iloc[-1])
        
        # Define Neutral Zone Bounds
        upper_bound = ma50_val + (sensitivity * curr_sd)
        lower_bound = ma50_val - (sensitivity * curr_sd)
        
        # --- SEMI-PRECISE SCORING ---
        # We calculate the position relative to the volatility band
        diff = curr_price - ma50_val
        threshold = sensitivity * curr_sd
        
        # Scale: 5 is center, 10 is upper bound, 1 is lower bound
        raw_score = 5 + (diff / threshold) * 5
        # Rounding to 1 decimal for "Semi-Precision"
        precise_score = round(max(1.0, min(10.0, raw_score)), 1)
        
        # Color and Label Logic
        if precise_score >= 6.0:
            color, label = "#00ff00", "BULLISH"
        elif precise_score <= 4.0:
            color, label = "#ff4b4b", "BEARISH"
        else:
            color, label = "#808080", "NEUTRAL"
            
        # UI - Adaptive Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${curr_price:.2f}")
        c2.metric("Trend (MA50)", f"${ma50_val:.2f}")
        c3.metric("Nomos Score", f"{precise_score}/10")
        c4.metric("Market State", label)

        st.markdown(f"### Sentiment: <span style='color:{color}'>{label} ({precise_score})</span>", unsafe_allow_html=True)
        st.progress(int(precise_score * 10))
        
        # Chart with the "Noise Floor" Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='Trend', line=dict(color='#FFD700', dash='dot')))
        
        # The Shaded Neutral Zone
        fig.add_trace(go.Scatter(
            x=df.index.tolist() + df.index.tolist()[::-1],
            y=(df['MA50'] + (sensitivity * df['SD'])).tolist() + (df['MA50'] - (sensitivity * df['SD'])).tolist()[::-1],
            fill='toself', fillcolor='rgba(128,128,128,0.12)', line=dict(color='rgba(255,255,255,0)'),
            name='Noise Floor'
        ))
        
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 **Analysis:** {ticker} is at {precise_score}/10. The score will flip to Neutral if price moves between **${lower_bound:.2f}** and **${upper_bound:.2f}**.")
    else:
        st.error("Incomplete market data for this ticker.")
