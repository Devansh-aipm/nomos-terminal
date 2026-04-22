import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PRE-FLIGHT CONFIG ---
st.set_page_config(page_title="Nomos Terminal | Predictive", layout="wide")

# Institutional Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #30363d; }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE ENGINES ---

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        return df
    except Exception:
        return pd.DataFrame()

# Deterministic RSI Logic (No AI, Pure Math)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_monte_carlo(start_price, sd, days=21, sims=100):
    daily_vol = sd / start_price
    simulation_df = pd.DataFrame()
    for i in range(sims):
        prices = [start_price]
        for _ in range(days):
            change = np.random.normal(0, daily_vol)
            prices.append(prices[-1] * (1 + change))
        simulation_df[f"Sim_{i}"] = prices
    return simulation_df

# --- SIDEBAR CONTROL ---
st.sidebar.title("🏛️ NOMOS CONTROL")
ticker = st.sidebar.text_input("Global Ticker Search", value="NVDA").upper()
sensitivity = st.sidebar.slider("Signal Sensitivity (Std Dev)", 0.5, 3.0, 1.5, step=0.1)
st.sidebar.divider()
st.sidebar.caption("v4.5 | Momentum Engine Active")

# --- MAIN TERMINAL ---
st.title("🏛️ NOMOS TERMINAL")

if ticker:
    df = fetch_data(ticker)
        
    if not df.empty and len(df) >= 50:
        # 1. Calculation Layer
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['SD'] = df['Close'].rolling(window=50).std()
        df['RSI'] = calculate_rsi(df['Close']) # Adding Momentum
        
        curr_price = float(df['Close'].iloc[-1])
        ma50_val = float(df['MA50'].iloc[-1])
        curr_sd = float(df['SD'].iloc[-1])
        curr_rsi = float(df['RSI'].iloc[-1])
        
        upper_bound = ma50_val + (sensitivity * curr_sd)
        lower_bound = ma50_val - (sensitivity * curr_sd)
        
        # --- MULTI-FACTOR DETERMINISTIC SCORING ---
        diff = curr_price - ma50_val
        threshold = sensitivity * curr_sd
        
        # Weighting: 70% Trend, 30% Momentum
        trend_score = (diff / threshold) * 5
        momentum_score = (curr_rsi - 50) / 10 
        
        raw_score = 5 + trend_score + momentum_score
        precise_score = round(max(1.0, min(10.0, raw_score)), 1)
        
        if precise_score >= 6.0:
            color, label = "#00ff00", "BULLISH"
        elif precise_score <= 4.0:
            color, label = "#ff4b4b", "BEARISH"
        else:
            color, label = "#808080", "NEUTRAL"

        # 2. Top Metrics Tray
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${curr_price:.2f}")
        c2.metric("Nomos Score", f"{precise_score}/10")
        c3.metric("RSI (14D)", f"{curr_rsi:.1f}")
        c4.metric("Market State", label)

        # 3. Primary Analysis Chart
        st.markdown(f"### Historical Trend & Noise Floor")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.tolist() + df.index.tolist()[::-1],
            y=(df['MA50'] + (sensitivity * df['SD'])).tolist() + (df['MA50'] - (sensitivity * df['SD'])).tolist()[::-1],
            fill='toself', fillcolor='rgba(128,128,128,0.1)', line=dict(color='rgba(255,255,255,0)'),
            name='Noise Floor'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FAFAFA', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50 Trend', line=dict(color='#FFD700', dash='dot')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 4. PREDICTIVE & LOGIC LAYER
        st.divider()
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 21-Day Risk Projection")
            mc_data = run_monte_carlo(curr_price, curr_sd)
            fig_mc = go.Figure()
            for col in mc_data.columns:
                fig_mc.add_trace(go.Scatter(
                    y=mc_data[col], mode='lines',
                    line=dict(width=1, color='rgba(0, 255, 150, 0.05)'),
                    showlegend=False
                ))
            fig_mc.add_trace(go.Scatter(
                y=mc_data.mean(axis=1), mode='lines',
                line=dict(color='#00ff00', width=3),
                name='Mean path'
            ))
            fig_mc.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)

        with col_right:
            st.markdown("### Terminal Intelligence")
            st.write(f"**Current Bias:** {label}")
            confidence = round(abs(precise_score - 5) * 20, 1)
            st.write(f"**Signal Strength:** {confidence}%")
            
            # Mathematical RSI Warnings
            if curr_rsi > 70:
                st.warning("⚠️ **OVERBOUGHT:** Price velocity is unsustainably high. Potential pullback risk.")
            elif curr_rsi < 30:
                st.success("✨ **OVERSOLD:** Price velocity is bottoming. Potential recovery zone.")
            else:
                st.info("📊 **STABLE:** Momentum is currently within the healthy range.")
                
            st.info(f"""
            **Thresholds:**
            - Bullish above: ${upper_bound:.2f}
            - Bearish below: ${lower_bound:.2f}
            """)

    else:
        st.error("Incomplete market data for this ticker.")
