import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t as t_dist

# ============================================================
# NOMOS TERMINAL v12.0
# Global multi-factor decision architecture
# ============================================================

st.set_page_config(
    page_title="Nomos Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Nomos Terminal v12.0 — Institutional Decision Architecture"}
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #05070A;
    color: #D4D8E1;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #080B0F !important;
    border-right: 1px solid #141820 !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: #080B0F !important;
    border: 1px solid #141820 !important;
    border-top: 2px solid #C8A84B !important;
    border-radius: 6px !important;
    padding: 18px 16px 14px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover { border-color: #C8A84B !important; }
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 500 !important;
    color: #E8EAF0 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #5A6070 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #141820 !important;
    gap: 0 !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #5A6070 !important;
    border-radius: 0 !important;
    padding: 10px 22px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: #C8A84B !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #C8A84B !important;
    border-bottom: 2px solid #C8A84B !important;
    background: transparent !important;
}

/* ── INPUTS ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #080B0F !important;
    border: 1px solid #1E2430 !important;
    border-radius: 6px !important;
    color: #E8EAF0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #C8A84B !important;
    box-shadow: 0 0 0 2px rgba(200,168,75,0.12) !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] [role="slider"] { background: #C8A84B !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: #080B0F !important;
    border: 1px solid #141820 !important;
    border-radius: 6px !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: #C8A84B !important; }

/* ── CUSTOM COMPONENTS ── */
.nomos-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 4px;
}
.nomos-wordmark {
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #E8EAF0;
    line-height: 1;
}
.nomos-version {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #C8A84B;
    letter-spacing: 0.12em;
    font-weight: 400;
    border: 1px solid #C8A84B;
    padding: 2px 7px;
    border-radius: 3px;
}
.nomos-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #5A6070;
    letter-spacing: 0.06em;
    font-weight: 500;
    margin-top: 4px;
}

.signal-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    padding: 4px 12px;
    border-radius: 3px;
    border: 1px solid;
    margin-right: 6px;
    margin-bottom: 4px;
}

.stat-card {
    background: #080B0F;
    border: 1px solid #141820;
    border-radius: 6px;
    padding: 18px;
    margin-bottom: 10px;
    transition: border-color 0.2s ease;
}
.stat-card:hover { border-color: #1E2430; }
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5A6070;
    font-weight: 600;
    margin: 0 0 6px 0;
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #E8EAF0;
    margin: 0 0 8px 0;
    line-height: 1;
}
.stat-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #5A6070;
    margin: 4px 0 0 0;
}

.gauge-wrap {
    background: #080B0F;
    border: 1px solid #141820;
    border-radius: 6px;
    padding: 20px;
}
.gauge-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5A6070;
    font-weight: 600;
    margin-bottom: 12px;
}
.gauge-track {
    background: #141820;
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 12px;
}
.gauge-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #C0392B 0%, #E67E22 35%, #C8A84B 60%, #27AE60 100%);
}
.gauge-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    text-align: center;
    line-height: 1;
    margin: 4px 0;
}
.gauge-verdict {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 700;
    text-align: center;
    margin-top: 6px;
}
.gauge-legend {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #3A4050;
    margin-top: 6px;
}

.score-guide {
    background: #080B0F;
    border: 1px solid #141820;
    border-left: 3px solid #C8A84B;
    border-radius: 0 6px 6px 0;
    padding: 14px 16px;
    margin-bottom: 16px;
}
.score-guide p { margin: 3px 0; font-size: 0.78rem; color: #8A9099; font-family: 'Inter', sans-serif; }
.score-guide span { font-family: 'IBM Plex Mono', monospace; font-weight: 500; }

.landing-card {
    background: #080B0F;
    border: 1px solid #141820;
    border-radius: 8px;
    padding: 32px;
    text-align: center;
    margin: 40px auto;
    max-width: 600px;
}
.landing-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #E8EAF0;
    letter-spacing: -0.02em;
    margin-bottom: 12px;
}
.landing-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #5A6070;
    line-height: 1.7;
    margin-bottom: 24px;
}
.landing-steps {
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.landing-step {
    background: #0D1018;
    border: 1px solid #1E2430;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #8A9099;
    text-align: center;
    min-width: 130px;
}
.landing-step strong {
    display: block;
    color: #C8A84B;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.tooltip-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 2px;
}
.tooltip-term {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #C8A84B;
    min-width: 80px;
}
.tooltip-def {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #5A6070;
}

.disclaimer-bar {
    background: #080B0F;
    border: 1px solid #141820;
    border-left: 2px solid #C8A84B;
    border-radius: 0 4px 4px 0;
    padding: 10px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #3A4050;
    line-height: 1.6;
    margin-bottom: 16px;
}

.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3A4050;
    border-bottom: 1px solid #141820;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.consistency-block {
    border-radius: 6px;
    padding: 14px 18px;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Ensure inputs are always interactive */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    pointer-events: auto !important;
    position: relative !important;
    z-index: 999 !important;
}

/* Hide default streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── CURRENCY DETECTION ───────────────────────────────────────────────────────

CURRENCY_MAP = {
    ".NS": ("₹", "INR"),  ".BO": ("₹", "INR"),
    ".L":  ("£", "GBP"),  ".IL": ("£", "GBP"),
    ".DE": ("€", "EUR"),  ".PA": ("€", "EUR"),  ".MI": ("€", "EUR"),
    ".AS": ("€", "EUR"),  ".MC": ("€", "EUR"),  ".BR": ("€", "EUR"),
    ".HK": ("HK$", "HKD"), ".T":  ("¥", "JPY"),
    ".SS": ("¥", "CNY"),  ".SZ": ("¥", "CNY"),
    ".AX": ("A$", "AUD"), ".TO": ("C$", "CAD"),
    ".KS": ("₩", "KRW"),
}

DEFAULT_RISK_FREE = {
    "INR": 7.0, "GBP": 4.5, "EUR": 3.5, "JPY": 0.5,
    "HKD": 4.8, "AUD": 4.35, "CAD": 4.75, "KRW": 3.5,
    "USD": 5.25,
}

def get_currency(ticker):
    for suffix, (sym, code) in CURRENCY_MAP.items():
        if ticker.upper().endswith(suffix):
            return sym, code
    return "$", "USD"

def fmt_price(val, sym):
    # JPY and KRW are whole-number currencies — no decimal places
    if sym in ("¥", "₩"):
        return f"{sym}{val:,.0f}"
    return f"{sym}{val:,.2f}"

# ─── DATA LAYER ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def fetch_data(ticker):
    """Try exact ticker first, then regional suffixes. Returns (df, resolved_ticker, info_dict)."""
    candidates = [ticker]
    # Only add Indian suffixes if user did not specify a suffix
    if "." not in ticker:
        candidates += [f"{ticker}.NS", f"{ticker}.BO"]
    for tk in candidates:
        try:
            obj  = yf.Ticker(tk)
            # Try download() first (more reliable on cloud), fall back to history()
            data = yf.download(tk, period="3y", auto_adjust=True, progress=False)
            if data.empty or len(data) < 150:
                data = obj.history(period="3y")
            # Flatten MultiIndex columns if present (download() returns them)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and len(data) >= 150:
                try:
                    raw_info = obj.fast_info
                    info = {k: getattr(raw_info, k, None) for k in [
                        "last_price", "market_cap", "fifty_two_week_high",
                        "fifty_two_week_low", "currency", "exchange"
                    ]}
                except Exception:
                    info = {}
                return data, tk, info
        except Exception:
            continue
    return pd.DataFrame(), ticker, {}

# ─── INDICATOR ENGINES ────────────────────────────────────────────────────────

def compute_indicators(df, sensitivity):
    df = df.copy()

    # Trend
    df['MA50']    = df['Close'].rolling(50).mean()
    df['MA200']   = df['Close'].rolling(200).mean()
    df['SD']      = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['MA50']) / df['SD']

    # RSI-14
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['Signal_Line']

    # ATR-14
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    df['ATR']     = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close']

    # Nomos Multi-Factor Score
    z_comp     = (df['Z_Score'] / sensitivity).clip(-1, 1) * 3
    rsi_comp   = ((df['RSI'] - 50) / 50).clip(-1, 1) * 1.5
    macd_norm  = (df['MACD_Hist'] / df['Close'].replace(0, np.nan)).clip(-0.01, 0.01) / 0.01
    macd_comp  = macd_norm * 1.0
    trend_comp = np.where(df['MA50'] > df['MA200'], 0.5, -0.5)
    df['Nomos_Score'] = (5 + z_comp + rsi_comp + macd_comp + trend_comp).clip(1, 10)

    return df

# ─── MONTE CARLO ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def run_monte_carlo(current_price, returns_tuple, days=21, sims=500):
    returns_series = pd.Series(returns_tuple)
    params = t_dist.fit(returns_series.dropna())
    df_t, loc_t, scale_t = params
    shocks = t_dist.rvs(df_t, loc=loc_t, scale=scale_t, size=(days, sims))
    paths  = np.zeros((days + 1, sims))
    paths[0] = current_price
    for d in range(1, days + 1):
        paths[d] = paths[d - 1] * (1 + shocks[d - 1])
    return paths

# ─── VaR / CVaR ───────────────────────────────────────────────────────────────

def compute_var_cvar(returns, confidence=0.95):
    clean = returns.dropna()
    var   = -np.percentile(clean, (1 - confidence) * 100)
    cvar  = -clean[clean <= -var].mean()
    return var, cvar

# ─── WALK-FORWARD BACKTEST ────────────────────────────────────────────────────

def walk_forward_backtest(df, risk_free_rate, n_folds=5):
    results  = []
    data     = df.dropna(subset=['Nomos_Score', 'Close']).copy()
    fold_size = len(data) // n_folds

    for i in range(n_folds):
        fold = data.iloc[i * fold_size: (i + 1) * fold_size]
        mid  = len(fold) // 2
        test = fold.iloc[mid:].copy()

        test['Signal']   = np.where(test['Nomos_Score'] > 7.5, 1,
                           np.where(test['Nomos_Score'] < 4.0, 0, np.nan))
        test['Position'] = test['Signal'].ffill().fillna(0).shift(1)

        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        ret      = test['Close'].pct_change()
        trades   = test['Position'].diff().abs().fillna(0)
        strat_r  = (ret * test['Position']) - (trades * 0.001) - daily_rf

        if strat_r.dropna().empty:
            continue

        cum    = (1 + strat_r.dropna()).cumprod()
        sharpe = (strat_r.mean() / strat_r.std()) * np.sqrt(252) if strat_r.std() != 0 else 0
        mdd    = (cum / cum.cummax() - 1).min()
        ann_r  = strat_r.mean() * 252
        calmar = abs(ann_r / mdd) if mdd != 0 else 0
        win_r  = (strat_r.dropna() > 0).mean()

        results.append({
            'fold': i + 1, 'sharpe': sharpe, 'mdd': mdd,
            'calmar': calmar, 'ann_return': ann_r,
            'win_rate': win_r, 'cum_returns': cum
        })

    return results

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────

PLOT_BG   = '#05070A'
PAPER_BG  = '#05070A'
GRID_COL  = '#0E1118'
GOLD      = '#C8A84B'
BLUE      = '#4A90D9'
GREEN     = '#2ECC71'
RED       = '#E74C3C'
MUTED     = '#3A4050'

def base_layout(height=380, xtitle=None, ytitle=None):
    return dict(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        height=height,
        margin=dict(l=0, r=0, t=8, b=0),
        font=dict(family="Inter, sans-serif", color="#8A9099", size=11),
        xaxis=dict(
            gridcolor=GRID_COL, gridwidth=1, zeroline=False,
            title=xtitle, title_font=dict(size=10, color=MUTED),
            tickfont=dict(family="IBM Plex Mono", size=10)
        ),
        yaxis=dict(
            gridcolor=GRID_COL, gridwidth=1, zeroline=False,
            title=ytitle, title_font=dict(size=10, color=MUTED),
            tickfont=dict(family="IBM Plex Mono", size=10)
        ),
        legend=dict(
            orientation='h', y=1.08, x=0,
            font=dict(family="Inter", size=10),
            bgcolor='rgba(0,0,0,0)'
        )
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Inter', sans-serif;font-size:1.3rem;font-weight:800;letter-spacing:-0.03em;color:#E8EAF0;">NOMOS &nbsp;v12.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Asset Search**")
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "NVDA"
    user_input = st.text_input(
        "Asset Search",
        value=st.session_state["ticker"],
        placeholder="e.g. NVDA, RELIANCE.NS, HSBA.L",
        help="Enter any global ticker. Add .NS for NSE, .L for London, .DE for Frankfurt, etc.",
        label_visibility="collapsed",
        key="ticker_input"
    )
    user_input = (user_input or "").strip().upper()
    if user_input:
        st.session_state["ticker"] = user_input

    sensitivity = st.slider(
        "Signal Sensitivity",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1,
        help="Lower = only strong signals trigger. Higher = more frequent signals."
    )

    rf_label = "Risk-Free Rate (%)"
    # Default to USD rate; will show a banner to update if currency differs
    rf_default = float(st.session_state.get("rf_default", 5.25))
    risk_free_rate = st.number_input(
        rf_label,
        value=rf_default,
        min_value=0.0,
        max_value=20.0,
        step=0.25,
        format="%.2f",
        help="Enter your local benchmark rate. US: ~5.25% | India: ~7% | EU: ~3.5% | UK: ~4.5%"
    ) / 100

    mc_sims = st.slider(
        "MC Simulations",
        min_value=250, max_value=5000, value=2000, step=250,
        help="More simulations = more accurate projections, but slower."
    )

    st.divider()
    st.markdown("""
    <div style="font-family:'Inter', sans-serif;font-size:0.7rem;color:#3A4050;line-height:1.7;margin-top:8px;">
        v12.0 | Institutional Decision Architecture
    </div>
    """, unsafe_allow_html=True)

# ─── DISCLAIMER BAR ──────────────────────────────────────────────────────────

st.markdown("""
<div class="disclaimer-bar">
    LEGAL NOTICE — Nomos Terminal is a quantitative simulation and decision-support tool only.
    Nothing on this platform constitutes financial advice, investment recommendation, or solicitation to trade.
    All backtests are hypothetical. Past simulated performance does not predict future results.
    Capital markets carry substantial risk of loss.
</div>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="nomos-header">
    <span class="nomos-wordmark">NOMOS TERMINAL</span>
    <span class="nomos-version">v12.0</span>
</div>
<div class="nomos-sub">Multi-Factor Decision Architecture &nbsp;·&nbsp; Fat-Tail Risk Engine &nbsp;·&nbsp; Walk-Forward Validation</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ─── LANDING STATE ────────────────────────────────────────────────────────────

if not user_input:
    st.markdown("""
    <div class="landing-card">
        <div class="landing-title">Enter a stock symbol in the sidebar to begin analysis.</div>
        <div class="landing-desc">
            Nomos Terminal synthesises trend, momentum, and volatility signals
            into a single score from 1 to 10, then stress-tests that signal
            with Monte Carlo simulation and walk-forward validation.
        </div>
        <div class="landing-steps">
            <div class="landing-step"><strong>Step 1</strong>Type any ticker symbol</div>
            <div class="landing-step"><strong>Step 2</strong>Read the Nomos Score</div>
            <div class="landing-step"><strong>Step 3</strong>Explore risk simulations</div>
            <div class="landing-step"><strong>Step 4</strong>Review backtest results</div>
        </div>
        <div style="font-family:'Inter', sans-serif;font-size:0.72rem;color:#3A4050;">
            Supports global equities: NYSE, NASDAQ, NSE, BSE, LSE, Euronext, TSE and more.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── DATA FETCH ───────────────────────────────────────────────────────────────

with st.spinner(f"Fetching data for {user_input}..."):
    df_raw, active_ticker, ticker_info = fetch_data(user_input)

if df_raw.empty or len(df_raw) < 150:
    st.error(
        f"Could not load data for **{user_input}**. "
        "Please check the symbol and try again. "
        "For Indian stocks add .NS (NSE) or .BO (BSE). "
        "For London stocks add .L. For Frankfurt add .DE."
    )
    st.stop()

# Currency
currency_sym, currency_code = get_currency(active_ticker)

# Auto-suggest risk-free rate based on currency
suggested_rf = DEFAULT_RISK_FREE.get(currency_code, 5.25)

# Indicators
df      = compute_indicators(df_raw, sensitivity)
curr    = df.iloc[-1]
returns = df['Close'].pct_change().dropna()

# ─── SIGNAL CALCULATIONS ─────────────────────────────────────────────────────

score_val   = curr['Nomos_Score']
score_color = GREEN if score_val >= 7 else RED if score_val <= 4 else GOLD

if score_val >= 8:    verdict = "STRONG BUY"
elif score_val >= 6.5: verdict = "BUY"
elif score_val >= 4.5: verdict = "HOLD"
elif score_val >= 3:   verdict = "SELL"
else:                  verdict = "STRONG SELL"

verdict_color = GREEN if "BUY" in verdict else RED if "SELL" in verdict else GOLD

vol_ratio  = curr['SD'] / df['SD'].mean()
conf_label = "STABLE" if vol_ratio < 1.3 else "ELEVATED" if vol_ratio < 1.8 else "CHAOTIC"
conf_color = GREEN if vol_ratio < 1.3 else GOLD if vol_ratio < 1.8 else RED

rsi_val   = curr['RSI']
rsi_state = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL"
rsi_color = RED if rsi_val > 70 else GREEN if rsi_val < 30 else MUTED

trend_bias = "BULLISH" if curr['MA50'] > curr['MA200'] else "BEARISH"
trend_col  = GREEN if trend_bias == "BULLISH" else RED
macd_cross = "BULLISH" if curr['MACD'] > curr['Signal_Line'] else "BEARISH"
macd_color = GREEN if macd_cross == "BULLISH" else RED
z_label    = "EXTENDED" if abs(curr['Z_Score']) > 1.5 else "MEAN-REVERTING"

var95, cvar95 = compute_var_cvar(returns, 0.95)

# ─── TOP METRIC BAR ───────────────────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ticker",       active_ticker)
c2.metric("Last Close",   fmt_price(curr['Close'], currency_sym))
c3.metric("Nomos Score",  f"{score_val:.1f} / 10")
c4.metric("RSI-14",       f"{rsi_val:.1f}", delta=rsi_state)
c5.metric("VaR 95%",      f"{var95*100:.2f}%", delta="Daily")
c6.metric("CVaR 95%",     f"{cvar95*100:.2f}%", delta="Expected Tail")

# Signal pills
pill_html = ""
for label, color, bg in [
    (f"SCORE {score_val:.1f} — {verdict}", verdict_color,
     "#0D2B1A" if "BUY" in verdict else "#2B0D0D" if "SELL" in verdict else "#1C1A0D"),
    (f"VOLATILITY: {conf_label}", conf_color,
     "#0D2B1A" if conf_label == "STABLE" else "#1C1A0D" if conf_label == "ELEVATED" else "#2B0D0D"),
    (f"TREND: {trend_bias}", trend_col,
     "#0D2B1A" if trend_bias == "BULLISH" else "#2B0D0D"),
    (f"MOMENTUM: {rsi_state}", rsi_color,
     "#2B0D0D" if rsi_val > 70 else "#0D2B1A" if rsi_val < 30 else "#141820"),
    (f"MACD: {macd_cross}", macd_color,
     "#0D2B1A" if macd_cross == "BULLISH" else "#2B0D0D"),
]:
    pill_html += f'<span class="signal-pill" style="color:{color};background:{bg};border-color:{color}33;">{label}</span>'

st.markdown(f'<div style="margin:14px 0 20px 0;line-height:2.4;">{pill_html}</div>', unsafe_allow_html=True)

# RF rate suggestion banner
if abs(suggested_rf - risk_free_rate * 100) > 1.5:
    st.markdown(
        f'<div style="background:#0D1018;border:1px solid #1E2430;border-left:2px solid {GOLD};'
        f'border-radius:0 4px 4px 0;padding:9px 16px;font-family:\'Inter\',sans-serif;font-size:0.75rem;'
        f'color:#8A9099;margin-bottom:16px;">'
        f'Detected currency: <span style="color:{GOLD};font-family:\'IBM Plex Mono\',monospace;">{currency_code}</span> — '
        f'suggested risk-free rate for this market is '
        f'<span style="color:{GOLD};font-family:\'IBM Plex Mono\',monospace;">{suggested_rf:.2f}%</span>. '
        f'Update in the sidebar if needed.</div>',
        unsafe_allow_html=True
    )

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Trend Architecture",
    "Fat-Tail Risk Engine",
    "Quant Vault",
    "Walk-Forward Backtest"
])

# ─── TAB 1: TREND ─────────────────────────────────────────────────────────────

with tab1:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])

    with col_a:
        # Price chart
        upper = df['MA50'] + (sensitivity * df['SD'])
        lower = df['MA50'] - (sensitivity * df['SD'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=upper,
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=lower,
            fill='tonexty', fillcolor='rgba(200,168,75,0.04)',
            line=dict(color='rgba(0,0,0,0)'), name='Bollinger Band', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            name='Price', line=dict(color='#D4D8E1', width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA50'],
            name='MA-50', line=dict(color=GOLD, dash='dot', width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA200'],
            name='MA-200', line=dict(color=BLUE, dash='dot', width=1.2)
        ))
        fig.update_layout(**base_layout(height=360, ytitle=f"Price ({currency_sym})"))
        st.plotly_chart(fig, use_container_width=True)

        # MACD
        fig_m = go.Figure()
        colors_macd = [GREEN if v >= 0 else RED for v in df['MACD_Hist'].fillna(0)]
        fig_m.add_trace(go.Bar(
            x=df.index, y=df['MACD_Hist'],
            name='Histogram', marker_color=colors_macd, opacity=0.7
        ))
        fig_m.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            name='MACD', line=dict(color=BLUE, width=1.5)
        ))
        fig_m.add_trace(go.Scatter(
            x=df.index, y=df['Signal_Line'],
            name='Signal', line=dict(color=GOLD, width=1.5)
        ))
        fig_m.update_layout(**base_layout(height=190))
        st.plotly_chart(fig_m, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Signal Breakdown</div>', unsafe_allow_html=True)

        # Z-Score card
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-label">Z-Score</p>
            <p class="stat-value">{curr['Z_Score']:.2f}</p>
            <span class="signal-pill" style="color:{'#2ECC71' if curr['Z_Score']>0 else '#E74C3C'};
                background:{'#0D2B1A' if curr['Z_Score']>0 else '#2B0D0D'};
                border-color:{'#2ECC7133' if curr['Z_Score']>0 else '#E74C3C33'};">{z_label}</span>
            <p class="stat-sub">Distance from 50-day mean in standard deviations</p>
        </div>
        """, unsafe_allow_html=True)

        # MACD card
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-label">MACD Cross</p>
            <p class="stat-value" style="font-size:1.1rem;">{curr['MACD']:.4f}</p>
            <span class="signal-pill" style="color:{'#2ECC71' if macd_cross=='BULLISH' else '#E74C3C'};
                background:{'#0D2B1A' if macd_cross=='BULLISH' else '#2B0D0D'};
                border-color:{'#2ECC7133' if macd_cross=='BULLISH' else '#E74C3C33'};">{macd_cross}</span>
            <p class="stat-sub">Signal line: {curr['Signal_Line']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        # ATR card
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-label">ATR (14-day)</p>
            <p class="stat-value" style="font-size:1.1rem;">{fmt_price(curr['ATR'], currency_sym)}</p>
            <p class="stat-sub">{curr['ATR_Pct']*100:.2f}% of current price — daily expected range</p>
        </div>
        """, unsafe_allow_html=True)

        # Nomos Score Gauge
        gauge_pct = (score_val - 1) / 9 * 100
        st.markdown(f"""
        <div class="gauge-wrap">
            <p class="gauge-label">Nomos Score</p>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{gauge_pct:.0f}%;"></div>
            </div>
            <p class="gauge-score" style="color:{score_color};">{score_val:.1f}</p>
            <p class="gauge-verdict" style="color:{verdict_color};">{verdict}</p>
            <div class="gauge-legend"><span>1</span><span>5</span><span>10</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Score guide
        st.markdown("""
        <div class="score-guide">
            <p><span style="color:#2ECC71;">8 – 10</span> &nbsp; Strong Buy</p>
            <p><span style="color:#2ECC71;">6.5 – 8</span> &nbsp; Buy</p>
            <p><span style="color:#C8A84B;">4.5 – 6.5</span> &nbsp; Hold</p>
            <p><span style="color:#E74C3C;">3 – 4.5</span> &nbsp; Sell</p>
            <p><span style="color:#E74C3C;">1 – 3</span> &nbsp; Strong Sell</p>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: MONTE CARLO ───────────────────────────────────────────────────────

with tab2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.spinner(f"Running {mc_sims:,} Monte Carlo paths..."):
        mc = run_monte_carlo(curr['Close'], tuple(returns.values), sims=mc_sims)

    p5, p25, p75, p95 = [np.percentile(mc, p, axis=1) for p in [5, 25, 75, 95]]
    prob_up = (mc[-1] > curr['Close']).mean()

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(
        y=p95, line=dict(color=f'rgba(46,204,113,0.2)', dash='dash'),
        name='P95 — Bull Case', hovertemplate=f"{currency_sym}%{{y:.2f}}"
    ))
    fig_mc.add_trace(go.Scatter(
        y=p75, fill='tonexty', fillcolor='rgba(46,204,113,0.05)',
        line=dict(color='rgba(46,204,113,0.25)'), name='P75'
    ))
    fig_mc.add_trace(go.Scatter(
        y=p25, fill='tonexty', fillcolor='rgba(231,76,60,0.05)',
        line=dict(color='rgba(231,76,60,0.25)'), name='P25'
    ))
    fig_mc.add_trace(go.Scatter(
        y=p5, line=dict(color='rgba(231,76,60,0.2)', dash='dash'),
        name='P5 — Bear Case', hovertemplate=f"{currency_sym}%{{y:.2f}}"
    ))
    fig_mc.add_trace(go.Scatter(
        y=mc.mean(axis=1), line=dict(color=GOLD, width=2.5),
        name='Expected Path', hovertemplate=f"{currency_sym}%{{y:.2f}}"
    ))
    fig_mc.add_hline(
        y=curr['Close'],
        line=dict(color=MUTED, width=1, dash='dot'),
        annotation_text="Current", annotation_font_color=MUTED
    )
    fig_mc.update_layout(**base_layout(height=420, xtitle="Trading Days", ytitle=f"Projected Price ({currency_sym})"))
    st.plotly_chart(fig_mc, use_container_width=True)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Prob. Upside",    f"{prob_up*100:.1f}%")
    mc2.metric("Bull Case (P95)", fmt_price(p95[-1], currency_sym))
    mc3.metric("Expected Path",   fmt_price(mc.mean(axis=1)[-1], currency_sym))
    mc4.metric("Bear Case (P5)",  fmt_price(p5[-1], currency_sym))
    mc5.metric("Bull / Bear",
               f"+{(p95[-1]/curr['Close']-1)*100:.1f}% / {(p5[-1]/curr['Close']-1)*100:.1f}%")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.expander("Methodology — Fat-Tail Simulation"):
        df_t_param = t_dist.fit(returns)[0]
        st.markdown(f"""
        <div style="font-family:'Inter', sans-serif;font-size:0.8rem;color:#8A9099;line-height:1.8;">
            <div class="tooltip-row"><span class="tooltip-term">Model</span>
                <span class="tooltip-def">Student's t-distribution — captures fat tails that Gaussian models miss</span></div>
            <div class="tooltip-row"><span class="tooltip-term">Degrees</span>
                <span class="tooltip-def">df = {df_t_param:.1f} (lower = fatter tails, more crash risk accounted for)</span></div>
            <div class="tooltip-row"><span class="tooltip-term">Fitted on</span>
                <span class="tooltip-def">{len(returns):,} daily return observations from 3-year history</span></div>
            <div class="tooltip-row"><span class="tooltip-term">Paths</span>
                <span class="tooltip-def">{mc_sims:,} independent simulations over 21 trading days (~1 month)</span></div>
            <div class="tooltip-row"><span class="tooltip-term">VaR 95%</span>
                <span class="tooltip-def">On any given day, 5% chance of losing more than {var95*100:.2f}%</span></div>
            <div class="tooltip-row"><span class="tooltip-term">CVaR 95%</span>
                <span class="tooltip-def">When those bad days hit, average loss is {cvar95*100:.2f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 3: QUANT VAULT ───────────────────────────────────────────────────────

with tab3:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    df['Signal']   = np.where(df['Nomos_Score'] > 7.5, 1,
                     np.where(df['Nomos_Score'] < 4.0, 0, np.nan))
    df['Position'] = df['Signal'].ffill().fillna(0).shift(1)
    daily_rf       = (1 + risk_free_rate) ** (1 / 252) - 1
    ret_series     = df['Close'].pct_change()
    trades         = df['Position'].diff().abs().fillna(0)
    strat_r        = (ret_series * df['Position']) - (trades * 0.001) - daily_rf
    mkt_r          = ret_series - daily_rf

    if not strat_r.dropna().empty:
        cum_strat = (1 + strat_r.dropna()).cumprod()
        cum_mkt   = (1 + mkt_r.dropna()).cumprod()

        sharpe   = (strat_r.mean() / strat_r.std()) * np.sqrt(252) if strat_r.std() != 0 else 0
        mdd      = (cum_strat / cum_strat.cummax() - 1).min()
        ann_r    = strat_r.mean() * 252
        calmar   = abs(ann_r / mdd) if mdd != 0 else 0
        downside = strat_r[strat_r < 0].std()
        sortino  = (strat_r.mean() / downside) * np.sqrt(252) if downside != 0 else 0
        var95_s, cvar95_s = compute_var_cvar(strat_r, 0.95)
        win_rate = (strat_r.dropna() > 0).mean()
        n_trades = int(trades.sum() / 2)

        q1, q2, q3, q4, q5, q6 = st.columns(6)
        q1.metric("Sharpe Ratio",  f"{sharpe:.2f}",
                  help="Return per unit of risk. Above 1 is good. Above 2 is excellent.")
        q2.metric("Sortino Ratio", f"{sortino:.2f}",
                  help="Like Sharpe but only penalises downside volatility.")
        q3.metric("Calmar Ratio",  f"{calmar:.2f}",
                  help="Annualised return divided by max drawdown.")
        q4.metric("Max Drawdown",  f"{mdd*100:.1f}%",
                  help="Worst peak-to-trough decline during the backtest period.")
        q5.metric("Win Rate",      f"{win_rate*100:.1f}%",
                  help="Percentage of trading days with positive excess return.")
        q6.metric("Round Trips",   str(n_trades),
                  help="Number of complete buy-then-sell cycles executed.")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Equity curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            y=cum_strat.values, name='Nomos Strategy',
            line=dict(color=GOLD, width=2),
            fill='tozeroy', fillcolor='rgba(200,168,75,0.04)'
        ))
        fig_eq.add_trace(go.Scatter(
            y=cum_mkt.values, name='Buy & Hold',
            line=dict(color=BLUE, width=1.5, dash='dot')
        ))
        fig_eq.update_layout(**base_layout(height=300, ytitle="Cumulative Return (x)"))
        st.plotly_chart(fig_eq, use_container_width=True)

        with st.expander("Methodology Disclosure"):
            st.markdown(f"""
            | Parameter | Value |
            |---|---|
            | Entry Signal | Nomos Score > 7.5 |
            | Exit Signal | Nomos Score < 4.0 |
            | Execution Lag | 1-day (eliminates lookahead bias) |
            | Transaction Cost | 10 bps per trade |
            | Risk-Free Rate | {risk_free_rate*100:.2f}% p.a. (daily-adjusted) |
            | Sharpe Basis | Excess return over risk-free rate |
            | Sortino Basis | Downside deviation only |
            | Calmar Basis | Annualised excess return / max drawdown |
            | Universe | Single asset, long-only |
            | Currency | {currency_code} ({currency_sym}) |
            """)
    else:
        st.warning("Insufficient signal data to run backtest for this period.")

# ─── TAB 4: WALK-FORWARD ──────────────────────────────────────────────────────

with tab4:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Inter', sans-serif;font-size:0.82rem;color:#8A9099;margin-bottom:16px;line-height:1.7;">
        The 3-year history is split into 5 independent folds. Each fold trains on its first half and is tested on the second.
        This prevents the strategy from simply memorising past data — if it performs across all folds, the signal is likely real.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running 5-fold walk-forward validation..."):
        wf_results = walk_forward_backtest(df, risk_free_rate)

    if wf_results:
        sharpes  = [r['sharpe']     for r in wf_results]
        calmars  = [r['calmar']     for r in wf_results]
        mdds     = [r['mdd']        for r in wf_results]
        win_rs   = [r['win_rate']   for r in wf_results]
        ann_rs   = [r['ann_return'] for r in wf_results]

        w1, w2, w3, w4, w5 = st.columns(5)
        w1.metric("Avg Sharpe (OOS)",    f"{np.mean(sharpes):.2f}",
                  delta=f"SD ±{np.std(sharpes):.2f}")
        w2.metric("Avg Calmar (OOS)",    f"{np.mean(calmars):.2f}",
                  delta=f"SD ±{np.std(calmars):.2f}")
        w3.metric("Avg Max Drawdown",    f"{np.mean(mdds)*100:.1f}%")
        w4.metric("Avg Win Rate (OOS)",  f"{np.mean(win_rs)*100:.1f}%")
        w5.metric("Avg Ann. Return",     f"{np.mean(ann_rs)*100:.1f}%")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Per-fold bar chart
        fold_labels = [f"Fold {r['fold']}" for r in wf_results]
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(name='Sharpe',   x=fold_labels, y=sharpes,   marker_color=GOLD,  opacity=0.85))
        fig_wf.add_trace(go.Bar(name='Calmar',   x=fold_labels, y=calmars,   marker_color=BLUE,  opacity=0.85))
        fig_wf.add_trace(go.Bar(name='Win Rate', x=fold_labels, y=win_rs,    marker_color=GREEN, opacity=0.85))
        fig_wf.update_layout(**base_layout(height=300), barmode='group')
        st.plotly_chart(fig_wf, use_container_width=True)

        # Cumulative returns per fold
        fold_colors = [GOLD, BLUE, GREEN, '#BC8CFF', '#FF7B72']
        fig_folds = go.Figure()
        for i, r in enumerate(wf_results):
            fig_folds.add_trace(go.Scatter(
                y=r['cum_returns'].values,
                name=f"Fold {r['fold']}  (Sharpe {r['sharpe']:.2f})",
                line=dict(color=fold_colors[i], width=1.5)
            ))
        fig_folds.update_layout(**base_layout(height=280, ytitle="Cumulative Return (x)"))
        st.plotly_chart(fig_folds, use_container_width=True)

        # Consistency verdict
        consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)
        if consistency >= 0.8:
            st.markdown(f"""
            <div class="consistency-block" style="background:#0D2B1A;border:1px solid #238636;color:#3FB950;">
                Strategy is consistent — {consistency*100:.0f}% of out-of-sample folds produced a positive Sharpe ratio.
                The signal appears robust across different market periods.
            </div>""", unsafe_allow_html=True)
        elif consistency >= 0.6:
            st.markdown(f"""
            <div class="consistency-block" style="background:#1C1A0D;border:1px solid #C8A84B55;color:#C8A84B;">
                Strategy is moderately consistent — {consistency*100:.0f}% of folds were profitable.
                Exercise caution and consider widening the hold zone.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="consistency-block" style="background:#2B0D0D;border:1px solid #DA3633;color:#F85149;">
                Strategy is inconsistent — only {consistency*100:.0f}% of out-of-sample folds were profitable.
                The Nomos Score may not be reliable for this asset at current sensitivity settings.
                Try adjusting the sensitivity slider.
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("Insufficient data to run walk-forward validation for this asset.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────

st.divider()
st.markdown(f"""
<div class="disclaimer-bar" style="margin-top:8px;">
    LEGAL DISCLOSURE — Nomos Terminal is a mathematical simulation and decision-support tool only.
    No content constitutes financial advice, investment recommendations, or solicitation to trade any security.
    Not registered with SEBI, SEC, FCA, or any regulatory authority.
    Past simulated performance does not guarantee future results. All backtests are hypothetical and subject to
    survivorship bias and model risk. Capital markets involve substantial risk of loss.
    Currency displayed: {currency_code} ({currency_sym}). Risk-free rate applied: {risk_free_rate*100:.2f}% p.a.
</div>
<div style="font-family:'IBM Plex Mono', monospace;font-size:0.6rem;color:#1E2430;text-align:right;margin-top:8px;">
    NOMOS TERMINAL v12.0 — MULTI-FACTOR DECISION ARCHITECTURE
</div>
""", unsafe_allow_html=True)
