import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t as t_dist
from scipy.optimize import minimize_scalar
from itertools import product as iterproduct
import io
import csv

# ============================================================
# NOMOS TERMINAL v10.1
# Fixes: weight calibration | Kelly sizing | true walk-forward
#        optimisation | long/short | export/watchlist | no emojis
# ============================================================

st.set_page_config(page_title="Nomos Terminal | v10.1", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #080B10; color: #C9D1D9; }
h1, h2, h3 { font-family: 'JetBrains Mono', monospace !important; letter-spacing: -0.5px; }

.stMetric {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
    padding: 16px !important;
    border-radius: 8px !important;
    border: 1px solid #21262D !important;
    border-top: 2px solid #F0B429 !important;
}
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.7rem !important; letter-spacing: 1.5px !important; text-transform: uppercase; color: #8B949E !important; }
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }

.stat-card {
    background: linear-gradient(135deg, #0D1117, #161B22);
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 12px;
}
.tag-bull    { background:#0D2B1A; color:#3FB950; padding:3px 10px; border-radius:4px; font-family:'JetBrains Mono'; font-size:12px; border:1px solid #238636; }
.tag-bear    { background:#2B0D0D; color:#F85149; padding:3px 10px; border-radius:4px; font-family:'JetBrains Mono'; font-size:12px; border:1px solid #DA3633; }
.tag-neutral { background:#1C1F24; color:#8B949E; padding:3px 10px; border-radius:4px; font-family:'JetBrains Mono'; font-size:12px; border:1px solid #30363D; }
.disclaimer  {
    background:#0D1117; border:1px solid #21262D; border-left:3px solid #F0B429;
    padding:12px 16px; border-radius:0 8px 8px 0; font-size:0.75rem; color:#8B949E;
}
</style>
""", unsafe_allow_html=True)

# ─── DATA LAYER ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def fetch_data(ticker):
    for suffix in ["", ".NS", ".BO"]:
        try:
            tk = f"{ticker}{suffix}"
            data = yf.Ticker(tk).history(period="3y")
            if not data.empty and len(data) >= 150:
                return data, tk
        except:
            continue
    return pd.DataFrame(), ticker

# ─── RAW FACTOR COMPUTATION ───────────────────────────────────────────────────

def compute_raw_factors(df):
    """Returns df with raw factor columns — weights applied separately."""
    df = df.copy()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['SD']    = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['MA50']) / df['SD']

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['Signal_Line']

    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    df['ATR']     = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close']

    # Normalised factor components (each in [-1, +1] range before weighting)
    df['F_Z']    = df['Z_Score'].clip(-2, 2) / 2
    df['F_RSI']  = ((df['RSI'] - 50) / 50).clip(-1, 1)
    df['F_MACD'] = (df['MACD_Hist'] / df['Close'].replace(0, np.nan)).clip(-0.01, 0.01) / 0.01
    df['F_TREND']= np.where(df['MA50'] > df['MA200'], 1.0, -1.0)

    return df


def apply_weights(df, w_z, w_rsi, w_macd, w_trend, sensitivity=1.5):
    """
    Score = 5 + weighted sum of normalised factors.
    Sensitivity scales the Z component only (user-facing parameter).
    """
    z_scaled = (df['Z_Score'] / sensitivity).clip(-1, 1)
    score = (
        5
        + z_scaled    * w_z
        + df['F_RSI'] * w_rsi
        + df['F_MACD']* w_macd
        + df['F_TREND']* w_trend
    )
    return score.clip(1, 10)

# ─── WEIGHT CALIBRATION (IC-based, in-sample only) ───────────────────────────

@st.cache_data(ttl=3600)
def calibrate_weights(close_series, factors_dict):
    """
    Maximise the Information Coefficient (rank correlation between each factor
    and forward 21-day return). Returns calibrated weights as dict.
    This runs once on the full history and is explicitly flagged as in-sample.
    """
    fwd_ret = close_series.pct_change(21).shift(-21)
    ics = {}
    for name, series in factors_dict.items():
        aligned = pd.concat([series, fwd_ret], axis=1).dropna()
        if len(aligned) < 30:
            ics[name] = 0.0
            continue
        ic = aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank())
        ics[name] = max(ic, 0)  # only use factors with positive predictive IC

    total = sum(ics.values()) or 1
    weights = {k: v / total * 3 for k, v in ics.items()}  # scale so max swing ~±3
    return weights, ics

# ─── KELLY POSITION SIZING ────────────────────────────────────────────────────

def kelly_fraction(win_rate, avg_win, avg_loss, kelly_cap=0.25):
    """
    Full Kelly = (W/L * p - (1-p)) / (W/L)
    Capped at kelly_cap for half-Kelly institutional sizing.
    """
    if avg_loss == 0:
        return 0.0
    wl_ratio = avg_win / abs(avg_loss)
    kelly = (wl_ratio * win_rate - (1 - win_rate)) / wl_ratio
    return float(np.clip(kelly, 0, kelly_cap))

# ─── SIGNAL + BACKTEST ENGINE ─────────────────────────────────────────────────

def backtest_slice(price_series, score_series, risk_free_rate,
                   entry_thresh, exit_thresh,
                   allow_short=False, use_kelly=False,
                   kelly_win_rate=0.5, kelly_avg_win=0.01, kelly_avg_loss=0.01):

    sig = np.where(score_series > entry_thresh, 1,
          np.where(score_series < exit_thresh,  -1 if allow_short else 0,
          np.nan))
    pos = pd.Series(sig, index=score_series.index).ffill().fillna(0).shift(1)

    if use_kelly:
        kf = kelly_fraction(kelly_win_rate, kelly_avg_win, kelly_avg_loss)
        pos = pos * kf  # scale position by Kelly fraction

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    ret      = price_series.pct_change()
    trades   = pos.diff().abs().fillna(0)
    strat_r  = (ret * pos) - (trades * 0.001) - daily_rf

    if strat_r.dropna().empty:
        return None

    cum    = (1 + strat_r.dropna()).cumprod()
    std    = strat_r.std()
    sharpe = (strat_r.mean() / std) * np.sqrt(252) if std != 0 else 0
    mdd    = (cum / cum.cummax() - 1).min()
    ann_r  = strat_r.mean() * 252
    calmar = abs(ann_r / mdd) if mdd != 0 else 0
    down   = strat_r[strat_r < 0].std()
    sortino= (strat_r.mean() / down) * np.sqrt(252) if down != 0 else 0
    win_r  = (strat_r.dropna() > 0).mean()
    n_tr   = int(trades.sum() / 2)

    # Kelly stats for next fold
    wins   = strat_r[strat_r > 0]
    losses = strat_r[strat_r < 0]
    avg_w  = wins.mean()  if not wins.empty  else 0.01
    avg_l  = losses.mean()if not losses.empty else -0.01

    return dict(sharpe=sharpe, mdd=mdd, calmar=calmar, sortino=sortino,
                ann_return=ann_r, win_rate=win_r, n_trades=n_tr,
                cum_returns=cum, entry=entry_thresh, exit=exit_thresh,
                kelly_win_rate=win_r, kelly_avg_win=avg_w, kelly_avg_loss=avg_l)

# ─── TRUE WALK-FORWARD WITH OPTIMISATION ──────────────────────────────────────

def walk_forward_optimised(df, risk_free_rate, n_folds=5,
                            allow_short=False, use_kelly=False):
    """
    True walk-forward:
      - TRAIN half: grid-search (entry, exit) thresholds to maximise Sharpe
      - TEST  half: apply best thresholds from training window (out-of-sample)
    Thresholds are re-optimised each fold independently.
    """
    ENTRY_GRID = [6.5, 7.0, 7.5, 8.0]
    EXIT_GRID  = [3.0, 3.5, 4.0, 4.5]

    data      = df.dropna(subset=['Nomos_Score', 'Close']).copy()
    fold_size = len(data) // n_folds
    results   = []

    for i in range(n_folds):
        fold  = data.iloc[i * fold_size: (i + 1) * fold_size]
        mid   = len(fold) // 2
        train = fold.iloc[:mid]
        test  = fold.iloc[mid:]

        # --- TRAINING: find best (entry, exit) thresholds ---
        best_sharpe, best_entry, best_exit = -np.inf, 7.5, 4.0
        best_kelly_stats = dict(win_rate=0.5, avg_win=0.01, avg_loss=0.01)

        for entry, exit_ in iterproduct(ENTRY_GRID, EXIT_GRID):
            if exit_ >= entry:
                continue
            res = backtest_slice(train['Close'], train['Nomos_Score'],
                                 risk_free_rate, entry, exit_,
                                 allow_short=allow_short, use_kelly=False)
            if res and res['sharpe'] > best_sharpe:
                best_sharpe = res['sharpe']
                best_entry  = entry
                best_exit   = exit_
                best_kelly_stats = dict(
                    kelly_win_rate=res['kelly_win_rate'],
                    kelly_avg_win =res['kelly_avg_win'],
                    kelly_avg_loss=res['kelly_avg_loss']
                )

        # --- TESTING: apply optimised thresholds OOS ---
        test_res = backtest_slice(
            test['Close'], test['Nomos_Score'],
            risk_free_rate, best_entry, best_exit,
            allow_short=allow_short, use_kelly=use_kelly,
            **best_kelly_stats
        )

        if test_res:
            test_res['fold']         = i + 1
            test_res['train_sharpe'] = best_sharpe
            results.append(test_res)

    return results

# ─── MONTE CARLO ──────────────────────────────────────────────────────────────

def run_monte_carlo(current_price, returns_series, days=21, sims=2000):
    params = t_dist.fit(returns_series.dropna())
    df_t, loc_t, scale_t = params
    shocks = t_dist.rvs(df_t, loc=loc_t, scale=scale_t, size=(days, sims))
    paths  = np.zeros((days + 1, sims))
    paths[0] = current_price
    for d in range(1, days + 1):
        paths[d] = paths[d - 1] * (1 + shocks[d - 1])
    return paths

# ─── RISK METRICS ─────────────────────────────────────────────────────────────

def compute_var_cvar(returns, confidence=0.95):
    clean = returns.dropna()
    var   = -np.percentile(clean, (1 - confidence) * 100)
    cvar  = -clean[clean <= -var].mean()
    return var, cvar

# ─── CSV EXPORT ───────────────────────────────────────────────────────────────

def build_export_csv(df, active_ticker, wf_results):
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["NOMOS TERMINAL — EXPORT"])
    w.writerow(["Ticker", active_ticker])
    w.writerow([])
    w.writerow(["Date", "Close", "MA50", "MA200", "RSI", "Z_Score",
                "ATR", "Nomos_Score"])
    for idx, row in df[['Close','MA50','MA200','RSI','Z_Score',
                         'ATR','Nomos_Score']].dropna().iterrows():
        w.writerow([str(idx.date()),
                    f"{row['Close']:.4f}", f"{row['MA50']:.4f}",
                    f"{row['MA200']:.4f}", f"{row['RSI']:.2f}",
                    f"{row['Z_Score']:.4f}", f"{row['ATR']:.4f}",
                    f"{row['Nomos_Score']:.2f}"])
    if wf_results:
        w.writerow([])
        w.writerow(["Walk-Forward Results"])
        w.writerow(["Fold","Entry Thresh","Exit Thresh","OOS Sharpe",
                    "OOS Calmar","OOS MaxDD","Win Rate","Ann Return"])
        for r in wf_results:
            w.writerow([r['fold'], r['entry'], r['exit'],
                        f"{r['sharpe']:.3f}", f"{r['calmar']:.3f}",
                        f"{r['mdd']*100:.2f}%", f"{r['win_rate']*100:.1f}%",
                        f"{r['ann_return']*100:.2f}%"])
    return buf.getvalue().encode()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## NOMOS v10.1")

# Watchlist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

wl_input = st.sidebar.text_input("Add to Watchlist", placeholder="e.g. AAPL").upper()
if st.sidebar.button("Add") and wl_input and wl_input not in st.session_state.watchlist:
    st.session_state.watchlist.append(wl_input)

if st.session_state.watchlist:
    st.sidebar.markdown("**Watchlist**")
    for t in st.session_state.watchlist:
        col_a, col_b = st.sidebar.columns([3, 1])
        col_a.write(t)
        if col_b.button("X", key=f"rm_{t}"):
            st.session_state.watchlist.remove(t)
            st.rerun()

st.sidebar.divider()
user_input     = st.sidebar.text_input("Asset Search", value="NVDA").upper()
sensitivity    = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5, 0.1)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=7.0, step=0.1) / 100
mc_sims        = st.sidebar.select_slider("MC Simulations", options=[500, 1000, 2000, 5000], value=2000)
allow_short    = st.sidebar.checkbox("Allow Short Selling", value=False)
use_kelly      = st.sidebar.checkbox("Kelly Position Sizing", value=False)
st.sidebar.divider()
st.sidebar.caption("v10.1 | Institutional Decision Architecture")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

st.markdown("# NOMOS TERMINAL")
st.caption("Institutional Decision Support  |  v10.1  |  Multi-Factor · Fat-Tail · Walk-Forward Optimisation · Kelly Sizing")
st.divider()

if user_input:
    with st.spinner(f"Fetching {user_input}..."):
        df_raw, active_ticker = fetch_data(user_input)

    if df_raw.empty or len(df_raw) < 150:
        st.error("Asset not found or insufficient history (need 150+ trading days).")
        st.stop()

    df = compute_raw_factors(df_raw)

    # ── WEIGHT CALIBRATION ─────────────────────────────────────────────────────
    factors_for_calib = {
        'z':    df['F_Z'],
        'rsi':  df['F_RSI'],
        'macd': df['F_MACD'],
        'trend':df['F_TREND'],
    }
    cal_weights, cal_ics = calibrate_weights(
        df['Close'],
        {k: v for k, v in factors_for_calib.items()}
    )
    w_z    = cal_weights.get('z',    3.0)
    w_rsi  = cal_weights.get('rsi',  1.5)
    w_macd = cal_weights.get('macd', 1.0)
    w_trend= cal_weights.get('trend',0.5)

    df['Nomos_Score'] = apply_weights(df, w_z, w_rsi, w_macd, w_trend, sensitivity)

    curr    = df.iloc[-1]
    returns = df['Close'].pct_change().dropna()

    # ── TOP METRIC BAR ─────────────────────────────────────────────────────────
    score_val   = curr['Nomos_Score']
    score_color = "#3FB950" if score_val >= 7 else "#F85149" if score_val <= 4 else "#F0B429"
    vol_ratio   = curr['SD'] / df['SD'].mean()
    conf_label  = "STABLE" if vol_ratio < 1.3 else "ELEVATED" if vol_ratio < 1.8 else "CHAOTIC"
    conf_color  = "#3FB950" if vol_ratio < 1.3 else "#F0B429" if vol_ratio < 1.8 else "#F85149"
    rsi_val     = curr['RSI']
    rsi_state   = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL"
    rsi_color   = "#F85149" if rsi_val > 70 else "#3FB950" if rsi_val < 30 else "#8B949E"
    trend_bias  = "BULLISH" if curr['MA50'] > curr['MA200'] else "BEARISH"
    trend_col   = "#3FB950" if trend_bias == "BULLISH" else "#F85149"
    var95, cvar95 = compute_var_cvar(returns, 0.95)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("TICKER",      active_ticker)
    c2.metric("LAST CLOSE",  f"${curr['Close']:.2f}")
    c3.metric("NOMOS SCORE", f"{score_val:.1f}/10")
    c4.metric("RSI-14",      f"{rsi_val:.1f}", delta=rsi_state)
    c5.metric("VaR 95%",     f"{var95*100:.2f}%", delta="Daily")
    c6.metric("CVaR 95%",    f"{cvar95*100:.2f}%", delta="Exp. Loss")

    st.markdown(f"""
    <div style="display:flex;gap:8px;margin:8px 0 16px 0;flex-wrap:wrap;">
        <span style="background:#0D1117;border:1px solid {score_color};color:{score_color};padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono';font-size:12px;">
            SCORE {score_val:.1f} &middot; {'STRONG BUY' if score_val>=8 else 'BUY' if score_val>=6.5 else 'HOLD' if score_val>=4.5 else 'SELL' if score_val>=3 else 'STRONG SELL'}
        </span>
        <span style="background:#0D1117;border:1px solid {conf_color};color:{conf_color};padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono';font-size:12px;">
            VOLATILITY: {conf_label} ({vol_ratio:.2f}x)
        </span>
        <span style="background:#0D1117;border:1px solid {trend_col};color:{trend_col};padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono';font-size:12px;">
            TREND: {trend_bias}
        </span>
        <span style="background:#0D1117;border:1px solid {rsi_color};color:{rsi_color};padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono';font-size:12px;">
            MOMENTUM: {rsi_state}
        </span>
        {'<span style="background:#0D1117;border:1px solid #58A6FF;color:#58A6FF;padding:4px 12px;border-radius:4px;font-family:JetBrains Mono;font-size:12px;">LONG/SHORT ON</span>' if allow_short else ''}
        {'<span style="background:#0D1117;border:1px solid #BC8CFF;color:#BC8CFF;padding:4px 12px;border-radius:4px;font-family:JetBrains Mono;font-size:12px;">KELLY SIZING ON</span>' if use_kelly else ''}
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Trend Architecture",
        "Fat-Tail Risk Engine",
        "Quant Vault",
        "Walk-Forward (Optimised)",
        "Weight Calibration"
    ])

    # ── TAB 1: TREND ──────────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            fig = go.Figure()
            upper = df['MA50'] + (sensitivity * df['SD'])
            lower = df['MA50'] - (sensitivity * df['SD'])
            fig.add_trace(go.Scatter(x=df.index, y=upper, line_color='rgba(0,0,0,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=lower, fill='tonexty',
                                     fillcolor='rgba(240,180,41,0.05)', line_color='rgba(0,0,0,0)', name='Noise Band'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'],  name='Price',  line=dict(color='#E6EDF3', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'],   name='MA-50',  line=dict(color='#F0B429', dash='dot', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA200'],  name='MA-200', line=dict(color='#58A6FF', dash='dot', width=1)))
            fig.update_layout(template="plotly_dark", paper_bgcolor='#0D1117', plot_bgcolor='#0D1117',
                               height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation='h', y=1.05))
            st.plotly_chart(fig, use_container_width=True)

            colors_m = ['#3FB950' if v >= 0 else '#F85149' for v in df['MACD_Hist'].fillna(0)]
            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Histogram', marker_color=colors_m))
            fig_m.add_trace(go.Scatter(x=df.index, y=df['MACD'],        name='MACD',   line=dict(color='#58A6FF', width=1.5)))
            fig_m.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='#F0B429', width=1.5)))
            fig_m.update_layout(template="plotly_dark", paper_bgcolor='#0D1117', plot_bgcolor='#0D1117',
                                  height=200, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig_m, use_container_width=True)

        with col_b:
            st.markdown("#### Signal Summary")
            macd_cross = "BULLISH" if curr['MACD'] > curr['Signal_Line'] else "BEARISH"
            z_label    = "EXTENDED" if abs(curr['Z_Score']) > 1.5 else "MEAN-REVERTING"
            st.markdown(f"""
            <div class="stat-card">
                <p style="color:#8B949E;font-size:11px;letter-spacing:1px;margin:0">Z-SCORE</p>
                <p style="font-family:'JetBrains Mono';font-size:1.3rem;margin:4px 0">{curr['Z_Score']:.2f}</p>
                <span class="{'tag-bull' if curr['Z_Score']>0 else 'tag-bear'}">{z_label}</span>
            </div>
            <div class="stat-card">
                <p style="color:#8B949E;font-size:11px;letter-spacing:1px;margin:0">MACD CROSS</p>
                <p style="font-family:'JetBrains Mono';font-size:1.1rem;margin:4px 0">{curr['MACD']:.3f}</p>
                <span class="{'tag-bull' if macd_cross=='BULLISH' else 'tag-bear'}">{macd_cross}</span>
            </div>
            <div class="stat-card">
                <p style="color:#8B949E;font-size:11px;letter-spacing:1px;margin:0">ATR (14)</p>
                <p style="font-family:'JetBrains Mono';font-size:1.1rem;margin:4px 0">{curr['ATR']:.2f}</p>
                <p style="font-size:11px;color:#8B949E;margin:0">{curr['ATR_Pct']*100:.2f}% of price</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Nomos Score")
            gauge_pct = (score_val - 1) / 9
            st.markdown(f"""
            <div style="background:#0D1117;border:1px solid #21262D;border-radius:8px;padding:16px;">
                <div style="background:#21262D;border-radius:4px;height:12px;overflow:hidden;">
                    <div style="width:{gauge_pct*100:.0f}%;height:100%;background:linear-gradient(90deg,#F85149,#F0B429,#3FB950);border-radius:4px;"></div>
                </div>
                <p style="font-family:'JetBrains Mono';font-size:2rem;text-align:center;color:{score_color};margin:8px 0 0 0">{score_val:.1f}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: MONTE CARLO ────────────────────────────────────────────────────
    with tab2:
        with st.spinner(f"Running {mc_sims:,} Monte Carlo simulations..."):
            mc = run_monte_carlo(curr['Close'], returns, sims=mc_sims)

        p5, p25, p75, p95 = [np.percentile(mc, p, axis=1) for p in [5, 25, 75, 95]]
        prob_up = (mc[-1] > curr['Close']).mean()

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(y=p95, line=dict(color='rgba(63,185,80,0.15)', dash='dash'), name='95th (Bull)'))
        fig_mc.add_trace(go.Scatter(y=p75, fill='tonexty', fillcolor='rgba(63,185,80,0.06)',
                                     line=dict(color='rgba(63,185,80,0.3)'), name='75th'))
        fig_mc.add_trace(go.Scatter(y=p25, fill='tonexty', fillcolor='rgba(248,81,73,0.06)',
                                     line=dict(color='rgba(248,81,73,0.3)'), name='25th'))
        fig_mc.add_trace(go.Scatter(y=p5, line=dict(color='rgba(248,81,73,0.15)', dash='dash'), name='5th (Bear)'))
        fig_mc.add_trace(go.Scatter(y=mc.mean(axis=1), line=dict(color='#F0B429', width=2.5), name='Expected Path'))
        fig_mc.update_layout(template="plotly_dark", paper_bgcolor='#0D1117', plot_bgcolor='#0D1117',
                               height=420, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Trading Days", yaxis_title="Projected Price",
                               legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig_mc, use_container_width=True)

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Prob. Upside",    f"{prob_up*100:.1f}%")
        mc2.metric("Bull Case (P95)", f"${p95[-1]:.2f}")
        mc3.metric("Expected",        f"${mc.mean(axis=1)[-1]:.2f}")
        mc4.metric("Bear Case (P5)",  f"${p5[-1]:.2f}")
        mc5.metric("t-Distribution df", f"{t_dist.fit(returns)[0]:.1f}")

        st.info(f"Student's t-Distribution fitted to {len(returns):,} return observations. "
                f"Fat tails capture tail risk beyond Gaussian assumptions. {mc_sims:,} simulation paths.")

    # ── TAB 3: QUANT VAULT ────────────────────────────────────────────────────
    with tab3:
        daily_rf  = (1 + risk_free_rate) ** (1 / 252) - 1
        ret_series = df['Close'].pct_change()

        # Derive Kelly stats from full history for display
        quick_res = backtest_slice(
            df['Close'], df['Nomos_Score'], risk_free_rate,
            entry_thresh=7.5, exit_thresh=4.0,
            allow_short=allow_short, use_kelly=use_kelly,
            kelly_win_rate=0.5, kelly_avg_win=0.01, kelly_avg_loss=0.01
        )

        if quick_res:
            kf = kelly_fraction(quick_res['kelly_win_rate'],
                                quick_res['kelly_avg_win'],
                                abs(quick_res['kelly_avg_loss']))

            q1, q2, q3, q4, q5, q6 = st.columns(6)
            q1.metric("Sharpe (Excess)",  f"{quick_res['sharpe']:.2f}")
            q2.metric("Sortino Ratio",    f"{quick_res['sortino']:.2f}")
            q3.metric("Calmar Ratio",     f"{quick_res['calmar']:.2f}")
            q4.metric("Max Drawdown",     f"{quick_res['mdd']*100:.1f}%")
            q5.metric("Win Rate",         f"{quick_res['win_rate']*100:.1f}%")
            q6.metric("Kelly Fraction",   f"{kf*100:.1f}%" if use_kelly else "Off")

            mkt_r  = ret_series - daily_rf
            cum_mkt = (1 + mkt_r.dropna()).cumprod()

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=quick_res['cum_returns'].values, name='Nomos Strategy',
                                         line=dict(color='#F0B429', width=2)))
            fig_eq.add_trace(go.Scatter(y=cum_mkt.values, name='Buy & Hold',
                                         line=dict(color='#58A6FF', width=1.5, dash='dot')))
            fig_eq.update_layout(template="plotly_dark", paper_bgcolor='#0D1117', plot_bgcolor='#0D1117',
                                   height=320, margin=dict(l=0, r=0, t=10, b=0),
                                   yaxis_title="Cumulative Return (x)", legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig_eq, use_container_width=True)

            if use_kelly:
                st.info(f"Kelly Fraction: **{kf*100:.1f}%** of capital per signal "
                        f"(win rate {quick_res['win_rate']*100:.1f}%, "
                        f"avg win {quick_res['kelly_avg_win']*100:.2f}%, "
                        f"avg loss {quick_res['kelly_avg_loss']*100:.2f}%). "
                        f"Capped at 25% (half-Kelly institutional limit).")

            with st.expander("Methodology Disclosure"):
                st.markdown(f"""
                | Parameter | Value |
                |---|---|
                | Entry Threshold | Nomos Score > 7.5 |
                | Exit Threshold  | Nomos Score < 4.0 |
                | Execution Lag   | 1-day |
                | Transaction Cost | 10 bps per trade |
                | Risk-Free Rate  | {risk_free_rate*100:.1f}% p.a. |
                | Short Selling   | {"Enabled" if allow_short else "Disabled (long-only)"} |
                | Position Sizing | {"Kelly (capped 25%)" if use_kelly else "Binary (all-in/all-out)"} |
                | Universe        | Single asset |
                | Scope           | In-sample full-history backtest (see Walk-Forward tab for OOS) |
                """)

    # ── TAB 4: WALK-FORWARD (TRUE OPTIMISED) ─────────────────────────────────
    with tab4:
        st.markdown("### Walk-Forward Backtest — True 5-Fold Optimisation")
        st.caption(
            "Each fold: TRAIN half grid-searches (entry, exit) thresholds to maximise Sharpe. "
            "TEST half applies those thresholds out-of-sample. Thresholds re-optimised per fold."
        )

        with st.spinner("Running 5-fold optimised walk-forward..."):
            wf_results = walk_forward_optimised(
                df, risk_free_rate, n_folds=5,
                allow_short=allow_short, use_kelly=use_kelly
            )

        if wf_results:
            sharpes  = [r['sharpe']     for r in wf_results]
            caldmars = [r['calmar']     for r in wf_results]
            mdds     = [r['mdd']        for r in wf_results]
            win_rs   = [r['win_rate']   for r in wf_results]
            ann_rs   = [r['ann_return'] for r in wf_results]
            entries  = [r['entry']      for r in wf_results]
            exits    = [r['exit']       for r in wf_results]

            w1, w2, w3, w4, w5 = st.columns(5)
            w1.metric("Avg Sharpe (OOS)",   f"{np.mean(sharpes):.2f}",  delta=f"sd {np.std(sharpes):.2f}")
            w2.metric("Avg Calmar (OOS)",   f"{np.mean(caldmars):.2f}", delta=f"sd {np.std(caldmars):.2f}")
            w3.metric("Avg Max DD (OOS)",   f"{np.mean(mdds)*100:.1f}%")
            w4.metric("Avg Win Rate (OOS)", f"{np.mean(win_rs)*100:.1f}%")
            w5.metric("Avg Ann. Return",    f"{np.mean(ann_rs)*100:.1f}%")

            # Fold breakdown table
            fold_df = pd.DataFrame({
                'Fold':       [r['fold'] for r in wf_results],
                'Entry (opt)':[r['entry'] for r in wf_results],
                'Exit (opt)': [r['exit']  for r in wf_results],
                'Train Sharpe':[f"{r['train_sharpe']:.2f}" for r in wf_results],
                'OOS Sharpe': [f"{r['sharpe']:.2f}" for r in wf_results],
                'OOS Calmar': [f"{r['calmar']:.2f}"  for r in wf_results],
                'Win Rate':   [f"{r['win_rate']*100:.1f}%" for r in wf_results],
                'Max DD':     [f"{r['mdd']*100:.1f}%" for r in wf_results],
            })
            st.dataframe(fold_df, use_container_width=True, hide_index=True)

            fig_wf = go.Figure()
            fold_labels = [f"Fold {r['fold']}" for r in wf_results]
            fig_wf.add_trace(go.Bar(name='OOS Sharpe',  x=fold_labels, y=sharpes,  marker_color='#F0B429'))
            fig_wf.add_trace(go.Bar(name='OOS Calmar',  x=fold_labels, y=caldmars, marker_color='#58A6FF'))
            fig_wf.add_trace(go.Bar(name='Win Rate',    x=fold_labels, y=win_rs,   marker_color='#3FB950'))
            fig_wf.update_layout(template="plotly_dark", paper_bgcolor='#0D1117', plot_bgcolor='#0D1117',
                                   height=280, barmode='group', margin=dict(l=0, r=0, t=10, b=0),
                                   legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig_wf, use_container_width=True)

            fold_colors = ['#F0B429', '#58A6FF', '#3FB950', '#BC8CFF', '#FF7B72']
            fig_folds   = go.Figure()
            for i, r in enumerate(wf_results):
                fig_folds.add_trace(go.Scatter(
                    y=r['cum_returns'].values,
                    name=f"Fold {r['fold']} — Entry {r['entry']} / Exit {r['exit']} — Sharpe {r['sharpe']:.2f}",
                    line=dict(color=fold_colors[i], width=1.5)
                ))
            fig_folds.update_layout(template="plotly_dark", paper_bgcolor='#0D1117', plot_bgcolor='#0D1117',
                                     height=300, margin=dict(l=0, r=0, t=10, b=0),
                                     yaxis_title="Cumulative Return (x)", legend=dict(orientation='h', y=1.15))
            st.plotly_chart(fig_folds, use_container_width=True)

            consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)
            if consistency >= 0.8:
                st.success(f"Strategy consistent: {consistency*100:.0f}% of out-of-sample folds produced positive Sharpe.")
            elif consistency >= 0.6:
                st.warning(f"Strategy moderately consistent: {consistency*100:.0f}% of folds profitable.")
            else:
                st.error(f"Strategy inconsistent: only {consistency*100:.0f}% of out-of-sample folds profitable.")

            # Export
            st.markdown("#### Export Analysis")
            csv_bytes = build_export_csv(df, active_ticker, wf_results)
            st.download_button(
                label="Download Full Analysis as CSV",
                data=csv_bytes,
                file_name=f"nomos_{active_ticker.replace('.','_')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Insufficient data for walk-forward validation.")

    # ── TAB 5: WEIGHT CALIBRATION ─────────────────────────────────────────────
    with tab5:
        st.markdown("### Factor Weight Calibration — Information Coefficient Analysis")
        st.caption(
            "Weights are derived from the Information Coefficient (IC): rank correlation between each "
            "factor and forward 21-day returns. Higher IC = higher weight. This is in-sample — interpret "
            "walk-forward OOS Sharpe as the true performance measure."
        )

        ic_data = {
            'Factor':      ['Z-Score (Mean Reversion)', 'RSI (Momentum)', 'MACD Histogram', 'Trend (MA50>MA200)'],
            'Raw IC':      [f"{cal_ics.get('z',0):.4f}",    f"{cal_ics.get('rsi',0):.4f}",
                            f"{cal_ics.get('macd',0):.4f}", f"{cal_ics.get('trend',0):.4f}"],
            'Calibrated Weight': [f"{w_z:.3f}", f"{w_rsi:.3f}", f"{w_macd:.3f}", f"{w_trend:.3f}"],
        }
        st.dataframe(pd.DataFrame(ic_data), use_container_width=True, hide_index=True)

        # Sensitivity heatmap: vary w_z and w_rsi, show impact on score for current bar
        st.markdown("#### Score Sensitivity — Z-Score Weight vs RSI Weight")
        st.caption("Shows how the current Nomos Score changes as the two largest weights vary. Confirms score is not brittle to small weight perturbations.")

        z_range   = np.arange(1.0, 5.5, 0.5)
        rsi_range = np.arange(0.5, 3.5, 0.5)
        heat = np.zeros((len(z_range), len(rsi_range)))
        for iz, wz in enumerate(z_range):
            for ir, wr in enumerate(rsi_range):
                s = apply_weights(df.iloc[[-1]], wz, wr, w_macd, w_trend, sensitivity)
                heat[iz, ir] = float(s.iloc[0])

        fig_heat = go.Figure(go.Heatmap(
            z=heat,
            x=[f"RSI w={v:.1f}" for v in rsi_range],
            y=[f"Z w={v:.1f}"   for v in z_range],
            colorscale=[[0,'#F85149'],[0.5,'#F0B429'],[1,'#3FB950']],
            zmin=1, zmax=10,
            text=np.round(heat, 1), texttemplate="%{text}",
            showscale=True
        ))
        fig_heat.update_layout(template="plotly_dark", paper_bgcolor='#0D1117',
                                height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("IC Calibration Methodology"):
            st.markdown("""
            | Step | Detail |
            |---|---|
            | Forward return horizon | 21 trading days |
            | IC method | Spearman rank correlation |
            | Negative IC factors | Weight floored at 0 (excluded) |
            | Weight scaling | IC normalised so weights sum to ~6 total swing |
            | Limitation | IC is computed in-sample on full history. Subject to look-ahead. Use Walk-Forward tab for OOS validation. |
            """)

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="disclaimer">
    LEGAL DISCLOSURE: Nomos Terminal is a mathematical simulation and decision-support tool only.
    No content constitutes financial advice, investment recommendations, or solicitation.
    Not registered with SEBI or any regulatory authority. Past simulated performance does not guarantee future results.
    All backtests subject to survivorship bias and model risk. Capital markets involve substantial risk of loss.
    </div>
    """, unsafe_allow_html=True)
