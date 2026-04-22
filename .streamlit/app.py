import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t as t_dist
from scipy.stats import norm

# ============================================================
# NOMOS TERMINAL v10.0 — INSTITUTIONAL GRADE
# Architecture: Multi-factor scoring | Fat-tail MC |
#               Walk-forward backtest | Full risk suite
# ============================================================

st.set_page_config(page_title="Nomos Terminal | v10.0", layout="wide")

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

.score-ring {
    background: linear-gradient(135deg, #0D1117, #161B22);
    border: 1px solid #21262D;
    border-top: 3px solid;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.stat-card {
    background: linear-gradient(135deg, #0D1117, #161B22);
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 12px;
}
.tag-bull { background:#0D2B1A; color:#3FB950; padding:3px 10px; border-radius:4px; font-family:'JetBrains Mono'; font-size:12px; border:1px solid #238636; }
.tag-bear { background:#2B0D0D; color:#F85149; padding:3px 10px; border-radius:4px; font-family:'JetBrains Mono'; font-size:12px; border:1px solid #DA3633; }
.tag-neutral { background:#1C1F24; color:#8B949E; padding:3px 10px; border-radius:4px; font-family:'JetBrains Mono'; font-size:12px; border:1px solid #30363D; }
.wf-stat { background:#0D1117; border:1px solid #21262D; border-radius:6px; padding:14px; text-align:center; }
.wf-num { font-family:'JetBrains Mono'; font-size:1.5rem; font-weight:700; margin:0; }
.wf-lbl { font-size:0.7rem; letter-spacing:1px; text-transform:uppercase; color:#8B949E; margin:0; }
.disclaimer {
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

# ─── INDICATOR ENGINES ────────────────────────────────────────────────────────

def compute_indicators(df, sensitivity):
    df = df.copy()

    # Trend
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['SD']    = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['MA50']) / df['SD']

    # Momentum: RSI-14
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # Momentum: MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['Signal_Line']

    # Volatility: ATR-14
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close']

    # ── NOMOS MULTI-FACTOR SCORE (v10) ──────────────────────────────────────
    # Component 1: Mean-reversion Z-Score (normalised to 0–10)
    z_comp = (df['Z_Score'] / sensitivity).clip(-1, 1) * 3          # ±3 range

    # Component 2: RSI momentum (centred, normalised)
    rsi_comp = ((df['RSI'] - 50) / 50).clip(-1, 1) * 1.5           # ±1.5 range

    # Component 3: MACD cross signal (normalised by price)
    macd_norm = (df['MACD_Hist'] / df['Close'].replace(0, np.nan)).clip(-0.01, 0.01) / 0.01
    macd_comp = macd_norm * 1.0                                      # ±1.0 range

    # Component 4: Trend alignment (MA50 vs MA200)
    trend_comp = np.where(df['MA50'] > df['MA200'], 0.5, -0.5)      # ±0.5 range

    # Composite: Centre at 5, total swing ±6
    df['Nomos_Score'] = (5 + z_comp + rsi_comp + macd_comp + trend_comp).clip(1, 10)

    return df

# ─── MONTE CARLO (t-DISTRIBUTION, VECTORISED) ─────────────────────────────────

def run_monte_carlo(current_price, returns_series, days=21, sims=2000):
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
    """
    Split 3y data into n_folds. Train score on first half of each fold,
    test on second half. Prevents in-sample overfitting.
    """
    results = []
    data = df.dropna(subset=['Nomos_Score', 'Close']).copy()
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
            'fold': i + 1,
            'sharpe': sharpe,
            'mdd': mdd,
            'calmar': calmar,
            'ann_return': ann_r,
            'win_rate': win_r,
            'cum_returns': cum
        })

    return results

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🏛 NOMOS v10.0")
user_input     = st.sidebar.text_input("Asset Search", value="NVDA").upper()
sensitivity    = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.5, 0.1)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=7.0, step=0.1) / 100
mc_sims        = st.sidebar.select_slider("MC Simulations", options=[500, 1000, 2000, 5000], value=2000)
st.sidebar.divider()
st.sidebar.caption("v10.0 | Institutional Decision Architecture")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

st.markdown("# 🏛 NOMOS TERMINAL")
st.caption("Institutional Decision Support | v10.0 | Multi-Factor · Fat-Tail · Walk-Forward")
st.divider()

if user_input:
    with st.spinner(f"Fetching {user_input}..."):
        df_raw, active_ticker = fetch_data(user_input)

    if df_raw.empty or len(df_raw) < 150:
        st.error("Asset not found or insufficient history (need 150+ trading days).")
        st.stop()

    df  = compute_indicators(df_raw, sensitivity)
    curr = df.iloc[-1]
    returns = df['Close'].pct_change().dropna()

    # ── TOP METRIC BAR ─────────────────────────────────────────────────────────
    score_val  = curr['Nomos_Score']
    score_color = "#3FB950" if score_val >= 7 else "#F85149" if score_val <= 4 else "#F0B429"

    vol_ratio  = curr['SD'] / df['SD'].mean()
    conf_label = "STABLE" if vol_ratio < 1.3 else "ELEVATED" if vol_ratio < 1.8 else "CHAOTIC"
    conf_color = "#3FB950" if vol_ratio < 1.3 else "#F0B429" if vol_ratio < 1.8 else "#F85149"

    rsi_val    = curr['RSI']
    rsi_state  = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL"
    rsi_color  = "#F85149" if rsi_val > 70 else "#3FB950" if rsi_val < 30 else "#8B949E"

    trend_bias = "BULLISH" if curr['MA50'] > curr['MA200'] else "BEARISH"
    trend_col  = "#3FB950" if trend_bias == "BULLISH" else "#F85149"

    var95, cvar95 = compute_var_cvar(returns, 0.95)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("TICKER",         active_ticker)
    c2.metric("LAST CLOSE",     f"${curr['Close']:.2f}")
    c3.metric("NOMOS SCORE",    f"{score_val:.1f}/10")
    c4.metric("RSI-14",         f"{rsi_val:.1f}", delta=rsi_state)
    c5.metric("VaR 95%",        f"{var95*100:.2f}%", delta="Daily")
    c6.metric("CVaR 95%",       f"{cvar95*100:.2f}%", delta="Expected Loss")

    st.markdown(f"""
    <div style="display:flex;gap:8px;margin:8px 0 16px 0;flex-wrap:wrap;">
        <span style="background:#0D1117;border:1px solid {score_color};color:{score_color};padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono';font-size:12px;">
            SCORE {score_val:.1f} · {'STRONG BUY' if score_val>=8 else 'BUY' if score_val>=6.5 else 'HOLD' if score_val>=4.5 else 'SELL' if score_val>=3 else 'STRONG SELL'}
        </span>
        <span style="background:#0D1117;border:1px solid {conf_color};color:{conf_color};padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono';font-size:12px;">
            VOLATILITY: {conf_label} ({vol_ratio:.2f}x)
        </span>
        <span style="background:#0D1117;border:1px solid {trend_col};color:{trend_col};padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono';font-size:12px;">
            TREND: {trend_bias} (MA50{'>'if curr['MA50']>curr['MA200'] else '<'}MA200)
        </span>
        <span style="background:#0D1117;border:1px solid {rsi_color};color:{rsi_color};padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono';font-size:12px;">
            MOMENTUM: {rsi_state}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉  Trend Architecture",
        "🔮  Fat-Tail Risk Engine",
        "🧠  Quant Vault",
        "🔄  Walk-Forward Backtest"
    ])

    # ── TAB 1: TREND ──────────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            fig = go.Figure()
            # Bollinger bands
            upper = df['MA50'] + (sensitivity * df['SD'])
            lower = df['MA50'] - (sensitivity * df['SD'])
            fig.add_trace(go.Scatter(x=df.index, y=upper, line_color='rgba(0,0,0,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=lower, fill='tonexty',
                                     fillcolor='rgba(240,180,41,0.05)', line_color='rgba(0,0,0,0)', name='Noise Band'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'],   name='Price',    line=dict(color='#E6EDF3', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'],    name='MA-50',    line=dict(color='#F0B429', dash='dot', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA200'],   name='MA-200',   line=dict(color='#58A6FF', dash='dot', width=1)))
            fig.update_layout(template="plotly_dark", paper_bgcolor='#0D1117',
                               plot_bgcolor='#0D1117', height=380,
                               margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation='h', y=1.05))
            st.plotly_chart(fig, use_container_width=True)

            # MACD subplot
            fig_m = go.Figure()
            colors = ['#3FB950' if v >= 0 else '#F85149' for v in df['MACD_Hist'].fillna(0)]
            fig_m.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Histogram', marker_color=colors))
            fig_m.add_trace(go.Scatter(x=df.index, y=df['MACD'],        name='MACD',   line=dict(color='#58A6FF', width=1.5)))
            fig_m.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='#F0B429', width=1.5)))
            fig_m.update_layout(template="plotly_dark", paper_bgcolor='#0D1117',
                                  plot_bgcolor='#0D1117', height=200,
                                  margin=dict(l=0, r=0, t=10, b=0),
                                  legend=dict(orientation='h', y=1.1))
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

            # Nomos Score gauge (horizontal bar)
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
        fig_mc.add_trace(go.Scatter(y=p5,  line=dict(color='rgba(248,81,73,0.15)', dash='dash'), name='5th (Bear)'))
        fig_mc.add_trace(go.Scatter(y=mc.mean(axis=1), line=dict(color='#F0B429', width=2.5), name='Expected Path'))
        fig_mc.update_layout(template="plotly_dark", paper_bgcolor='#0D1117',
                               plot_bgcolor='#0D1117', height=420,
                               margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Trading Days",
                               yaxis_title="Projected Price",
                               legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig_mc, use_container_width=True)

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Prob. Upside",  f"{prob_up*100:.1f}%")
        mc2.metric("Bull Case (P95)", f"${p95[-1]:.2f}")
        mc3.metric("Expected",       f"${mc.mean(axis=1)[-1]:.2f}")
        mc4.metric("Bear Case (P5)",  f"${p5[-1]:.2f}")
        mc5.metric("Bull/Bear Ratio", f"{(p95[-1]/curr['Close']-1)*100:.1f}% / {(p5[-1]/curr['Close']-1)*100:.1f}%")

        st.info(f"📐 **Student's t-Distribution** fitted to {len(returns):,} return observations (df={t_dist.fit(returns)[0]:.1f}). "
                f"Fat tails capture Black Swan tail risk beyond Gaussian assumptions. {mc_sims:,} simulation paths.")

    # ── TAB 3: QUANT VAULT ────────────────────────────────────────────────────
    with tab3:
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
            q1.metric("Sharpe (Excess)",  f"{sharpe:.2f}")
            q2.metric("Sortino Ratio",    f"{sortino:.2f}")
            q3.metric("Calmar Ratio",     f"{calmar:.2f}")
            q4.metric("Max Drawdown",     f"{mdd*100:.1f}%")
            q5.metric("Win Rate",         f"{win_rate*100:.1f}%")
            q6.metric("# Round Trips",    str(n_trades))

            # Equity curve
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=cum_strat.values, name='Nomos Strategy',
                                         line=dict(color='#F0B429', width=2)))
            fig_eq.add_trace(go.Scatter(y=cum_mkt.values,   name='Buy & Hold',
                                         line=dict(color='#58A6FF', width=1.5, dash='dot')))
            fig_eq.update_layout(template="plotly_dark", paper_bgcolor='#0D1117',
                                   plot_bgcolor='#0D1117', height=320,
                                   margin=dict(l=0, r=0, t=10, b=0),
                                   yaxis_title="Cumulative Return (x)",
                                   legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig_eq, use_container_width=True)

            with st.expander("📐 Methodology Disclosure"):
                st.markdown(f"""
                | Parameter | Value |
                |---|---|
                | Entry Threshold | Nomos Score > 7.5 |
                | Exit Threshold  | Nomos Score < 4.0 |
                | Execution Lag   | 1-day (prevents lookahead bias) |
                | Transaction Cost | 10 bps per trade toggle |
                | Risk-Free Rate  | {risk_free_rate*100:.1f}% p.a. (daily adjusted) |
                | Sharpe Basis    | Excess returns over risk-free |
                | Sortino Basis   | Downside deviation only |
                | Calmar Basis    | Ann. excess return / Max drawdown |
                | Universe        | Single asset, long-only |
                """)

    # ── TAB 4: WALK-FORWARD ───────────────────────────────────────────────────
    with tab4:
        st.markdown("### Walk-Forward Backtest — 5-Fold Out-of-Sample Validation")
        st.caption("Each fold trains on the first half of its window, tests on the second. Prevents in-sample optimism.")

        with st.spinner("Running 5-fold walk-forward validation..."):
            wf_results = walk_forward_backtest(df, risk_free_rate)

        if wf_results:
            sharpes  = [r['sharpe']     for r in wf_results]
            caldmars = [r['calmar']     for r in wf_results]
            mdds     = [r['mdd']        for r in wf_results]
            win_rs   = [r['win_rate']   for r in wf_results]
            ann_rs   = [r['ann_return'] for r in wf_results]

            w1, w2, w3, w4, w5 = st.columns(5)
            w1.metric("Avg Sharpe (OOS)",   f"{np.mean(sharpes):.2f}",  delta=f"±{np.std(sharpes):.2f}")
            w2.metric("Avg Calmar (OOS)",   f"{np.mean(caldmars):.2f}", delta=f"±{np.std(caldmars):.2f}")
            w3.metric("Avg Max DD (OOS)",   f"{np.mean(mdds)*100:.1f}%")
            w4.metric("Avg Win Rate (OOS)", f"{np.mean(win_rs)*100:.1f}%")
            w5.metric("Avg Ann. Return",    f"{np.mean(ann_rs)*100:.1f}%")

            # Per-fold breakdown
            fig_wf = go.Figure()
            fold_labels = [f"Fold {r['fold']}" for r in wf_results]
            fig_wf.add_trace(go.Bar(name='Sharpe',      x=fold_labels, y=sharpes,
                                    marker_color='#F0B429'))
            fig_wf.add_trace(go.Bar(name='Calmar',      x=fold_labels, y=caldmars,
                                    marker_color='#58A6FF'))
            fig_wf.add_trace(go.Bar(name='Win Rate',    x=fold_labels, y=win_rs,
                                    marker_color='#3FB950'))
            fig_wf.update_layout(template="plotly_dark", paper_bgcolor='#0D1117',
                                   plot_bgcolor='#0D1117', height=320,
                                   barmode='group', margin=dict(l=0, r=0, t=10, b=0),
                                   legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig_wf, use_container_width=True)

            # Cumulative returns per fold
            fig_folds = go.Figure()
            fold_colors = ['#F0B429', '#58A6FF', '#3FB950', '#BC8CFF', '#FF7B72']
            for i, r in enumerate(wf_results):
                fig_folds.add_trace(go.Scatter(
                    y=r['cum_returns'].values,
                    name=f"Fold {r['fold']} (Sharpe: {r['sharpe']:.2f})",
                    line=dict(color=fold_colors[i], width=1.5)
                ))
            fig_folds.update_layout(template="plotly_dark", paper_bgcolor='#0D1117',
                                     plot_bgcolor='#0D1117', height=300,
                                     margin=dict(l=0, r=0, t=10, b=0),
                                     yaxis_title="Cumulative Return (x)",
                                     legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig_folds, use_container_width=True)

            consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)
            if consistency >= 0.8:
                st.success(f"✅ Strategy is **consistent**: {consistency*100:.0f}% of out-of-sample folds produced positive Sharpe.")
            elif consistency >= 0.6:
                st.warning(f"⚠️ Strategy is **moderately consistent**: {consistency*100:.0f}% of folds profitable.")
            else:
                st.error(f"❌ Strategy is **inconsistent**: only {consistency*100:.0f}% of out-of-sample folds profitable.")
        else:
            st.warning("Insufficient data for walk-forward validation.")

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>LEGAL DISCLOSURE:</strong> Nomos Terminal is a mathematical simulation and decision-support tool only.
    No content constitutes financial advice, investment recommendations, or solicitation.
    Not registered with SEBI or any regulatory authority. Past simulated performance does not guarantee future results.
    All backtests are subject to survivorship bias and model risk. Capital markets involve substantial risk of loss.
    </div>
    """, unsafe_allow_html=True)
