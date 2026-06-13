"""
PDV Model — Real-Time SPX/VIX Dashboard

Run locally:
    python3.10 -m streamlit run app.py

Deploy free to Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io → New app → point to app.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from src.data.fetcher import get_market_snapshot

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PDV Model Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")

    refresh_mins = st.slider(
        "Auto-refresh interval (minutes)", min_value=1, max_value=30, value=5
    )
    st_autorefresh(interval=refresh_mins * 60 * 1000, key="autorefresh")

    n_spx = st.selectbox("SPX expiries to show", [1, 2, 3], index=1)

    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- SPX options: Yahoo Finance (yfinance)")
    st.markdown("- VIX level: Yahoo Finance")
    st.markdown("- VIX futures: *spot proxy* (⚠ paid feed needed for true futures)")
    st.divider()
    st.markdown("**Model**")
    st.markdown("Guyon-Lekeufack four-factor PDV")
    st.markdown("[Paper (Risk.net, Feb 2026)](https://www.risk.net)")


# ── Fetch data ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=refresh_mins * 60)
def load_snapshot(n_spx_expiries):
    return get_market_snapshot(n_spx_expiries=n_spx_expiries, n_vix_expiries=1)


with st.spinner("Fetching market data..."):
    try:
        snap = load_snapshot(n_spx)
        fetch_ok = True
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        fetch_ok = False

# ── Header: live market levels ────────────────────────────────────────────────

st.title("PDV Model — Live SPX / VIX Dashboard")
st.caption(f"Last updated: {snap['fetched_at'] if fetch_ok else '—'}")

if fetch_ok:
    lvl = snap["levels"]
    c1, c2, c3 = st.columns(3)
    c1.metric("S&P 500 (SPX)", f"{lvl['spx']:,.1f}")
    c2.metric("VIX", f"{lvl['vix']:.2f}")
    vix_fut = snap["vix_futures"]
    c3.metric(
        "VIX Futures proxy",
        f"{vix_fut[0]['futures_price']:.2f}" if vix_fut else "—",
        help="VIX spot used as front-month futures proxy. "
             "Use a paid feed for exact futures prices.",
    )

# ── SPX Implied Volatility Smiles ─────────────────────────────────────────────

st.subheader("SPX Implied Volatility Smile")

if fetch_ok and snap["spx_smiles"]:
    smiles = snap["spx_smiles"]
    spot   = snap["levels"]["spx"]

    fig = go.Figure()
    colours = ["royalblue", "tomato", "seagreen", "orange"]

    for i, smile in enumerate(smiles):
        moneyness = smile["strikes"] / spot
        col = colours[i % len(colours)]

        # Bid-ask band
        fig.add_trace(go.Scatter(
            x=np.concatenate([moneyness, moneyness[::-1]]),
            y=np.concatenate([smile["ask"], smile["bid"][::-1]]),
            fill="toself",
            fillcolor=col,
            opacity=0.12,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Mid IV
        fig.add_trace(go.Scatter(
            x=moneyness,
            y=smile["ivs"],
            mode="lines+markers",
            name=f"T = {smile['T']*365:.0f}d  ({smile['expiry']})",
            line=dict(color=col, width=2),
            marker=dict(size=5),
        ))
        # ATM line
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        xaxis_title="Moneyness K / S₀",
        yaxis_title="Implied Volatility",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(t=20, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Shaded bands show bid-ask spread. "
        "Downward slope = leverage effect (puts more expensive than calls). "
        "Dashed line = at-the-money."
    )
else:
    st.info("No SPX smile data available. Markets may be closed.")

# ── VIX Options Smile ─────────────────────────────────────────────────────────

st.subheader("VIX Implied Volatility Smile (vol-of-vol)")

if fetch_ok and snap["vix_smiles"]:
    vsmile = snap["vix_smiles"][0]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=vsmile["moneyness"],
        y=vsmile["ivs"],
        mode="lines+markers",
        name=f"T = {vsmile['T']*365:.0f}d  ({vsmile['expiry']})",
        line=dict(color="darkorchid", width=2),
        marker=dict(size=5),
    ))
    fig2.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.4)
    fig2.update_layout(
        xaxis_title="Moneyness K / F_VIX",
        yaxis_title="Implied Volatility of VIX",
        yaxis_tickformat=".0%",
        height=380,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "VIX smile slopes upward (right skew): VIX spikes are more extreme "
        "than VIX drops, making high-strike calls relatively expensive."
    )
else:
    st.info("No VIX options data available.")

# ── Raw data expander ─────────────────────────────────────────────────────────

with st.expander("Raw options data"):
    if fetch_ok and snap["spx_smiles"]:
        import pandas as pd
        for smile in snap["spx_smiles"]:
            st.markdown(f"**SPX — {smile['expiry']} (T = {smile['T']*365:.0f} days)**")
            df = pd.DataFrame({
                "Strike":    smile["strikes"],
                "Moneyness": (smile["strikes"] / snap["levels"]["spx"]).round(4),
                "Bid":       smile["bid"],
                "Ask":       smile["ask"],
                "IV (mid)":  smile["ivs"].round(4),
            })
            st.dataframe(df, use_container_width=True)

# ── Calibration placeholder ───────────────────────────────────────────────────

st.divider()
st.subheader("Model Calibration")

st.info(
    "**Coming in Phase 4.** Once the neural networks are trained on a full "
    "dataset, clicking 'Run Calibration' will find the model parameters "
    "(β₀, β₁, β₂, β₁₂, λ₁,₀, λ₁,₁, θ₁, λ₂,₀, λ₂,₁, θ₂) that best fit "
    "the current SPX smile and VIX data shown above. Pricing will reduce to "
    "a matrix-vector product — fast enough to re-run in real time.",
    icon="ℹ️",
)

col_a, col_b = st.columns(2)
with col_a:
    st.button("Run Calibration", disabled=True, help="Requires trained networks (Phase 4)")
with col_b:
    st.button("Export calibrated params", disabled=True)
