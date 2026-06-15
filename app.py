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

import torch
from src.data.fetcher import get_market_snapshot
from src.training.networks import build_spx_network, build_vix_network
from src.calibration.calibrate import calibrate

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

# ── Calibration ───────────────────────────────────────────────────────────────

st.divider()
st.subheader("Model Calibration")

# Network file uploaders
with st.expander("Load trained networks", expanded=True):
    st.caption(
        "Upload the .pt checkpoint files saved after Phase 3 training. "
        "Without these the button stays disabled."
    )
    col_u1, col_u2 = st.columns(2)
    spx_file = col_u1.file_uploader("SPX network (.pt)", type="pt")
    vix_file  = col_u2.file_uploader("VIX network (.pt)", type="pt")


@st.cache_resource
def load_networks(spx_bytes, vix_bytes):
    spx_net = build_spx_network()
    vix_net = build_vix_network()
    spx_net.load_state_dict(torch.load(spx_bytes, map_location='cpu', weights_only=True))
    vix_net.load_state_dict(torch.load(vix_bytes, map_location='cpu', weights_only=True))
    spx_net.eval()
    vix_net.eval()
    return spx_net, vix_net


networks_ready = spx_file is not None and vix_file is not None

col_a, col_b, col_c = st.columns([2, 2, 3])
n_steps = col_b.slider("Gradient steps", 50, 500, 200, disabled=not networks_ready)
run_btn = col_a.button("Run Calibration", disabled=not networks_ready)

if run_btn and networks_ready and fetch_ok:
    spx_net, vix_net = load_networks(spx_file, vix_file)

    progress_bar = st.progress(0)
    loss_display = st.empty()
    loss_log     = []

    def _cb(step, loss_val):
        loss_log.append(loss_val)
        progress_bar.progress(step / n_steps)
        loss_display.caption(f"Step {step}/{n_steps} — loss {loss_val:.5f}")

    with st.spinner("Calibrating..."):
        cal_params, loss_hist = calibrate(
            spx_net, vix_net, snap,
            n_steps=n_steps, verbose=False,
            progress_callback=_cb,
        )

    st.success("Calibration complete.")
    st.session_state['cal_params']  = cal_params
    st.session_state['loss_history'] = loss_hist

# Display results if calibration has been run this session
if 'cal_params' in st.session_state:
    import pandas as pd
    p = st.session_state['cal_params']

    st.markdown("**Calibrated parameters**")
    param_df = pd.DataFrame({
        "Parameter": ["β₀", "β₁", "β₂", "β₁₂",
                      "λ₁,₀", "λ₁,₁", "θ₁",
                      "λ₂,₀", "λ₂,₁", "θ₂"],
        "Value": [p.beta0, p.beta1, p.beta2, p.beta12,
                  p.lam1_0, p.lam1_1, p.theta1,
                  p.lam2_0, p.lam2_1, p.theta2],
        "Meaning": [
            "Baseline volatility",
            "Trend coefficient (leverage)",
            "Variability coefficient",
            "Parabolic (right-tail) term",
            "R₁ fast decay rate",
            "R₁ slow decay rate",
            "R₁ mixing weight",
            "R₂ fast decay rate",
            "R₂ slow decay rate",
            "R₂ mixing weight",
        ],
    })
    param_df["Value"] = param_df["Value"].round(4)
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    # Loss curve
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(
        y=st.session_state['loss_history'],
        mode='lines', line=dict(color='steelblue', width=2),
        name='Joint loss',
    ))
    loss_fig.update_layout(
        xaxis_title="Gradient step",
        yaxis_title="Loss",
        height=250,
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(loss_fig, use_container_width=True)

    # VIX term structure — every trading day for 6 months
    st.markdown("**Model-implied VIX term structure (6-month daily horizon)**")
    import torch as _torch
    from src.calibration.calibrate import params_to_raw, raw_to_params_tensor, _build_inputs

    _, vix_net_ts = load_networks(spx_file, vix_file)
    raw_p = params_to_raw(p)
    with _torch.no_grad():
        theta_M = raw_to_params_tensor(raw_p)
        theta14 = _torch.cat([theta_M,
                               _torch.tensor([p.R1_0_0, p.R1_1_0,
                                              p.R2_0_0, p.R2_1_0])])
        # 126 trading days = ~6 calendar months
        N = 126
        T_grid = _torch.linspace(1 / 252, 126 / 252, N)
        m_grid = _torch.ones(N)   # ATM moneyness → futures price
        inputs_ts = _build_inputs(theta14, T_grid, m_grid)
        futures_mdl = vix_net_ts(inputs_ts)[:, 0].numpy()

    cal_days = np.round(T_grid.numpy() * 365).astype(int)
    vix_now  = snap['levels']['vix']

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=cal_days, y=futures_mdl,
        mode='lines', name='Model fair value',
        line=dict(color='steelblue', width=2),
    ))
    for fut in snap.get('vix_futures', []):
        fig_ts.add_trace(go.Scatter(
            x=[round(fut['T'] * 365)], y=[fut['futures_price']],
            mode='markers', name=f"Market proxy ({fut['expiry']})",
            marker=dict(color='tomato', size=10, symbol='circle'),
        ))
    fig_ts.add_hline(
        y=vix_now, line_dash='dash', line_color='gray', opacity=0.5,
        annotation_text=f"VIX spot {vix_now:.1f}",
        annotation_position="bottom right",
    )
    fig_ts.update_layout(
        xaxis_title="Calendar days from today",
        yaxis_title="VIX futures fair value",
        height=360,
        margin=dict(t=10, b=40),
        hovermode='x unified',
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption(
        "Smooth curve = model fair value for a VIX futures contract expiring on that day. "
        "Red dot = current market proxy price. "
        "Dashed line = VIX spot. "
        "An upward-sloping curve (contango) is normal; a downward slope (backwardation) signals stress."
    )

    # Model smile overlay on SPX chart
    if fetch_ok and snap['spx_smiles'] and 'spx_net' in dir():
        st.markdown("**Model vs. market (SPX smile)**")
        fig_overlay = go.Figure()
        colours = ["royalblue", "tomato"]
        spot = snap['levels']['spx']

        for i, smile in enumerate(snap['spx_smiles'][:2]):
            col = colours[i % 2]
            moneyness = smile['strikes'] / spot
            # Market
            fig_overlay.add_trace(go.Scatter(
                x=moneyness, y=smile['ivs'],
                mode='markers', name=f"Market {smile['expiry']}",
                marker=dict(color=col, size=6, symbol='circle-open'),
            ))
            # Model
            import torch as _torch
            from src.calibration.calibrate import _build_inputs, raw_to_params_tensor
            T_v  = _torch.full((len(smile['strikes']),), smile['T'], dtype=_torch.float32)
            K_v  = _torch.tensor(smile['strikes'] / spot, dtype=_torch.float32)
            from src.calibration.calibrate import params_to_raw
            raw_p = params_to_raw(p)
            from src.calibration.calibrate import raw_to_params_tensor, compute_R_factors
            theta_M = raw_to_params_tensor(raw_p)
            with _torch.no_grad():
                inputs = _build_inputs(
                    _torch.cat([theta_M,
                                _torch.tensor([p.R1_0_0, p.R1_1_0,
                                               p.R2_0_0, p.R2_1_0])]),
                    T_v, K_v)
                iv_mdl = spx_net(inputs).squeeze(1).numpy()
            fig_overlay.add_trace(go.Scatter(
                x=moneyness, y=iv_mdl,
                mode='lines', name=f"Model {smile['expiry']}",
                line=dict(color=col, width=2),
            ))

        fig_overlay.update_layout(
            xaxis_title="Moneyness K / S₀",
            yaxis_title="Implied Volatility",
            yaxis_tickformat=".0%",
            height=380,
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
