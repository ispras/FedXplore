"""Visual primitives shared by the FedXplore Streamlit workbench.

Keeping tokens and chart defaults here makes the research UI predictable across
the dashboard, individual-run analytics and comparison views.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "app": "#F7F8FA",
    "surface": "#FFFFFF",
    "text": "#111827",
    "muted": "#667085",
    "border": "#E4E7EC",
    "border_hover": "#D0D5DD",
    "accent": "#0F766E",
    "accent_soft": "#F0FDFA",
    "danger": "#B42318",
}
# Keep enough distinct hues for the six-run personalization Example before
# cycling the palette.  In particular, FedAvg and FedAMP must not both become
# teal merely because they are the first and sixth displayed series.
CHART_PALETTE = ["#0F766E", "#2563EB", "#D97706", "#7C3AED", "#DC2626", "#0891B2"]


def inject_global_styles() -> None:
    """Install the neutral application theme once per Streamlit rerun."""

    st.markdown(
        """
        <style>
        header[data-testid="stHeader"], div[data-testid="stDecoration"] { display: none; }
        .stApp { background: #F7F8FA; color: #111827; }
        .main .block-container, [data-testid="stMainBlockContainer"] {
            max-width: 1500px; padding-top: 0 !important; padding-bottom: 2rem;
            margin-top: -.75rem;
        }
        /* Invisible helper components are inserted before every page.  Their
           zero-height iframes must not leave vertical element margins behind. */
        [data-testid="stElementContainer"]:has(iframe[height="0"]),
        [data-testid="stElementContainer"]:has(iframe.stCustomComponentV1),
        [data-testid="stElementContainer"]:has(style),
        [data-testid="stElementContainer"]:has([data-testid="stCustomComponentV1"] iframe[height="0"]) {
            display: none !important;
        }
        .fx-brand-title { margin: 0; font-weight: 720; line-height: 1.15; color: #111827; }
        .fx-brand-fed, .fx-brand-xplore { color: #111827; }
        .fx-brand-suffix { color: #667085; font-weight: 600; }
        [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E4E7EC; }
        [data-testid="stSidebar"] .stButton > button { margin-bottom: .15rem; }

        .stButton > button {
            min-height: 2.35rem; border: 1px solid #D0D5DD; border-radius: 8px;
            background: #FFFFFF; color: #344054; box-shadow: none; font-weight: 600;
            white-space: nowrap;
            transition: border-color 120ms ease, background-color 120ms ease, box-shadow 120ms ease;
        }
        .stButton > button:hover:not(:disabled) {
            border-color: #98A2B3; background: #F9FAFB; box-shadow: 0 1px 2px rgba(16,24,40,.06);
        }
        .stButton > button.fx-button-primary {
            background: #0F766E !important; border-color: #0F766E !important; color: #FFFFFF !important;
        }
        .stButton > button.fx-button-primary:hover:not(:disabled) { background: #115E59 !important; }
        .stButton > button.fx-button-examples {
            background:#7C3AED !important; border-color:#7C3AED !important; color:#FFFFFF !important;
        }
        .stButton > button.fx-button-examples:hover:not(:disabled) { background:#6D28D9 !important; border-color:#6D28D9 !important; }
        .stButton > button.fx-button-danger {
            color: #B42318 !important; border-color: #FDA29B !important; background: #FFFFFF !important;
        }
        .stButton > button.fx-button-danger:hover:not(:disabled) { background: #FEF3F2 !important; }
        .stButton > button:disabled { opacity: .55; box-shadow: none; }

        .fx-topline { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.8rem; }
        .fx-page-actions { display:flex; align-items:center; justify-content:flex-end; gap:.6rem; }
        .fx-selection-count { color:#667085; font-size:.875rem; white-space:nowrap; padding-top:.4rem; }
        .fx-card, .fx-chart-card, .fx-panel {
            background:#FFFFFF; border:1px solid #E4E7EC; border-radius:10px; box-shadow:none;
        }
        .fx-card { padding:1rem 1.05rem; }
        .fx-card-label { color:#667085; font-size:.8rem; font-weight:600; margin-bottom:.35rem; }
        .fx-card-value { color:#111827; font-size:1.55rem; font-weight:700; line-height:1.1; }
        .fx-headline-card { min-height:92px; }
        .fx-chart-card { padding:0; margin-bottom:.25rem; }
        [class*="st-key-metric-card-"] { background:#FFFFFF; border:1px solid #E4E7EC; border-radius:10px; padding:.9rem .95rem .25rem; margin-bottom:1rem; }
        .fx-chart-title { color:#111827; font-size:1rem; font-weight:650; margin:0 0 .1rem; }
        .fx-chart-subtitle { color:#667085; font-size:.8rem; margin:0 0 .25rem; }
        .fx-panel { padding:.8rem .9rem; }
        [class*="st-key-create-card-"] {
            background:#FFFFFF; border:1px solid #E4E7EC; border-radius:10px;
            padding:1rem 1.05rem .85rem; margin-bottom:.8rem;
        }
        .fx-create-card-title { color:#111827; font-size:1rem; font-weight:700; margin-bottom:.14rem; }
        .fx-create-card-description { color:#667085; font-size:.84rem; margin-bottom:.7rem; }
        .st-key-create-summary {
            background:#FFFFFF; border:1px solid #E4E7EC; border-radius:10px;
            padding:1rem; position:sticky; top:.75rem;
        }
        .fx-summary-title { color:#111827; font-size:1rem; font-weight:700; margin-bottom:.7rem; }
        .fx-summary-row { display:grid; gap:.1rem; padding:.4rem 0; border-top:1px solid #F2F4F7; }
        .fx-summary-label { color:#667085; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.03em; }
        .fx-summary-value { color:#344054; font-size:.9rem; font-weight:600; }
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-list-"]) {
            border:1px solid #E4E7EC; border-radius:8px; padding:.35rem .45rem .25rem;
            background:#FFFFFF; margin:0 0 .42rem; max-width:none; position:relative;
            transition:opacity 120ms ease, border-color 120ms ease, background-color 120ms ease;
        }
        [class*="st-key-research-catalog-federated_method-"]:not([class*="st-key-research-catalog-list-"]) {
            border-radius:7px; padding:.14rem .3rem .08rem; margin:0 auto .24rem; max-width:17.5rem;
        }
        /* Only method cards use an invisible full-card button.  The other
           catalogs deliberately retain ordinary, generously sized buttons. */
        [class*="st-key-research-catalog-federated_method-"]:not([class*="st-key-research-catalog-list-"]) [data-testid="stElementContainer"]:has(> .stButton) {
            position:absolute; inset:0; z-index:3; margin:0 !important;
        }
        [class*="st-key-research-catalog-federated_method-"]:not([class*="st-key-research-catalog-list-"]) .stButton,
        [class*="st-key-research-catalog-federated_method-"]:not([class*="st-key-research-catalog-list-"]) .stButton > button {
            width:100%; height:100%; min-height:100%; border:0; background:transparent; color:transparent; padding:0; box-shadow:none;
        }
        /* Short catalogs use the same compact visual rhythm as methods, but
           keep visible native buttons instead of a full-card overlay. */
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-federated_method-"]):not([class*="st-key-research-catalog-list-"]) {
            border:0; background:transparent; padding:0; margin:0 0 .24rem auto; max-width:15.5rem; 
        }
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-federated_method-"]):not([class*="st-key-research-catalog-list-"]) .stButton > button {
            min-height:1.85rem; padding:.05rem .25rem; font-size:.88rem; line-height:1.2;
            border:1px solid #E4E7EC; border-radius:7px; background:#FFFFFF; color:#475467;
        }
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-federated_method-"]):not([class*="st-key-research-catalog-list-"])[class*="-selected"] .stButton > button {
            border-color:#0F766E; background:#F0FDFA; color:#115E59;
        }
        .fx-research-catalog-name { color:#475467; font-size:.92rem; line-height:1.25; text-align:center; padding:.14rem .25rem; }
        [class*="st-key-research-catalog-federated_method-"] .fx-research-catalog-name { font-size:.88rem; line-height:1.2; padding:.1rem .2rem; }
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-list-"]):hover { border-color:#98A2B3; background:#F9FAFB; }
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-list-"])[class*="-selected"] { border-color:#0F766E; background:#F0FDFA; }
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-list-"])[class*="-selected"] .fx-research-catalog-name { color:#115E59; }
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-list-"])[class*="-muted"] { opacity:.74; }
        [class*="st-key-research-catalog-"]:not([class*="st-key-research-catalog-list-"])[class*="-muted"]:hover { opacity:1; }
        .st-key-research-catalog-list-federated_method { max-height:68vh; overflow-y:auto; padding-right:.25rem; }
        .fx-research-tags { display:flex; flex-wrap:wrap; gap:.18rem; padding:0 .16rem .08rem; }
        .fx-research-tag { display:inline-flex; align-items:center; gap:.22rem; padding:.07rem .26rem; border-radius:999px; background:var(--tag-bg); color:#344054; font-size:.62rem; font-weight:650; }
        .fx-research-tag i { width:.3rem; height:.3rem; border-radius:50%; background:var(--tag-dot); }
        [class*="st-key-template-card-"] {
            background:#FFFFFF; border:1px solid #E4E7EC; border-radius:10px;
            padding:.9rem 1rem .65rem; min-height:10rem; margin-bottom:.7rem;
        }
        [class*="st-key-template-card-"]:has(button:hover), [class*="st-key-template-card-"]:hover {
            border-color:#0F766E; box-shadow:0 1px 3px rgba(16,24,40,.08);
        }
        [class*="st-key-template-card-selected-"] { border-color:#0F766E; background:#F0FDFA; }
        [class*="st-key-example-card-"] {
            background:#FFFFFF; border:1px solid #E4E7EC; border-radius:12px;
            padding:1.05rem 1.1rem .9rem; min-height:20rem; margin-bottom:.8rem;
            transition:border-color 120ms ease, background-color 120ms ease, box-shadow 120ms ease;
        }
        [class*="st-key-example-card-"]:hover { border-color:#C4B5FD; box-shadow:0 2px 6px rgba(16,24,40,.08); }
        [class*="st-key-example-card-selected-"] { border-color:#7C3AED; background:#FAF5FF; }
        .stApp:has(.fx-examples-page-marker) [data-testid="stMainBlockContainer"] { max-width:1900px; }
        [data-testid="stElementContainer"]:has(.fx-examples-page-marker) { display:none !important; }
        .fx-example-category { color:#7C3AED; font-size:.72rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.45rem; }
        .fx-example-title { color:#111827; font-size:1.2rem; font-weight:720; line-height:1.24; min-height:3rem; margin-bottom:.55rem; }
        [class*="st-key-example-card-"] [data-testid="stImage"] img {
            width:100%; aspect-ratio:1672 / 941; object-fit:cover; border-radius:8px;
        }
        .fx-example-description { color:#475467; font-size:.9rem; line-height:1.5; min-height:6.1rem; }
        .fx-example-tags { display:flex; flex-wrap:wrap; gap:.3rem; margin:.85rem 0 .8rem; }
        .fx-example-tag { background:#F5F3FF; border-radius:999px; color:#5B21B6; font-size:.72rem; font-weight:650; padding:.16rem .45rem; }
        .fx-example-footer { border-top:1px solid #EDE9FE; color:#667085; font-size:.8rem; margin-top:.75rem; padding-top:.7rem; }
        .fx-example-context { background:#FAF5FF; border:1px solid #DDD6FE; border-radius:10px; padding:.75rem .9rem; margin:.15rem 0 .8rem; }
        .fx-example-context-title { color:#5B21B6; font-weight:720; margin-bottom:.18rem; }
        [data-testid="stRadio"] [role="radiogroup"] { gap:.55rem; }

        .fx-status { display:inline-block; border-radius:999px; padding:.18rem .65rem; font-size:.76rem; font-weight:650; }
        .fx-status.running { background:#ECFDF3; color:#027A48; }
        .fx-status.stopping { background:#FFFAEB; color:#B54708; }
        .fx-status.stopped, .fx-status.finished, .fx-status.default { background:#F2F4F7; color:#475467; }
        .fx-status.failed_to_start, .fx-status.missing_status, .fx-status.invalid_status, .fx-status.missing_pid { background:#FEF3F2; color:#B42318; }

        .fx-table { border:1px solid #E4E7EC; border-radius:10px; background:#FFFFFF; padding:.45rem .7rem; }
        .fx-table-header { color:#667085; font-size:.72rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; margin:.15rem 0; }
        .fx-divider { height:1px; background:#EAECF0; margin:.12rem 0; }
        .fx-section { margin-top:.5rem; padding-top:.15rem; }
        .fx-param-card { border:1px solid #E4E7EC; border-radius:10px; padding:.8rem .9rem .25rem; background:#FFFFFF; margin-bottom:.7rem; }
        .fx-kv { display:grid; grid-template-columns:180px 1fr; gap:.55rem 1rem; align-items:start; }
        .fx-kv-label { color:#667085; font-weight:600; }
        .fx-detail-hero { padding:.25rem 0 1rem; border-bottom:1px solid #E4E7EC; margin-bottom:1rem; }
        .fx-detail-title { color:#111827; font-size:1.65rem; font-weight:700; line-height:1.15; margin-bottom:.2rem; }
        .fx-detail-subtitle { color:#667085; font-size:.86rem; }
        .fx-detail-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.9rem 1.25rem; margin-top:.9rem; }
        .fx-detail-item { min-width:0; }
        .fx-detail-label { color:#667085; font-size:.72rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; margin-bottom:.18rem; }
        .fx-detail-value { color:#344054; font-size:.92rem; font-weight:600; line-height:1.35; word-break:break-word; }
        .fx-step-note { color:#667085; margin-bottom:.55rem; }
        .fx-gpu-panel { border:1px solid #E4E7EC; border-radius:10px; padding:.85rem 1rem; background:#FFFFFF; margin-bottom:.65rem; }
        .fx-gpu-head { display:flex; justify-content:space-between; gap:1rem; margin-bottom:.55rem; font-weight:600; color:#344054; }
        .fx-gpu-bar { height:8px; border-radius:999px; background:#EAECF0; overflow:hidden; margin-bottom:.45rem; }
        .fx-gpu-bar > span { display:block; height:100%; border-radius:999px; background:#0F766E; }
        button[data-baseweb="tab"] { min-height:2.8rem; padding:.65rem 1rem; font-size:.92rem; font-weight:650; }
        [data-testid="stDataFrame"] { border:1px solid #E4E7EC; border-radius:8px; overflow:hidden; }

        /* Run rows use keyed containers: selection space remains reserved while
           the checkbox itself appears only on hover or after selection. */
        [class*="st-key-run-row-"] {
            border:1px solid transparent; border-radius:8px; padding:.06rem .18rem;
            margin:0 -.18rem; transition:background-color 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
        }
        [class*="st-key-run-row-"]:hover { background:#FFFFFF; border-color:#D0D5DD; box-shadow:0 1px 3px rgba(16,24,40,.08); }
        [class*="st-key-run-row-"] [data-testid="stCheckbox"] { opacity:0; transition:opacity 120ms ease; }
        [class*="st-key-run-row-"]:hover [data-testid="stCheckbox"],
        [class*="st-key-run-row-"]:has(input:checked) [data-testid="stCheckbox"] { opacity:1; }
        [class*="st-key-run-row-"]:has(input:checked) { background:#F0FDFA; border-color:#0F766E; }
        [class*="st-key-run-row-"] p { margin-bottom:0; }
        [class*="st-key-run-row-"] [data-testid="stHorizontalBlock"] { align-items:center; }
        [class*="st-key-run-row-"] [data-testid="stColumn"] [data-testid="stVerticalBlock"] {
            height:2.25rem; min-height:2.25rem; display:flex; flex-direction:column; justify-content:center;
        }
        [class*="st-key-run-row-"] [data-testid="stElementContainer"] { margin-bottom:0 !important; }
        [class*="st-key-run-row-"] [data-testid="stCheckbox"] {
            margin:0; padding-top:0; transform:translateY(-.2rem);
        }
        [class*="st-key-run-row-"] [data-testid="stCheckbox"] label {
            min-height:1.3rem; height:1.3rem; margin:0; padding:0; display:flex; align-items:center;
        }
        [class*="st-key-run-row-"] [data-testid="stCheckbox"] label > div { margin-top:0; }
        [class*="st-key-run-row-"] [data-testid="stMarkdownContainer"] > p {
            line-height:1.3rem; margin:0;
        }
        /* Streamlit's markdown wrapper keeps a baseline offset inside a
           compact row. Move only text/status content, not the checkbox or
           Open button, onto their shared visual centre line. */
        [class*="st-key-run-row-"] [data-testid="stMarkdownContainer"] {
            transform:translateY(-.45rem);
        }
        [class*="st-key-run-row-"] .stButton > button [data-testid="stMarkdownContainer"] {
            transform:none;
        }
        [class*="st-key-run-row-"] .stButton > button { min-height:2rem; }
        .st-key-dashboard-runs-table > div > [data-testid="stElementContainer"] { margin-bottom:0 !important; }

        .st-key-compare-selected-runs { background:#FFFFFF; border:1px solid #E4E7EC; border-radius:10px; padding:.6rem .8rem .25rem; margin-bottom:.75rem; }
        [class*="st-key-compare-chip-"] { border:1px solid #D1FAE5; background:#F0FDFA; border-radius:8px; padding:.35rem .55rem; margin-bottom:.4rem; }
        .st-key-final-metrics [data-testid="stDataFrame"] { min-height:132px; }
        @media (max-width: 900px) {
            .main .block-container { padding-left:1rem; padding-right:1rem; }
            .fx-detail-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .fx-kv { grid-template-columns:130px 1fr; }
            .st-key-create-summary { position:static; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-") or "metric"


def metric_figure(
    frame: pd.DataFrame,
    *,
    show_legend: bool = False,
) -> go.Figure:
    """Build a compact MLflow-like line chart from a chart-ready dataframe."""

    fig = go.Figure()
    if not frame.empty:
        labels = list(dict.fromkeys(frame["run_label"].astype(str)))
        for index, label in enumerate(labels):
            series = frame.loc[frame["run_label"].astype(str) == label].sort_values("x")
            fig.add_trace(
                go.Scatter(
                    x=series["x"],
                    y=series["value"],
                    name=label,
                    mode="lines",
                    line={
                        "width": 2.25,
                        "color": CHART_PALETTE[index % len(CHART_PALETTE)],
                    },
                    hovertemplate="%{y:.6g}<extra>%{fullData.name}</extra>",
                )
            )
    fig.update_layout(
        height=300,
        margin={"l": 12, "r": 12, "t": 8, "b": 48 if show_legend else 24},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={
            "family": "Inter, ui-sans-serif, system-ui, sans-serif",
            "size": 14,
            "color": COLORS["text"],
        },
        hovermode="x unified",
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
        },
    )
    fig.update_xaxes(
        title=None,
        nticks=6,
        tickfont={"size": 13, "color": COLORS["muted"]},
        title_font={"size": 14},
        gridcolor="#EAECF0",
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        title=None,
        nticks=6,
        tickfont={"size": 13, "color": COLORS["muted"]},
        title_font={"size": 14},
        gridcolor="#EAECF0",
        zeroline=False,
        automargin=True,
    )
    return fig


def render_metric_chart_card(
    metric: str, frame: pd.DataFrame, *, compare: bool = False, display_title: str | None = None
) -> None:
    """Render a consistently sized chart card with a concise metric heading."""

    x_axis = str(frame["x_axis"].iloc[0]) if not frame.empty else "Step"
    key = f"metric-card-{_safe_key(metric)}-{'compare' if compare else 'single'}"
    with st.container(key=key):
        st.markdown(
            f"<div class='fx-chart-card'><div class='fx-chart-title'>{display_title or metric}</div>"
            f"<div class='fx-chart-subtitle'>X axis: {x_axis}</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            metric_figure(frame, show_legend=compare),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
            key=f"plot-{key}",
        )


def render_final_metrics_table(rows: list[Mapping[str, Any]], *, key: str) -> None:
    """Render the detailed metrics on demand, keeping the default view compact."""

    with st.container(key=key):
        with st.expander("All metrics", expanded=False):
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                row_height=44,
            )
