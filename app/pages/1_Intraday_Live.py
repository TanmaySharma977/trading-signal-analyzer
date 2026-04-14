"""
Intraday Live Monitoring Dashboard
Reads stock from URL query params: ?stock=RELIANCE&exchange=NSE
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
import time as time_module
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_ingestion.realtime_data import RealtimeDataFetcher
from src.data_ingestion.news_data import NewsDataFetcher
from src.preprocessing.market_preprocessor import MarketPreprocessor
from src.preprocessing.news_preprocessor import NewsPreprocessor
from src.analysis.intraday_analyzer import IntradayAnalyzer
from src.analysis.pattern_detector import PatternDetector
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.utils.constants import POPULAR_STOCKS

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Live Intraday Monitor",
    page_icon="🔴",
    layout="wide",
)

# =====================
# Read Query Params (from main page link)
# =====================
query_params = st.query_params

# Get stock from URL or default
default_stock = query_params.get("stock", "RELIANCE")
default_exchange = query_params.get("exchange", "NSE")
default_interval = query_params.get("interval", "5m")

# =====================
# Custom CSS
# =====================
st.markdown("""
<style>
    .live-badge {
        display: inline-block;
        background: #ff1744;
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .signal-buy-intra {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white; padding: 15px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,200,83,0.4);
    }
    .signal-sell-intra {
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: white; padding: 15px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,23,68,0.4);
    }
    .signal-hold-intra {
        background: linear-gradient(135deg, #546e7a, #78909c);
        color: white; padding: 15px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .price-up { color: #00e676; font-size: 32px; font-weight: bold; }
    .price-down { color: #ff1744; font-size: 32px; font-weight: bold; }
    .strategy-bullish {
        border-left: 4px solid #00e676; padding: 8px 12px;
        margin: 4px 0; background: rgba(0,230,118,0.1); border-radius: 4px;
    }
    .strategy-bearish {
        border-left: 4px solid #ff1744; padding: 8px 12px;
        margin: 4px 0; background: rgba(255,23,68,0.1); border-radius: 4px;
    }
    .strategy-neutral {
        border-left: 4px solid #78909c; padding: 8px 12px;
        margin: 4px 0; background: rgba(120,144,156,0.1); border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# =====================
# Session State
# =====================
if "intraday_running" not in st.session_state:
    st.session_state.intraday_running = False
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []


# =====================
# Sidebar
# =====================
st.sidebar.title("LIVE Intraday Monitor")
st.sidebar.markdown('<span class="live-badge">LIVE</span>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Pre-fill from query params
all_stocks = list(POPULAR_STOCKS.keys())
default_idx = all_stocks.index(default_stock) if default_stock in all_stocks else 0

stock_method = st.sidebar.radio("Stock:", ["Popular", "Custom"])
if stock_method == "Popular":
    selected_stock = st.sidebar.selectbox(
        "Select:", all_stocks,
        format_func=lambda x: f"{x} - {POPULAR_STOCKS[x]}",
        index=default_idx,
    )
else:
    selected_stock = st.sidebar.text_input("Symbol:", default_stock).upper().strip()

exchange_options = ["NSE", "BSE"]
exchange_idx = exchange_options.index(default_exchange) if default_exchange in exchange_options else 0
exchange = st.sidebar.selectbox("Exchange:", exchange_options, index=exchange_idx)

st.sidebar.markdown("---")

interval_options = ["5m", "1m", "15m"]
interval_idx = interval_options.index(default_interval) if default_interval in interval_options else 0
interval = st.sidebar.selectbox("Candle Interval:", interval_options, index=interval_idx)

auto_refresh = st.sidebar.toggle("Auto-Refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh every (sec):", 30, 300, 60)
include_news = st.sidebar.toggle("Include News", value=True)

st.sidebar.markdown("---")

# Back to main page link
main_url = f"/?stock={selected_stock}&exchange={exchange}"
st.sidebar.markdown(
    f'<a href="{main_url}" style="color: #00e676; text-decoration: none;">'
    f'← Back to Main Analysis</a>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
manual_refresh = st.sidebar.button("Refresh Now", use_container_width=True, type="primary")


# =====================
# Chart Builder
# =====================
def create_intraday_chart(df, strategies=None, symbol=""):
    """Build intraday chart with VWAP, EMA, pivot lines."""

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} Intraday", "Volume", "RSI (7)"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff1744",
        ),
        row=1, col=1,
    )

    # VWAP
    if "vwap" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["vwap"], name="VWAP",
                line=dict(color="#ffeb3b", width=2, dash="dash"),
            ),
            row=1, col=1,
        )

    # EMA lines
    if "ema_9" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["ema_9"], name="EMA 9",
                line=dict(color="#2196f3", width=1.5),
            ),
            row=1, col=1,
        )
    if "ema_21" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["ema_21"], name="EMA 21",
                line=dict(color="#ff9800", width=1.5),
            ),
            row=1, col=1,
        )

    # Pivot lines
    if strategies and "pivot" in strategies:
        pivot_data = strategies["pivot"]
        for level, color, dash in [
            ("r2", "red", "dot"), ("r1", "red", "dash"),
            ("pivot", "white", "solid"),
            ("s1", "green", "dash"), ("s2", "green", "dot"),
        ]:
            if level in pivot_data:
                fig.add_hline(
                    y=pivot_data[level], line_dash=dash,
                    line_color=color, opacity=0.5,
                    annotation_text=f"{level.upper()}: {pivot_data[level]}",
                    row=1, col=1,
                )

    # Volume
    colors = [
        "#00e676" if r["close"] >= r["open"] else "#ff1744"
        for _, r in df.iterrows()
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1,
    )

    # RSI
    if "rsi_7" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["rsi_7"], name="RSI (7)",
                line=dict(color="#ab47bc", width=1.5),
            ),
            row=3, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=60, b=30),
    )

    return fig


# =====================
# Main Analysis
# =====================
def run_intraday_analysis():
    """Run complete intraday analysis."""

    fetcher = RealtimeDataFetcher(exchange)
    market_status = fetcher.is_market_open()

    # Header
    col_title, col_status = st.columns([3, 1])
    with col_title:
        stock_name = POPULAR_STOCKS.get(selected_stock, selected_stock)
        st.title(f"Intraday: {selected_stock} ({stock_name})")
    with col_status:
        if market_status["is_open"]:
            st.markdown(
                '<span class="live-badge">LIVE</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"{market_status.get('minutes_remaining', '?')} min left")
        else:
            st.warning(market_status["reason"])

    # Fetch data
    with st.spinner("Fetching live data..."):
        df = fetcher.fetch_intraday(selected_stock, interval=interval, period="5d")
        prev_df = fetcher.fetch_intraday(selected_stock, interval="1d", period="5d")
        snapshot = fetcher.get_live_snapshot(selected_stock)

    if df.empty:
        st.error(f"No intraday data for {selected_stock}.")
        st.info("Tip: Intraday data is available during/after market hours on trading days.")
        return

    df = MarketPreprocessor.clean(df)
    df = MarketPreprocessor.add_basic_features(df)

    # ---- Price Banner ----
    if snapshot:
        price = snapshot["price"]
        change = snapshot["change"]
        change_pct = snapshot["change_pct"]

        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        with pcol1:
            css = "price-up" if change >= 0 else "price-down"
            arrow = "^" if change >= 0 else "v"
            st.markdown(f'<div class="{css}">{arrow} Rs.{price:.2f}</div>', unsafe_allow_html=True)
        with pcol2:
            st.metric("Change", f"Rs.{change:+.2f}", f"{change_pct:+.2f}%")
        with pcol3:
            st.metric("Open", f"Rs.{snapshot['open']:.2f}")
        with pcol4:
            st.metric("High", f"Rs.{snapshot['high']:.2f}")
        with pcol5:
            st.metric("Low", f"Rs.{snapshot['low']:.2f}")

    st.markdown("---")

    # ---- Run Strategies ----
    analyzer = IntradayAnalyzer(df, prev_df)
    intraday_result = analyzer.run_all_strategies()

    detector = PatternDetector(df)
    detector.detect_all()
    pattern_score = detector.get_signal_score()

    sentiment_score = 0.0
    sentiment_result = None
    if include_news:
        try:
            stock_name = POPULAR_STOCKS.get(selected_stock, selected_stock)
            nf = NewsDataFetcher()
            raw_news = nf.fetch_news(stock_name, "auto", 5, 3)
            if raw_news:
                cleaned = NewsPreprocessor.clean_articles(raw_news)
                sa = SentimentAnalyzer("vader")
                sentiment_result = sa.analyze_articles(cleaned)
                sentiment_score = sentiment_result["overall_score"]
        except Exception:
            pass

    final_score = (
        intraday_result["score"] * 0.50
        + pattern_score * 0.25
        + sentiment_score * 0.25
    )

    if final_score > 0.12:
        final_signal = "BUY"
    elif final_score < -0.12:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    final_confidence = min(abs(final_score) * 150, 95)

    # ---- Signal Display ----
    sig_col, detail_col = st.columns([1, 2])

    with sig_col:
        css_map = {
            "BUY": "signal-buy-intra",
            "SELL": "signal-sell-intra",
            "HOLD": "signal-hold-intra",
        }
        st.markdown(
            f'<div class="{css_map[final_signal]}">'
            f'{final_signal}<br>{final_confidence:.0f}% confidence</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Score: {final_score:+.4f} | "
            f"Intraday: {intraday_result['score']:+.4f} | "
            f"Pattern: {pattern_score:+.4f} | "
            f"News: {sentiment_score:+.4f}"
        )

    with detail_col:
        st.markdown(f"**Reason:** {intraday_result['reason']}")
        st.markdown(f"**Strategies Active:** {intraday_result['active_strategies']}/8")

        strategies = intraday_result.get("strategies", {})
        for name, data in strategies.items():
            score = data["score"]
            if score > 0.1:
                css = "strategy-bullish"
            elif score < -0.1:
                css = "strategy-bearish"
            else:
                css = "strategy-neutral"

            st.markdown(
                f'<div class="{css}">'
                f'<b>{data.get("name", name)}</b>: {data["description"]} '
                f'(score: {score:+.2f})</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---- Chart ----
    st.subheader("Intraday Chart")
    fig = create_intraday_chart(analyzer.df, strategies=strategies, symbol=selected_stock)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Signal History ----
    st.session_state.signal_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "signal": final_signal,
        "confidence": round(final_confidence, 1),
        "score": round(final_score, 4),
        "price": snapshot.get("price", 0) if snapshot else 0,
    })
    st.session_state.signal_history = st.session_state.signal_history[-50:]

    if len(st.session_state.signal_history) > 1:
        st.subheader("Signal History")
        hist_df = pd.DataFrame(st.session_state.signal_history)
        st.dataframe(hist_df.tail(20), use_container_width=True)

    # ---- News ----
    if sentiment_result and include_news:
        st.markdown("---")
        st.subheader(
            f"News: {sentiment_result['overall_sentiment']} "
            f"({sentiment_result['overall_score']:+.4f})"
        )
        for art in sentiment_result.get("article_sentiments", [])[:5]:
            icon = "+" if art["sentiment"] == "Positive" else "-" if art["sentiment"] == "Negative" else "="
            st.caption(f"{icon} [{art['sentiment']}] {art['title']}")

    # Footer
    st.markdown("---")
    st.session_state.refresh_count += 1
    st.caption(
        f"Updated: {datetime.now().strftime('%H:%M:%S')} | "
        f"Refresh #{st.session_state.refresh_count} | "
        f"Candles: {len(df)} | Interval: {interval}"
    )


# =====================
# Run
# =====================
run_intraday_analysis()

if auto_refresh:
    time_module.sleep(refresh_seconds)
    st.rerun()

if manual_refresh:
    st.rerun()