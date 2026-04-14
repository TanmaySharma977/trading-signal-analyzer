"""
Trading Signal Analyzer - Main Streamlit Application
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_ingestion.market_data import MarketDataFetcher
from src.data_ingestion.news_data import NewsDataFetcher
from src.preprocessing.market_preprocessor import MarketPreprocessor
from src.preprocessing.news_preprocessor import NewsPreprocessor
from src.analysis.pattern_detector import PatternDetector
from src.analysis.technical_indicators import TechnicalIndicators
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.models.rule_based_engine import RuleBasedEngine
from src.backtesting.backtester import Backtester
from src.utils.constants import POPULAR_STOCKS, Signal


# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Trading Signal Analyzer",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================
# Custom CSS
# =====================
st.markdown("""
<style>
    .signal-buy {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,200,83,0.4);
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255,23,68,0.4);
    }
    .signal-hold {
        background: linear-gradient(135deg, #546e7a, #78909c);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(84,110,122,0.4);
    }
    .metric-card {
        background: #1e1e2f;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00e676;
        margin: 5px 0;
    }
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)


# =====================
# Initialize Session State
# =====================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "results" not in st.session_state:
    st.session_state.results = None


# =====================
# Cached Initializations
# =====================
@st.cache_resource
def load_sentiment_analyzer(model_type="vader"):
    return SentimentAnalyzer(model_type=model_type)


@st.cache_data(ttl=300)
def fetch_market_data(symbol, period, interval, exchange):
    fetcher = MarketDataFetcher(exchange=exchange)
    return fetcher.fetch_historical(symbol=symbol, period=period, interval=interval)


@st.cache_data(ttl=600)
def fetch_news_data(query, source, max_articles, days_back):
    fetcher = NewsDataFetcher()
    return fetcher.fetch_news(
        query=query, source=source,
        max_articles=max_articles, days_back=days_back
    )


# =====================
# Chart Builder
# =====================
def create_candlestick_chart(df, patterns=None, symbol=""):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} Candlestick Chart", "Volume", "RSI"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff1744",
        ),
        row=1, col=1,
    )

    if "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20",
                       line=dict(color="#ffeb3b", width=1)),
            row=1, col=1,
        )

    if "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50",
                       line=dict(color="#2196f3", width=1)),
            row=1, col=1,
        )

    if "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                       line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                       line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                       fill="tonexty", fillcolor="rgba(255,255,255,0.05)"),
            row=1, col=1,
        )

    if patterns:
        bullish_patterns = [p for p in patterns if p["type"] == "Bullish"]
        bearish_patterns = [p for p in patterns if p["type"] == "Bearish"]

        if bullish_patterns:
            dates = [p["date"] for p in bullish_patterns]
            prices = [df.loc[d, "low"] * 0.995 if d in df.index else None for d in dates]
            texts = [p["pattern"] for p in bullish_patterns]
            valid = [(d, p, t) for d, p, t in zip(dates, prices, texts) if p is not None]
            if valid:
                fig.add_trace(
                    go.Scatter(
                        x=[v[0] for v in valid], y=[v[1] for v in valid],
                        mode="markers+text", name="Bullish Pattern",
                        marker=dict(symbol="triangle-up", size=14, color="#00e676",
                                    line=dict(width=1, color="white")),
                        text=[v[2] for v in valid], textposition="bottom center",
                        textfont=dict(size=9, color="#00e676"),
                    ),
                    row=1, col=1,
                )

        if bearish_patterns:
            dates = [p["date"] for p in bearish_patterns]
            prices = [df.loc[d, "high"] * 1.005 if d in df.index else None for d in dates]
            texts = [p["pattern"] for p in bearish_patterns]
            valid = [(d, p, t) for d, p, t in zip(dates, prices, texts) if p is not None]
            if valid:
                fig.add_trace(
                    go.Scatter(
                        x=[v[0] for v in valid], y=[v[1] for v in valid],
                        mode="markers+text", name="Bearish Pattern",
                        marker=dict(symbol="triangle-down", size=14, color="#ff1744",
                                    line=dict(width=1, color="white")),
                        text=[v[2] for v in valid], textposition="top center",
                        textfont=dict(size=9, color="#ff1744"),
                    ),
                    row=1, col=1,
                )

    colors = [
        "#00e676" if row["close"] >= row["open"] else "#ff1744"
        for _, row in df.iterrows()
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1,
    )

    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                       line=dict(color="#ab47bc", width=1.5)),
            row=3, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=800, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.update_yaxes(title_text="Price (Rs.)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    return fig


def create_portfolio_chart(portfolio_df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_df["date"], y=portfolio_df["value"],
            mode="lines", name="Portfolio Value",
            line=dict(color="#00e676", width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.1)",
        )
    )
    fig.update_layout(
        template="plotly_dark", title="Portfolio Value Over Time",
        yaxis_title="Value (Rs.)", height=400,
    )
    return fig


# =====================
# SIDEBAR
# =====================
st.sidebar.title("Trading Signal Analyzer")
st.sidebar.markdown("---")

# Stock Selection
st.sidebar.subheader("Stock Selection")
stock_input_method = st.sidebar.radio(
    "Choose stock:", ["Popular Stocks", "Custom Symbol"]
)

if stock_input_method == "Popular Stocks":
    selected_stock = st.sidebar.selectbox(
        "Select Stock:",
        options=list(POPULAR_STOCKS.keys()),
        format_func=lambda x: f"{x} - {POPULAR_STOCKS[x]}",
        index=0,
    )
else:
    selected_stock = st.sidebar.text_input(
        "Enter NSE Symbol:", value="RELIANCE"
    ).upper().strip()

exchange = st.sidebar.selectbox("Exchange:", ["NSE", "BSE"], index=0)

st.sidebar.markdown("---")

# Timeframe
st.sidebar.subheader("Timeframe")

interval = st.sidebar.selectbox(
    "Candle Interval:",
    ["1d", "1h", "5m", "15m", "1wk"],
    index=0,
    help="Each candle represents this timeframe",
)

VALID_PERIODS = {
    "1m":  ["1d", "5d"],
    "5m":  ["1d", "5d", "1mo"],
    "15m": ["1d", "5d", "1mo"],
    "1h":  ["1d", "5d", "1mo", "3mo"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    "1wk": ["3mo", "6mo", "1y", "2y", "5y"],
    "1mo": ["6mo", "1y", "2y", "5y"],
}

allowed_periods = VALID_PERIODS.get(interval, ["1mo", "3mo", "6mo", "1y"])

period = st.sidebar.selectbox(
    "Data Period:",
    allowed_periods,
    index=min(1, len(allowed_periods) - 1),
    help="How far back to fetch data",
)

if interval in ["1m", "5m", "15m"]:
    st.sidebar.warning(f"{interval} candles are limited to recent data only.")

st.sidebar.markdown("---")

# Analysis Settings
st.sidebar.subheader("Settings")
include_news = st.sidebar.toggle("Include News Sentiment", value=True)
include_technical = st.sidebar.toggle("Include Technical Indicators", value=True)

sentiment_model = st.sidebar.selectbox(
    "Sentiment Model:", ["vader", "finbert"], index=0,
    help="VADER=Fast, FinBERT=Accurate (requires more RAM)",
)

max_news = st.sidebar.slider("Max News Articles:", 3, 20, 10)

st.sidebar.markdown("---")

# Mode
st.sidebar.subheader("Mode")
mode = st.sidebar.radio("Analysis Mode:", ["Live Analysis", "Backtesting"])

st.sidebar.markdown("---")

# Intraday Link
st.sidebar.subheader("Live Intraday")
intraday_sidebar_url = f"/Intraday_Live?stock={selected_stock}&exchange={exchange}"
st.sidebar.markdown(
    f"""
    <a href="{intraday_sidebar_url}" target="_blank" style="
        display: block;
        text-align: center;
        background: #ff1744;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        font-size: 14px;
    ">
        LIVE Intraday: {selected_stock}
    </a>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# Single Analyze Button
analyze_button = st.sidebar.button(
    "ANALYZE NOW",
    use_container_width=True,
    type="primary",
)


# =====================
# MAIN CONTENT
# =====================
st.title("AI Trading Signal Analyzer")
st.markdown(
    f"**Analyzing:** `{selected_stock}` on `{exchange}` | "
    f"**Interval:** `{interval}` | **Period:** `{period}`"
)
st.markdown("---")


# =====================
# ANALYSIS PIPELINE
# =====================
if analyze_button:
    progress = st.progress(0, text="Starting analysis...")

    try:
        # STEP 1: Fetch Market Data
        progress.progress(10, text="Fetching market data...")
        market_df = fetch_market_data(selected_stock, period, interval, exchange)

        if market_df.empty:
            st.error(
                f"No data found for {selected_stock}. "
                "Check: Is the symbol correct? Is the market open?"
            )
            st.stop()

        st.success(f"Fetched {len(market_df)} candles for {selected_stock}")

        # STEP 2: Preprocess
        progress.progress(20, text="Cleaning market data...")
        preprocessor = MarketPreprocessor()
        market_df = preprocessor.clean(market_df)
        market_df = preprocessor.add_basic_features(market_df)

        # STEP 3: Technical Indicators
        progress.progress(30, text="Calculating technical indicators...")
        market_df = TechnicalIndicators.add_all(market_df)
        technical_score = TechnicalIndicators.get_trend_signal(market_df) if include_technical else 0.0

        # STEP 4: Pattern Detection
        progress.progress(45, text="Detecting candlestick patterns...")
        detector = PatternDetector(market_df)
        all_patterns = detector.detect_all()
        recent_patterns = detector.get_latest_patterns(n=5)
        pattern_score = detector.get_signal_score()

        # STEP 5: News Sentiment
        sentiment_result = None
        sentiment_score = 0.0

        if include_news:
            progress.progress(60, text="Fetching news articles...")

            stock_name = POPULAR_STOCKS.get(selected_stock, selected_stock)
            raw_news = fetch_news_data(stock_name, "auto", max_news, 7)

            if not raw_news:
                raw_news = fetch_news_data(selected_stock, "auto", max_news, 7)

            if not raw_news:
                raw_news = fetch_news_data("Indian stock market", "auto", max_news, 7)
                if raw_news:
                    st.info("Using general market news (no stock-specific news found)")

            progress.progress(70, text="Analyzing sentiment...")

            if raw_news:
                news_preprocessor = NewsPreprocessor()
                cleaned_news = news_preprocessor.clean_articles(raw_news)
                analyzer = load_sentiment_analyzer(sentiment_model)
                sentiment_result = analyzer.analyze_articles(cleaned_news)
                sentiment_score = sentiment_result["overall_score"]
            else:
                st.warning("No news articles found. Proceeding without sentiment.")
        else:
            progress.progress(70, text="Skipping news sentiment...")

        # STEP 6: Generate Signal
        progress.progress(85, text="Generating trading signal...")
        engine = RuleBasedEngine()
        signal_result = engine.generate_signal(
            pattern_score=pattern_score,
            sentiment_score=sentiment_score,
            technical_score=technical_score,
            include_news=include_news,
            include_technical=include_technical,
        )

        # STEP 7: Backtesting
        backtest_result = None
        if mode == "Backtesting":
            progress.progress(92, text="Running backtest...")
            bt_signals = []
            for i in range(5, len(market_df)):
                chunk = market_df.iloc[: i + 1]
                det = PatternDetector(chunk)
                det.detect_all()
                p_score = det.get_signal_score()

                if p_score > 0.3:
                    sig = "BUY"
                elif p_score < -0.3:
                    sig = "SELL"
                else:
                    sig = "HOLD"
                bt_signals.append({"date": market_df.index[i], "signal": sig})

            backtester = Backtester(initial_capital=100000)
            backtest_result = backtester.run(market_df, bt_signals)

        progress.progress(100, text="Analysis complete!")

        st.session_state.analysis_done = True
        st.session_state.results = {
            "market_df": market_df,
            "all_patterns": all_patterns,
            "recent_patterns": recent_patterns,
            "pattern_score": pattern_score,
            "sentiment_result": sentiment_result,
            "sentiment_score": sentiment_score,
            "technical_score": technical_score,
            "signal_result": signal_result,
            "backtest_result": backtest_result,
            "stock": selected_stock,
        }

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# =====================
# DISPLAY RESULTS
# =====================
if st.session_state.analysis_done and st.session_state.results:
    r = st.session_state.results
    signal_result = r["signal_result"]
    market_df = r["market_df"]

    # ---- Signal Banner ----
    signal = signal_result["signal"]
    confidence = signal_result["confidence"]
    reason = signal_result["reason"]

    if signal == "BUY":
        css_class = "signal-buy"
        emoji = "BUY"
    elif signal == "SELL":
        css_class = "signal-sell"
        emoji = "SELL"
    else:
        css_class = "signal-hold"
        emoji = "HOLD"

    # FIX: This is OUTSIDE the if/elif/else now
    st.markdown(
        f'<div class="{css_class}">'
        f'{emoji} -- Confidence: {confidence:.0f}%<br>'
        f'<span style="font-size:16px;">{reason}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ---- Intraday Link ----
    intraday_url = f"/Intraday_Live?stock={r['stock']}&exchange={exchange}&interval=5m"

    link_col1, link_col2, link_col3 = st.columns([1, 2, 1])
    with link_col2:
        st.markdown(
            f"""
            <a href="{intraday_url}" target="_blank" style="
                display: block;
                text-align: center;
                background: linear-gradient(135deg, #ff6d00, #ff9100);
                color: white;
                padding: 14px 28px;
                border-radius: 12px;
                text-decoration: none;
                font-size: 18px;
                font-weight: bold;
                box-shadow: 0 4px 15px rgba(255,109,0,0.4);
                transition: transform 0.2s;
            " onmouseover="this.style.transform='scale(1.05)'"
               onmouseout="this.style.transform='scale(1)'">
                LIVE Intraday Monitor for {r['stock']} ->
            </a>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Opens in a new tab with real-time intraday strategies")

    st.markdown("")

    # ---- Score Breakdown ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Pattern Score",
            f"{r['pattern_score']:+.2f}",
            delta="Bullish" if r["pattern_score"] > 0 else "Bearish" if r["pattern_score"] < 0 else "Neutral",
        )
    with col2:
        st.metric(
            "Sentiment Score",
            f"{r['sentiment_score']:+.2f}",
            delta="Positive" if r["sentiment_score"] > 0 else "Negative" if r["sentiment_score"] < 0 else "Neutral",
        )
    with col3:
        st.metric(
            "Technical Score",
            f"{r['technical_score']:+.2f}",
            delta="Bullish" if r["technical_score"] > 0 else "Bearish" if r["technical_score"] < 0 else "Neutral",
        )
    with col4:
        st.metric(
            "Composite Score",
            f"{signal_result['composite_score']:+.4f}",
        )

    st.markdown("---")

    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Chart", "Patterns", "News Sentiment", "Technical", "Backtest",
    ])

    # TAB 1: Chart
    with tab1:
        st.subheader(f"{r['stock']} Candlestick Chart")
        fig = create_candlestick_chart(market_df, patterns=r["all_patterns"], symbol=r["stock"])
        st.plotly_chart(fig, use_container_width=True)

        last = market_df.iloc[-1]
        prev = market_df.iloc[-2] if len(market_df) > 1 else last
        change_pct = ((last["close"] - prev["close"]) / prev["close"]) * 100

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("Close", f"Rs.{last['close']:.2f}", f"{change_pct:+.2f}%")
        with pcol2:
            st.metric("High", f"Rs.{last['high']:.2f}")
        with pcol3:
            st.metric("Low", f"Rs.{last['low']:.2f}")
        with pcol4:
            st.metric("Volume", f"{last['volume']:,.0f}")

    # TAB 2: Patterns
    with tab2:
        st.subheader("Detected Candlestick Patterns")

        recent = r["recent_patterns"]
        if recent:
            for p in recent:
                ptype = p["type"]
                if ptype == "Bullish":
                    icon = "[+]"
                elif ptype == "Bearish":
                    icon = "[-]"
                else:
                    icon = "[=]"

                with st.expander(
                    f"{icon} **{p['pattern']}** -- {p['type']} "
                    f"(Strength: {p['strength']:.0%})",
                    expanded=True,
                ):
                    st.write(f"**Date:** {p['date']}")
                    st.write(f"**Description:** {p['description']}")
                    st.progress(p["strength"])
        else:
            st.info("No candlestick patterns detected in the last 5 candles.")

        if r["all_patterns"]:
            st.markdown("---")
            st.subheader("All Patterns Summary")
            pattern_df = pd.DataFrame(r["all_patterns"])
            pattern_counts = pattern_df["pattern"].value_counts()
            st.bar_chart(pattern_counts)
            st.dataframe(
                pattern_df[["pattern", "type", "date", "strength"]].tail(20),
                use_container_width=True,
            )

    # TAB 3: News Sentiment
    with tab3:
        st.subheader("News Sentiment Analysis")

        if r["sentiment_result"]:
            sr = r["sentiment_result"]

            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.metric("Overall Sentiment", sr["overall_sentiment"])
            with scol2:
                st.metric("Score", f"{sr['overall_score']:+.4f}")
            with scol3:
                st.metric("Confidence", f"{sr['overall_confidence']:.2%}")

            st.markdown("---")
            st.subheader("Individual Articles")

            for article in sr.get("article_sentiments", []):
                sent = article["sentiment"]
                if sent == "Positive":
                    icon = "[+]"
                elif sent == "Negative":
                    icon = "[-]"
                else:
                    icon = "[=]"

                with st.expander(
                    f"{icon} {article['title'][:80]}... "
                    f"(Score: {article['score']:+.3f})"
                ):
                    st.write(f"**Sentiment:** {article['sentiment']}")
                    st.write(f"**Score:** {article['score']:+.4f}")
                    st.write(f"**Confidence:** {article['confidence']:.2%}")
        else:
            st.info("News sentiment was not included in this analysis.")

    # TAB 4: Technical Indicators
    with tab4:
        st.subheader("Technical Indicators")

        last_row = market_df.iloc[-1]

        tcol1, tcol2, tcol3 = st.columns(3)

        with tcol1:
            st.markdown("### RSI")
            rsi_val = last_row.get("rsi", None)
            if rsi_val and not pd.isna(rsi_val):
                st.metric("RSI (14)", f"{rsi_val:.1f}")
                if rsi_val > 70:
                    st.warning("Overbought Zone")
                elif rsi_val < 30:
                    st.success("Oversold Zone")
                else:
                    st.info("Neutral Zone")
            else:
                st.write("Not enough data for RSI")

        with tcol2:
            st.markdown("### MACD")
            macd_val = last_row.get("macd", None)
            macd_sig = last_row.get("macd_signal", None)
            if macd_val and not pd.isna(macd_val):
                st.metric("MACD", f"{macd_val:.4f}")
                st.metric("Signal", f"{macd_sig:.4f}")
                hist = last_row.get("macd_histogram", 0)
                if hist > 0:
                    st.success("Bullish MACD")
                else:
                    st.error("Bearish MACD")
            else:
                st.write("Not enough data for MACD")

        with tcol3:
            st.markdown("### Moving Averages")
            sma10 = last_row.get("sma_10", None)
            sma20 = last_row.get("sma_20", None)
            sma50 = last_row.get("sma_50", None)

            if sma10 and not pd.isna(sma10):
                st.metric("SMA 10", f"Rs.{sma10:.2f}")
            if sma20 and not pd.isna(sma20):
                st.metric("SMA 20", f"Rs.{sma20:.2f}")
            if sma50 and not pd.isna(sma50):
                st.metric("SMA 50", f"Rs.{sma50:.2f}")

            if sma10 and sma20 and not pd.isna(sma10) and not pd.isna(sma20):
                if sma10 > sma20:
                    st.success("SMA 10 > SMA 20 (Bullish)")
                else:
                    st.error("SMA 10 < SMA 20 (Bearish)")

        st.markdown("---")
        st.subheader("Raw Indicator Data (Last 10 Rows)")
        indicator_cols = [
            col for col in market_df.columns
            if col in [
                "close", "volume", "rsi", "macd", "macd_signal",
                "macd_histogram", "sma_10", "sma_20", "sma_50",
                "bb_upper", "bb_lower", "atr",
            ]
        ]
        st.dataframe(
            market_df[indicator_cols].tail(10).round(4),
            use_container_width=True,
        )

    # TAB 5: Backtesting
    with tab5:
        st.subheader("Backtesting Results")

        if r["backtest_result"]:
            bt = r["backtest_result"]

            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            with bcol1:
                color = "normal" if bt["total_return_pct"] >= 0 else "inverse"
                st.metric(
                    "Total Return",
                    f"{bt['total_return_pct']:+.2f}%",
                    delta=f"Rs.{bt['final_value'] - bt['initial_capital']:+,.0f}",
                    delta_color=color,
                )
            with bcol2:
                st.metric("Win Rate", f"{bt['win_rate']:.0f}%")
            with bcol3:
                st.metric("Total Trades", bt["total_trades"])
            with bcol4:
                st.metric("Max Drawdown", f"{bt['max_drawdown_pct']:.1f}%")

            if not bt["portfolio_values"].empty:
                fig_bt = create_portfolio_chart(bt["portfolio_values"])
                st.plotly_chart(fig_bt, use_container_width=True)

            if bt["trades"]:
                st.markdown("### Trade Log")
                trades_df = pd.DataFrame(bt["trades"])
                st.dataframe(trades_df, use_container_width=True)
        else:
            st.info(
                "Backtesting not run. Select 'Backtesting' mode in the sidebar "
                "and click Analyze."
            )


# =====================
# FOOTER
# =====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <b>AI Trading Signal Analyzer</b> |
        Built with Streamlit, yfinance, FinBERT/VADER |
        <i>For educational purposes only -- not financial advice.</i>
    </div>
    """,
    unsafe_allow_html=True,
)