<div align="center">

# 📈 AI Trading Signal Analyzer

### AI/ML-Powered Stock Trading Signal Analyzer for Indian NSE/BSE Markets

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://trading-signal-analyzer.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

An intelligent stock trading signal analyzer combining **candlestick pattern recognition**, **technical indicators**, and **real-time news sentiment analysis** to generate **BUY / SELL / HOLD** signals — with **live intraday monitoring** and **backtesting** capabilities.

🌐 **Live Demo:** [trading-signal-analyzer.streamlit.app](https://trading-signal-analyzer.streamlit.app)

---

</div>

## 📸 Screenshots

<table>
  <tr>
    <td align="center"><b>Main Analysis</b></td>
    <td align="center"><b>Signal Output</b></td>
    <td align="center"><b>Intraday Live</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/main-analysis.png" width="300"/></td>
    <td><img src="docs/screenshots/signal-output.png" width="300"/></td>
    <td><img src="docs/screenshots/intraday-live.png" width="300"/></td>
  </tr>
  <tr>
    <td align="center"><b>Candlestick Chart</b></td>
    <td align="center"><b>News Sentiment</b></td>
    <td align="center"><b>Backtesting</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/candlestick-chart.png" width="300"/></td>
    <td><img src="docs/screenshots/news-sentiment.png" width="300"/></td>
    <td><img src="docs/screenshots/backtest-results.png" width="300"/></td>
  </tr>
</table>

---

## ✨ Features

### 📊 Core Analysis Engine
- 🕯️ Detect **21 candlestick patterns** with volume confirmation
- 📉 Evaluate **8 technical indicators** (RSI, MACD, SMA, EMA, Bollinger Bands, etc.)
- 🧠 **News sentiment analysis** using **VADER + FinBERT** NLP models
- 🔀 Combine **rule-based** and **ML-based** decision engines
- 📈 **Backtest** strategies on historical data

### 🔴 Live Intraday Monitoring
- ⚡ Auto-refresh live prices and signals
- 🎯 Supports **8 intraday trading strategies**
- 🕐 Market hours awareness for NSE/BSE
- 📜 Signal history tracking
- 🔗 Quick navigation from main analysis to live dashboard

### 🇮🇳 Indian Market Support
- 📋 Monitor **15+ popular NSE stocks** or custom symbols
- 🔄 Toggle between **NSE / BSE** exchanges
- 📰 Indian news sources integrated for sentiment scoring

---
## 🏗️ Architecture

<pre>
┌──────────────────────────────────────────────────┐
│                    FRONTEND                      │
│           Streamlit + Plotly Dashboards          │
│   Main Analysis  │  Signal Output  │  Intraday   │
└────────────────────────┬─────────────────────────┘
                         │ API Requests
┌────────────────────────▼─────────────────────────┐
│                     BACKEND                      │
│              Python 3.9+ + FastAPI               │
│   Stock Data  │  Candlestick Patterns  │  ML/AI  │
└────────────────────────┬─────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────┐
│                      DATA                        │
│  NSE/BSE Stocks  │  Yahoo Finance API  │  News   │
│       Historical OHLC  │  Sentiment Models       │
└──────────────────────────────────────────────────┘
</pre>

---

## 🧱 Tech Stack

| Layer              | Technology                                  |
| :----------------- | :------------------------------------------ |
| **Frontend**       | Streamlit, Plotly, Matplotlib               |
| **Backend**        | Python 3.9+, FastAPI                        |
| **Data Sources**   | Yahoo Finance API, Google News RSS          |
| **ML / NLP**       | scikit-learn, pandas, numpy, FinBERT, VADER |
| **Visualization**  | Plotly, Matplotlib                          |
| **Deployment**     | Streamlit Cloud / Local                     |

---
## 📁 Project Structure

<pre>
trading-signal-analyzer/
│
├── app/
│   ├── main.py                  # Main Streamlit application
│   ├── intraday.py              # Intraday monitoring dashboard
│   ├── config.py                # Constants & API keys
│   └── utils/
│       ├── data_fetcher.py      # Fetch stock & news data
│       ├── patterns.py          # Candlestick pattern detection
│       ├── sentiment.py         # NLP sentiment analysis
│       └── strategies.py        # Intraday & rule-based engines
│
├── tests/                       # Unit & integration tests
│
├── docs/
│   └── screenshots/             # App screenshots
│
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
</pre>

---
## 🔑 API & Signal Logic

### Data Layer

| Module             | Function                           | Description                                                         |
| :----------------- | :--------------------------------- | :------------------------------------------------------------------ |
| `data_fetcher.py`  | `get_stock_data(symbol, interval)` | Fetches OHLCV data from Yahoo Finance for given symbol & timeframe  |
| `data_fetcher.py`  | `get_news(symbol)`                 | Fetches latest news articles for a stock                            |

### Analysis Layer

| Module             | Function                    | Description                                            |
| :----------------- | :-------------------------- | :----------------------------------------------------- |
| `patterns.py`      | `detect_patterns(df)`       | Detects 21 candlestick patterns in OHLCV data          |
| `patterns.py`      | `pattern_signal(df)`        | Returns BUY/SELL/HOLD signals from detected patterns   |
| `sentiment.py`     | `analyze_sentiment(text)`   | Computes sentiment score using VADER + FinBERT         |

### Strategy Layer

| Module             | Function                               | Description                                                   |
| :----------------- | :------------------------------------- | :------------------------------------------------------------ |
| `strategies.py`    | `intraday_strategy_1(df)`              | One of 8 live intraday strategies                             |
| `strategies.py`    | `rule_based_engine(df, sentiment)`     | Combines candlestick + indicators + sentiment for signals     |

### Application Layer

| Module             | Function                      | Description                                                  |
| :----------------- | :---------------------------- | :----------------------------------------------------------- |
| `main.py`          | `generate_signal(symbol)`     | Master function for main analysis mode                       |
| `intraday.py`      | `monitor_intraday(symbol)`    | Real-time price & signal monitoring during market hours      |

---

## 📊 Intraday Strategies

| #  | Strategy                      | Description                                              |
| :- | :---------------------------- | :------------------------------------------------------- |
| 1  | **Breakout**                  | Detects price breakout from previous high/low            |
| 2  | **Moving Average Crossover**  | Signals BUY/SELL on SMA/EMA crossovers                   |
| 3  | **RSI Oversold/Overbought**   | Uses RSI thresholds to generate signals                  |
| 4  | **MACD Momentum**             | Signals trend continuation based on MACD                 |
| 5  | **Bollinger Bands**           | Signals BUY/SELL when price touches bands                |
| 6  | **Volume Spike**              | Detects abnormal volume patterns for entries             |
| 7  | **News Sentiment**            | Incorporates real-time sentiment into trade signals      |
| 8  | **Combined Engine**           | Aggregates all strategy outputs into a final signal      |

---
## 🕯️ Candlestick Patterns (21 Patterns)

<details>
<summary><b>Click to expand full pattern list</b></summary>
<br>

| #  | Pattern              | Type      | Signal              |
| :- | :------------------- | :-------- | :------------------ |
| 1  | Doji                 | Neutral   | Hold                |
| 2  | Hammer               | Bullish   | Buy                 |
| 3  | Hanging Man          | Bearish   | Sell                |
| 4  | Inverted Hammer      | Bullish   | Buy                 |
| 5  | Shooting Star        | Bearish   | Sell                |
| 6  | Bullish Engulfing    | Bullish   | Buy                 |
| 7  | Bearish Engulfing    | Bearish   | Sell                |
| 8  | Morning Star         | Bullish   | Buy                 |
| 9  | Evening Star         | Bearish   | Sell                |
| 10 | Three White Soldiers | Bullish   | Buy                 |
| 11 | Three Black Crows    | Bearish   | Sell                |
| 12 | Piercing Line        | Bullish   | Buy                 |
| 13 | Dark Cloud Cover     | Bearish   | Sell                |
| 14 | Harami Bullish       | Bullish   | Buy                 |
| 15 | Harami Bearish       | Bearish   | Sell                |
| 16 | Tweezer Bottom       | Bullish   | Buy                 |
| 17 | Tweezer Top          | Bearish   | Sell                |
| 18 | Marubozu             | Depends   | Trend Continuation  |
| 19 | Spinning Top         | Neutral   | Hold                |
| 20 | Long-Legged Doji     | Neutral   | Hold                |
| 21 | Belt Hold            | Depends   | Buy/Sell (trend)    |

</details>

---
## 🚀 Getting Started

### Prerequisites

| Requirement     | Version / Details              |
| :-------------- | :----------------------------- |
| **Python**      | 3.9 – 3.13                    |
| **pip**         | Latest                         |
| **Git**         | Any recent version             |
| **RAM**         | 4GB+ minimum (8GB for FinBERT) |

---

### 1️⃣ Clone the Repository

<pre>
git clone https://github.com/TanmaySharma977/trading-signal-analyzer.git
cd trading-signal-analyzer
</pre>

### 2️⃣ Create Virtual Environment

<pre>
python -m venv venv
</pre>

**Activate:**

<pre>
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
</pre>

### 3️⃣ Install Dependencies

<pre>
pip install --upgrade pip
pip install -r requirements.txt
</pre>

### 4️⃣ Download NLTK Data

<pre>
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
</pre>

### 5️⃣ Configure Environment *(Optional)*

<pre>
cp .env.example .env
</pre>

Edit `.env` and add your API key:

<pre>
NEWS_API_KEY=your_newsapi_key_here
</pre>

### 6️⃣ Run the Application

<pre>
streamlit run app/main.py
</pre>

Open your browser at:

<pre>
http://localhost:8501
</pre>

### 7️⃣ Run Intraday Monitoring

Navigate via sidebar or directly visit:

<pre>
http://localhost:8501/Intraday_Live?stock=RELIANCE&exchange=NSE
</pre>

### 8️⃣ Run Tests

<pre>
python -m unittest discover tests
</pre>

---
## 🗺️ Roadmap

- [x] Candlestick pattern detection (21 patterns)
- [x] Technical indicator analysis
- [x] News sentiment with VADER + FinBERT
- [x] Live intraday monitoring
- [x] Backtesting engine
- [ ] Portfolio tracking & management
- [ ] Alert notifications (Email / Telegram)
- [ ] Options chain analysis
- [ ] Multi-timeframe analysis
- [ ] Mobile-responsive UI

---

## ⚠️ Disclaimer

> **This project is for educational and informational purposes only.**
> It is **NOT** financial advice.
> Always consult a certified financial advisor before making any trading decisions.
> The authors are not responsible for any financial losses incurred using this tool.

---

## 👨‍💻 Author

<div align="center">

**Tanmay Sharma**

[![GitHub](https://img.shields.io/badge/GitHub-TanmaySharma977-181717?style=for-the-badge&logo=github)](https://github.com/TanmaySharma977)
[![Email](https://img.shields.io/badge/Email-tanmaysharma977@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:tanmaysharma977@gmail.com)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com) — Stock data API
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) — Sentiment analysis
- [FinBERT](https://github.com/ProsusAI/finBERT) — Financial NLP model
- [Streamlit](https://streamlit.io) — App framework
- [Plotly](https://plotly.com) — Interactive charts
- [MoneyControl](https://www.moneycontrol.com) — Indian market data reference
- [Economic Times](https://economictimes.indiatimes.com) — News source

---
