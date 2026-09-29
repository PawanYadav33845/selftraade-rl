# ⚡ SelfTrade-RL | Autonomous Multi-Category Trading & Continuous Self-Learning System

`SelfTrade-RL` is a discrete GPU-accelerated, multi-asset autonomous trading platform powered by Reinforcement Learning (PPO Agent), Cognitive Trade Reasoning, an 8-Strategy Ensemble, and real-time MetaTrader 5 (MT5) execution integration.

---

## 🌟 Key Features & Architecture

### 🧠 1. Cognitive Multi-Factor Trade Reasoning Engine
- **Strict Confluence Gating ($\ge 0.65$)**: Filters out bad entries by evaluating market sessions, trend memory, pattern signals, and strategy ensemble alignment before approving trade orders.
- **Adaptive Holding Strategy**: Automatically toggles between **Quick Scalps (1:1.5 - 1:2.0 R:R)** for ranging markets and **Big Trend Runners (1:3.0+ R:R)** during strong breakout momentum.
- **Dynamic Dual Target Geometry**: Sets primary and secondary Take-Profit targets ($\text{TP1}$ scalp / $\text{TP2}$ trend), Break-Even protection shifts ($+1.0\text{R}$ profit trigger), and ATR-based dynamic trailing stops.

### 🛡️ 2. Multi-Category Concurrency & Position Scaling
- **Multi-Category Asset Support**: Trades **Precious Metals / Commodities** (Gold `XAUUSD`, Silver `XAGUSD`) and **Crypto** (`BTCUSD`).
- **Category Position Scaling**: Supports opening **at least 2 concurrent trade positions per asset category / symbol** simultaneously, subject to portfolio risk limits.
- **Deterministic Risk Gating**: Enforces hard stop-loss limits ($2.0\%$), $5.0\%$ daily circuit breakers, margin availability checks, and maximum portfolio position caps ($10$ positions max).

### 📊 3. 8-Strategy Ensemble Synthesis
Evaluates **8 famous technical strategies** simultaneously with category-optimized weighting:
1. **EMA Trend Crossover (20 / 50 / 200)**
2. **RSI Overbought / Oversold (14)**
3. **MACD Momentum**
4. **Bollinger Bands Volatility Reversion**
5. **Support & Resistance Pivot Rejection**
6. **Price Action Candlestick Patterns (Engulfing, Hammer, Shooting Star)**
7. **Stochastic Oscillator (%K/%D)**
8. **Keltner Volatility Channel Breakout**

### 🖥️ 4. Resizable Pro-Trader Desktop Control Center
Overhauled CustomTkinter desktop app featuring a **Pro-Trader Tabbed Workstation**:
- **Executive Overview & Analytics**: Real-time equity curves, Net Worth metrics, and Win/Loss pie chart infographics.
- **2x2 Multi-Asset Live Chart Grid**: Dark-theme Matplotlib subplots displaying live candles, Support/Resistance levels, and entry markers for Gold, Silver, BTC, and Portfolio Equity.
- **Active Orders & Category Positions**: Interactive position table rendering all active concurrent trades per category.
- **Strategy Diagnostics Engine**: Real-time 8-strategy vote breakdown bar chart.
- **Trade Journal & Session Reports**: Comprehensive execution history with CSV export and integrated report viewer.

### 🔌 5. Auto-Healing MetaTrader 5 (MT5) Integration
- **Auto-Reconnecting Broker Adapter**: Automatically tests and restores connection if MT5 terminal drops.
- **Simulated Paper Fallback**: Built-in market data generator for risk-free paper trading simulations.

### 📝 6. Automatic Session Learning Report Generator
Auto-generates structured **Markdown (`.md`)** and **Plain Text (`.txt`)** performance reports upon session closure (`reports/session_learning_report_<TIMESTAMP>.txt` & `.md`), detailing:
- Starting & Final Capital, Net PnL ($ and %)
- Win Rate %, Profit Factor, Max Drawdown
- Per-asset breakdown (Trades, Wins, Losses, PnL)
- PPO continuous self-learning model evolution checkpoints

---

## 📂 Project Directory Structure

```text
SelfTrade-RL/
├── selftrade/
│   ├── agent/                # PyTorch PPO Agent & Replay Buffers
│   ├── analysis/             # Pattern Engine, Candlesticks & Trend Memory
│   ├── continuous_learning/  # Online Self-Improver & Performance Monitor
│   ├── data/                 # Data Ingestion & Market Data Generators
│   ├── env/                  # Gymnasium RL Trading Environment
│   ├── gui/                  # CustomTkinter Desktop GUI Workstation
│   ├── journal/              # Persistent Trade Diary & CSV Exporter
│   ├── live/                 # MT5 Adapter & MultiSymbolLiveManager
│   ├── reasoning/            # Trade Reasoning Engine & Confluence Evaluator
│   ├── reports/              # Session Learning Report Generator (.md & .txt)
│   ├── risk/                 # Deterministic Risk & Circuit Breaker Engine
│   ├── sessions/             # Market Session Volatility Bias Engine
│   └── strategies/           # 8-Strategy Ensemble & Backtester
├── tests/                    # 45 Automated Unit & Integration Tests
├── journal/                  # Saved Trade Journal Data
├── reports/                  # Generated Session Learning Reports (.md & .txt)
├── .gitignore                # Git Ignore Rules
└── README.md                 # Project Documentation
```

---

## 🚀 Quick Start Guide

### 1. Requirements & Installation

- Python 3.10+
- (Optional) MetaTrader 5 Terminal installed on Windows for live demo execution.

Clone repository and activate virtual environment:
```powershell
cd SelfTrade-RL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```powershell
pip install torch pandas numpy matplotlib customtkinter MetaTrader5 pytest
```

### 2. Launch the Desktop Control Center

Run the main GUI application:
```powershell
python -m selftrade.gui.app
```

### 3. Run Automated Tests

Execute the 45 unit and integration test suite:
```powershell
python -m pytest tests/ -v
```

---

## 📜 License

This project is open-source under the MIT License.
