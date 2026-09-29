"""
Unit Tests for Learning Report Generator, Trade Confluence Filter, MT5 Connection Recovery,
and Multi-Symbol Resource Optimization.
"""

from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import os
import pytest
import pandas as pd
import numpy as np

from selftrade.reports.learning_report import SessionLearningReportGenerator
from selftrade.reasoning.trade_reasoner import TradeReasoner, ReasonedTradeDecision
from selftrade.live.mt5_adapter import MT5BrokerAdapter
from selftrade.live.multi_symbol_manager import MultiSymbolLiveManager
from selftrade.analysis.pattern_engine import MarketStructure, TrendType, CandlestickPattern


def test_session_learning_report_generation(tmp_path):
    reports_dir = str(tmp_path / "reports")
    generator = SessionLearningReportGenerator(reports_dir=reports_dir)

    start_time = datetime.now() - timedelta(minutes=30)
    end_time = datetime.now()

    dummy_trades = [
        {"symbol": "XAUUSD", "pnl_usd": 150.0, "pnl_pct": 1.5},
        {"symbol": "XAUUSD", "pnl_usd": -50.0, "pnl_pct": -0.5},
        {"symbol": "EURUSD", "pnl_usd": 100.0, "pnl_pct": 1.0},
    ]

    report_data = generator.generate_report(
        session_start_time=start_time,
        session_end_time=end_time,
        execution_mode="Simulated Paper",
        active_symbols=["XAUUSD", "EURUSD"],
        initial_capital=10000.0,
        final_capital=10200.0,
        trades_history=dummy_trades,
        model_versions_trained=["v1.0_multi_active_finetuned_ep01"],
        active_model_version="v1.0_multi_active_finetuned_ep01",
    )

    assert report_data["total_trades"] == 3
    assert report_data["num_wins"] == 2
    assert report_data["num_losses"] == 1
    assert pytest.approx(report_data["win_rate"], 0.1) == 66.7
    assert report_data["net_pnl_usd"] == 200.0
    assert report_data["gross_profit"] == 250.0
    assert report_data["gross_loss"] == 50.0
    assert report_data["profit_factor"] == 5.0
    assert len(report_data["model_versions_trained"]) == 1

    assert os.path.exists(report_data["report_filepath"])
    with open(report_data["report_filepath"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "# 🧠 SelfTrade-RL | Session Learning & Performance Report" in content
        assert "Session Win Rate" in content
        assert "XAUUSD" in content


def test_trade_reasoner_confluence_threshold():
    reasoner = TradeReasoner(target_risk_reward=2.0)
    assert reasoner.min_confluence_threshold == 0.65

    # Mock market structure with weak confluence
    market_struct = MarketStructure(
        trend=TrendType.SIDEWAYS_CONSOLIDATION,
        previous_trend=TrendType.SIDEWAYS_CONSOLIDATION,
        candlestick_pattern=CandlestickPattern.NONE,
        support_level=100.0,
        resistance_level=110.0,
        ema_fast=102.0,
        ema_slow=103.0,
        ema_baseline=102.5,
        trend_score=0.1,
    )

    dates = pd.date_range("2026-01-01", periods=50, freq="5min")
    df = pd.DataFrame({
        "close": np.linspace(100, 105, 50),
        "high": np.linspace(101, 106, 50),
        "low": np.linspace(99, 104, 50),
        "open": np.linspace(100, 105, 50),
    }, index=dates)

    decision = reasoner.evaluate_trade(
        proposed_action=1,  # Buy signal
        current_price=105.0,
        atr=1.0,
        market_structure=market_struct,
        df=df,
    )

    # Confluence should be low (< 0.65) so approved action must be rejected to HOLD (0)
    assert decision.confluence_score < 0.65
    assert decision.approved_action == 0


@patch("selftrade.live.mt5_adapter.mt5")
def test_mt5_adapter_ensure_connected(mock_mt5):
    adapter = MT5BrokerAdapter(account=123456, password="pwd", server="MetaQuotes-Demo")

    # Scenario 1: Already connected
    adapter.is_connected = True
    assert adapter.ensure_connected() is True

    # Scenario 2: Disconnected, reconnection succeeds
    adapter.is_connected = False
    mock_mt5.initialize.return_value = True
    mock_mt5.login.return_value = True

    assert adapter.ensure_connected() is True
    assert adapter.is_connected is True


def test_multi_symbol_manager_report_on_stop(tmp_path):
    journal_file = str(tmp_path / "test_journal.json")
    manager = MultiSymbolLiveManager(target_rr_ratio=2.0, journal_path=journal_file)

    symbols = ["XAUUSD", "EURUSD"]
    manager.start_trading(symbols=symbols, mode="Simulated Paper")

    # Let worker loop run briefly
    import time
    time.sleep(0.5)

    report = manager.stop_trading()

    assert report is not None
    assert "session_id" in report
    assert "win_rate" in report
    assert "report_filepath" in report
    assert os.path.exists(report["report_filepath"])


def test_category_strategy_optimization_and_multi_positions():
    reasoner = TradeReasoner(target_risk_reward=2.0)

    market_struct = MarketStructure(
        trend=TrendType.BULLISH_UPTREND,
        previous_trend=TrendType.BULLISH_UPTREND,
        candlestick_pattern=CandlestickPattern.BULLISH_ENGULFING,
        support_level=2600.0,
        resistance_level=2700.0,
        ema_fast=2650.0,
        ema_slow=2630.0,
        ema_baseline=2620.0,
        trend_score=0.8,
        trend_continuation_signal=True,
    )

    dates = pd.date_range("2026-01-01", periods=50, freq="5min")
    df = pd.DataFrame({
        "close": np.linspace(2650, 2670, 50),
        "high": np.linspace(2655, 2675, 50),
        "low": np.linspace(2645, 2665, 50),
        "open": np.linspace(2650, 2670, 50),
    }, index=dates)

    # Gold evaluation (XAUUSD)
    decision_gold = reasoner.evaluate_trade(
        proposed_action=1,
        current_price=2670.0,
        atr=15.0,
        market_structure=market_struct,
        df=df,
        timestamp=pd.Timestamp("2026-09-14 14:30:00"),
        symbol="XAUUSD",
    )
    assert decision_gold.approved_action == 1

    # Crypto evaluation (BTCUSD)
    decision_btc = reasoner.evaluate_trade(
        proposed_action=1,
        current_price=65000.0,
        atr=500.0,
        market_structure=market_struct,
        df=df,
        timestamp=pd.Timestamp("2026-09-14 14:30:00"),
        symbol="BTCUSD",
    )
    assert decision_btc.approved_action == 1
