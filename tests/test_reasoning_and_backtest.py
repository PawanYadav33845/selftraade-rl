"""
Unit Tests for Pattern Detection, Multi-Factor Trade Reasoning, and Strategy Backtesting.
"""

import pytest
import numpy as np

from selftrade.data.generator import MarketDataGenerator
from selftrade.agent.ppo_agent import PPOAgent
from selftrade.analysis.pattern_engine import PatternTrendEngine, TrendType, CandlestickPattern
from selftrade.reasoning.trade_reasoner import TradeReasoner, ReasonedTradeDecision
from selftrade.backtest.backtester import StrategyBacktester, BacktestReport


def test_pattern_trend_engine_analysis():
    gen = MarketDataGenerator(seed=42)
    df = gen.generate_symbol_data(symbol="XAUUSD", num_steps=100)

    engine = PatternTrendEngine()
    market_struct = engine.analyze_chart(df)

    assert isinstance(market_struct.trend, TrendType)
    assert isinstance(market_struct.candlestick_pattern, CandlestickPattern)
    assert market_struct.support_level > 0.0
    assert market_struct.resistance_level >= market_struct.support_level


def test_trade_reasoner_tp_sl_calculation():
    reasoner = TradeReasoner(min_confluence_threshold=0.60, target_risk_reward=2.0)

    gen = MarketDataGenerator(seed=42)
    df = gen.generate_symbol_data(symbol="XAUUSD", num_steps=100)
    engine = PatternTrendEngine()
    market_struct = engine.analyze_chart(df)

    current_price = float(df["close"].iloc[-1])
    atr = current_price * 0.01

    # Evaluate BUY trade
    decision: ReasonedTradeDecision = reasoner.evaluate_trade(
        proposed_action=1,
        current_price=current_price,
        atr=atr,
        market_structure=market_struct,
    )

    assert decision.stop_loss_price < current_price
    assert decision.take_profit_price > current_price
    assert decision.risk_reward_ratio == 2.0
    assert decision.stop_loss_price > 0.0


def test_strategy_backtester():
    gen = MarketDataGenerator(seed=42)
    df = gen.generate_symbol_data(symbol="GBPUSD", num_steps=150)
    agent = PPOAgent()

    backtester = StrategyBacktester(initial_capital=10000.0)
    report: BacktestReport = backtester.run_backtest(df, agent)

    assert isinstance(report, BacktestReport)
    assert report.initial_capital == 10000.0
    assert report.win_rate_pct >= 0.0 and report.win_rate_pct <= 100.0
    assert report.total_trades >= 0


def test_trade_reasoner_trend_memory_confluence():
    reasoner = TradeReasoner(min_confluence_threshold=0.60, target_risk_reward=2.0)
    gen = MarketDataGenerator(seed=42)
    df = gen.generate_symbol_data(symbol="BTCUSD", num_steps=100)
    engine = PatternTrendEngine()
    market_struct = engine.analyze_chart(df)

    # Force continuation signal
    market_struct.trend_continuation_signal = True
    market_struct.trend = TrendType.BULLISH_UPTREND

    decision = reasoner.evaluate_trade(
        proposed_action=1,
        current_price=50000.0,
        atr=500.0,
        market_structure=market_struct,
        df=df,
    )

    assert "Trend Memory" in decision.reasoning_summary
    assert decision.confluence_score > 0.50

