"""
Unit Tests for Profit Factor Optimization, Adaptive Holding Logic (Scalp Quick vs Trend Runner),
Break-Even Protection, and Dynamic Trailing Stop Loss.
"""

from unittest.mock import MagicMock
import pytest
import pandas as pd
import numpy as np

from selftrade.reasoning.trade_reasoner import TradeReasoner, ReasonedTradeDecision
from selftrade.risk.risk_engine import DeterministicRiskEngine, RiskEvaluationResult
from selftrade.analysis.pattern_engine import MarketStructure, TrendType, CandlestickPattern


def test_trade_reasoner_adaptive_rr_and_dual_targets():
    reasoner = TradeReasoner(min_confluence_threshold=0.50, target_risk_reward=2.0)

    # 1. Test Strong Trend Continuation -> TREND_BIG_PROFIT mode (Scaled R:R >= 3.0)
    market_struct = MarketStructure(
        trend=TrendType.BULLISH_UPTREND,
        previous_trend=TrendType.BULLISH_UPTREND,
        candlestick_pattern=CandlestickPattern.BULLISH_ENGULFING,
        support_level=95.0,
        resistance_level=110.0,
        ema_fast=101.0,
        ema_slow=99.0,
        ema_baseline=98.0,
        trend_score=0.8,
        trend_continuation_signal=True,
        trend_reversal_signal=False,
    )

    decision = reasoner.evaluate_trade(
        proposed_action=1,  # BUY
        current_price=100.0,
        atr=2.0,
        market_structure=market_struct,
    )

    assert decision.approved_action == 1
    assert decision.holding_mode == "TREND_BIG_PROFIT"
    assert decision.risk_reward_ratio >= 3.0
    assert decision.take_profit_tp2 > decision.take_profit_tp1
    assert decision.break_even_trigger_price > 100.0

    # 2. Test Ranging / Neutral Market -> SCALP_QUICK mode
    neutral_struct = MarketStructure(
        trend=TrendType.SIDEWAYS_CONSOLIDATION,
        previous_trend=TrendType.SIDEWAYS_CONSOLIDATION,
        candlestick_pattern=CandlestickPattern.DOJI,
        support_level=98.0,
        resistance_level=102.0,
        ema_fast=100.0,
        ema_slow=100.0,
        ema_baseline=100.0,
        trend_score=0.0,
        trend_continuation_signal=False,
        trend_reversal_signal=False,
    )

    decision_scalp = reasoner.evaluate_trade(
        proposed_action=1,
        current_price=100.0,
        atr=2.0,
        market_structure=neutral_struct,
        timestamp=pd.Timestamp("2026-09-14 14:30:00"),
    )

    assert decision_scalp.approved_action == 1
    assert decision_scalp.holding_mode == "SCALP_QUICK"
    assert decision_scalp.risk_reward_ratio == 2.0


def test_risk_engine_break_even_protection():
    risk_engine = DeterministicRiskEngine()

    # Buy position entry at $100.0, initial SL $97.0 (Risk = $3.0). BE Trigger = $103.0.
    symbol = "EURUSD"
    entry_price = 100.0
    initial_sl = 97.0
    be_trigger = 103.0
    atr = 2.0

    # Current price = $101.0 (< BE trigger) -> No SL modification
    new_sl, modified, reason = risk_engine.update_trailing_stop_and_be(
        symbol=symbol,
        side="BUY",
        current_price=101.0,
        entry_price=entry_price,
        current_sl=initial_sl,
        current_tp=106.0,
        be_trigger_price=be_trigger,
        atr=atr,
        holding_mode="SCALP_QUICK",
    )
    assert not modified
    assert new_sl == initial_sl

    # Current price = $103.5 (>= BE trigger) -> SL shifted to entry price ($100.0)
    new_sl, modified, reason = risk_engine.update_trailing_stop_and_be(
        symbol=symbol,
        side="BUY",
        current_price=103.5,
        entry_price=entry_price,
        current_sl=initial_sl,
        current_tp=106.0,
        be_trigger_price=be_trigger,
        atr=atr,
        holding_mode="SCALP_QUICK",
    )
    assert modified
    assert new_sl == entry_price
    assert reason == "BREAK_EVEN_PROTECTION_ACTIVATED"


def test_risk_engine_dynamic_trailing_stop():
    risk_engine = DeterministicRiskEngine()

    symbol = "BTCUSD"
    entry_price = 50000.0
    initial_sl = 49000.0
    be_trigger = 51000.0
    atr = 1000.0

    # In TREND_BIG_PROFIT mode, price advances to $54000. Trailing SL candidate = 54000 - 1.5*1000 = 52500
    new_sl, modified, reason = risk_engine.update_trailing_stop_and_be(
        symbol=symbol,
        side="BUY",
        current_price=54000.0,
        entry_price=entry_price,
        current_sl=50000.0,  # Currently at BE
        current_tp=56000.0,
        be_trigger_price=be_trigger,
        atr=atr,
        holding_mode="TREND_BIG_PROFIT",
    )

    assert modified
    assert new_sl == 52500.0
    assert reason == "DYNAMIC_TRAILING_STOP_RAISED"
