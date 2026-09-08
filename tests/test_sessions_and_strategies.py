"""
Unit Tests for Market Opening Sessions and Multi-Strategy Ensemble.
"""

import pytest
import pandas as pd
import numpy as np

from selftrade.sessions.market_sessions import MarketSessionsEngine, MarketSessionType, MarketSessionInfo
from selftrade.strategies.ensemble import StrategyEnsemble, EnsembleEvaluationResult
from selftrade.analysis.pattern_engine import PatternTrendEngine
from selftrade.data.generator import MarketDataGenerator


def test_market_sessions_engine():
    # London/NY Overlap (14:00 UTC)
    ts_overlap = pd.Timestamp("2026-09-08 14:00:00", tz="UTC")
    info_overlap = MarketSessionsEngine.get_session_info(ts_overlap)
    assert info_overlap.session_type == MarketSessionType.LONDON_NY_OVERLAP
    assert info_overlap.is_high_liquidity is True
    assert info_overlap.volatility_multiplier == 1.5

    # Asian Session (04:00 UTC)
    ts_asian = pd.Timestamp("2026-09-08 04:00:00", tz="UTC")
    info_asian = MarketSessionsEngine.get_session_info(ts_asian)
    assert info_asian.session_type == MarketSessionType.ASIAN_SESSION
    assert info_asian.is_high_liquidity is False


def test_strategy_ensemble_evaluation():
    gen = MarketDataGenerator(seed=42)
    df = gen.generate_symbol_data(symbol="XAUUSD", num_steps=100)
    pattern_engine = PatternTrendEngine()
    market_struct = pattern_engine.analyze_chart(df)

    ensemble = StrategyEnsemble()
    res: EnsembleEvaluationResult = ensemble.evaluate_strategies(df, market_struct)

    assert isinstance(res, EnsembleEvaluationResult)
    assert res.overall_score >= -1.0 and res.overall_score <= 1.0
    assert res.ensemble_bias in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert "EMA_Trend_Cross" in res.strategy_votes
    assert "RSI_Oversold_Overbought" in res.strategy_votes
    assert "MACD_Momentum" in res.strategy_votes
