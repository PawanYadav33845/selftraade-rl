"""
Unit and Integration Tests for GPU Acceleration, Multi-Strategy Ensemble,
Portfolio Position Scaling (>2 trades), and Online Self-Improving PPO Engine.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from selftrade.agent.ppo_agent import PPOAgent
from selftrade.strategies.ensemble import StrategyEnsemble
from selftrade.risk.risk_engine import DeterministicRiskEngine, RiskEvaluationResult
from selftrade.reasoning.trade_reasoner import TradeReasoner, ReasonedTradeDecision
from selftrade.analysis.pattern_engine import PatternTrendEngine
from selftrade.data.generator import MarketDataGenerator
from selftrade.agent.buffer import LiveExperienceBuffer, HistoricalBuffer
from selftrade.continuous_learning.monitor import PerformanceMonitor
from selftrade.continuous_learning.self_improver import SelfImprovingEngine
from selftrade.live.multi_symbol_manager import MultiSymbolLiveManager


def test_device_auto_detection():
    agent = PPOAgent()
    assert agent.device.type in ["cuda", "cpu"]
    is_cuda = False
    try:
        is_cuda = torch.cuda.is_available()
    except Exception:
        is_cuda = False
    if is_cuda:
        assert agent.device.type == "cuda"
        assert next(agent.network.parameters()).is_cuda


def test_strategy_ensemble_expanded():
    ensemble = StrategyEnsemble()
    generator = MarketDataGenerator(seed=42)
    df = generator.generate_symbol_data(symbol="XAUUSD", num_steps=200)
    pattern_engine = PatternTrendEngine()
    market_struct = pattern_engine.analyze_chart(df)

    res = ensemble.evaluate_strategies(df, market_struct)
    assert res.ensemble_bias in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert len(res.strategy_votes) >= 8
    assert "Stochastic_Oscillator" in res.strategy_votes
    assert "Keltner_Channel_Breakout" in res.strategy_votes
    assert -1.0 <= res.overall_score <= 1.0


def test_portfolio_position_cap():
    risk_engine = DeterministicRiskEngine(max_concurrent_trades=10)
    mock_info = {
        "portfolio_value": 10000.0,
        "position": 0.0,
        "position_entry_price": 0.0,
        "cash": 10000.0,
        "total_open_trades": 9,
    }

    # 10th position should be approved
    verdict = risk_engine.evaluate_action(1, mock_info)
    assert verdict.safe_action == 1

    # 11th position should be blocked (cap is 10)
    mock_info["total_open_trades"] = 10
    verdict_blocked = risk_engine.evaluate_action(1, mock_info)
    assert verdict_blocked.safe_action == 0
    assert "MAX_CONCURRENT_TRADES_CAP_REACHED" in verdict_blocked.override_reason


def test_trade_reasoner_adaptive_rr():
    reasoner = TradeReasoner(target_risk_reward=2.5)
    generator = MarketDataGenerator(seed=100)
    df = generator.generate_symbol_data(symbol="BTCUSD", num_steps=150)

    decision = reasoner.evaluate_trade(
        proposed_action=1,
        current_price=50000.0,
        atr=500.0,
        market_structure=None,
        df=df,
    )

    assert decision.holding_mode in ["SCALP_QUICK", "TREND_RUNNER"]
    assert decision.stop_loss_price < 50000.0
    assert decision.take_profit_price > 50000.0
    assert decision.break_even_trigger_price > 50000.0
    assert decision.risk_reward_ratio >= 1.5


def test_online_self_improver_trigger():
    agent = PPOAgent()
    live_buf = LiveExperienceBuffer(capacity=100)
    hist_buf = HistoricalBuffer()
    monitor = PerformanceMonitor(periodic_interval=5)

    engine = SelfImprovingEngine(
        agent=agent,
        live_buffer=live_buf,
        historical_buffer=hist_buf,
        monitor=monitor,
        fine_tune_epochs=1,
        batch_size=4,
    )

    state = np.random.randn(7).astype(np.float32)
    next_state = np.random.randn(7).astype(np.float32)

    triggered = False
    candidate = None
    desc = ""

    for i in range(10):
        engine.record_transition(
            state=state,
            action=1,
            reward=0.5,
            next_state=next_state,
            done=False,
            log_prob=-0.5,
            value=0.2,
        )
        trig, cand, d = engine.check_and_fine_tune(reward=0.5, market_atr_norm=0.02)
        if trig:
            triggered = trig
            candidate = cand
            desc = d

    assert triggered is True
    assert candidate is not None
    assert "model_v1." in desc


def test_multi_symbol_manager_simulation():
    manager = MultiSymbolLiveManager(target_rr_ratio=2.0)
    symbols = ["XAUUSD", "EURUSD", "BTCUSD"]
    success = manager.start_trading(symbols=symbols, mode="Simulated Paper")
    assert success is True

    import time
    time.sleep(2.0)
    assert manager.is_running is True

    for sym in symbols:
        assert sym in manager.telemetry
        telem = manager.telemetry[sym]
        assert telem.last_price > 0.0

    manager.stop_trading()
    assert manager.is_running is False
