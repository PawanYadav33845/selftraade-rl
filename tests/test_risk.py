"""
Unit Tests for Deterministic Risk & Safety Engine (Hard Gating Layer).
"""

import pytest
from selftrade.risk.risk_engine import DeterministicRiskEngine, RiskEvaluationResult


def test_risk_engine_stop_loss_trigger():
    risk_engine = DeterministicRiskEngine(max_stop_loss_pct=0.02)  # 2.0% stop loss

    env_info = {
        "close_price": 97.0,              # Current price down 3.0%
        "position": 10.0,
        "position_entry_price": 100.0,    # Entry price 100
        "portfolio_value": 970.0,
        "cash": 0.0,
    }

    # Proposed action: 0 (Hold)
    result: RiskEvaluationResult = risk_engine.evaluate_action(proposed_action=0, env_info=env_info)

    # Should override action 0 -> 2 (Sell) due to hard stop-loss breach
    assert result.is_overridden is True
    assert result.safe_action == 2
    assert "HARD_STOP_LOSS_TRIGGERED" in result.override_reason


def test_risk_engine_circuit_breaker_trigger():
    risk_engine = DeterministicRiskEngine(max_daily_drawdown_pct=0.05, circuit_breaker_cooldown=24)

    # Simulate portfolio drop from $10,000 to $9,400 (> 5% drawdown) over steps
    for step in range(5):
        val = 10000.0 if step < 2 else 9400.0
        env_info = {
            "close_price": 100.0,
            "position": 0.0,
            "position_entry_price": 0.0,
            "portfolio_value": val,
            "cash": val,
        }
        res = risk_engine.evaluate_action(proposed_action=0, env_info=env_info)

    # Now attempt a BUY (action 1) while circuit breaker is active
    env_info = {
        "close_price": 100.0,
        "position": 0.0,
        "position_entry_price": 0.0,
        "portfolio_value": 9400.0,
        "cash": 9400.0,
    }
    result = risk_engine.evaluate_action(proposed_action=1, env_info=env_info)

    assert result.is_overridden is True
    assert result.safe_action == 0  # Override to Hold
    assert result.circuit_breaker_active is True
    assert "CIRCUIT_BREAKER_ACTIVE" in result.override_reason


def test_risk_engine_insufficient_capital_buy_override():
    risk_engine = DeterministicRiskEngine(min_cash_ratio_for_buy=0.01)

    env_info = {
        "close_price": 100.0,
        "position": 100.0,
        "position_entry_price": 100.0,
        "portfolio_value": 10000.0,
        "cash": 0.0,  # Zero cash
    }

    # Attempt BUY (action 1) when cash is 0
    result = risk_engine.evaluate_action(proposed_action=1, env_info=env_info)

    assert result.is_overridden is True
    assert result.safe_action == 0  # Override to Hold
    assert "INSUFFICIENT_CAPITAL_FOR_BUY" in result.override_reason
