"""
Deterministic Risk & Safety Engine (Hard Gating Layer).
Enforces non-overridable execution rules outside the neural network:
1. Hard Stop-Loss (-2.0% per position)
2. Daily Circuit Breaker (-5.0% portfolio drawdown cap)
3. Order & Capital Validation
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class RiskEvaluationResult:
    """Dataclass holding risk evaluation verdict."""
    original_action: int
    safe_action: int
    is_overridden: bool
    override_reason: str
    circuit_breaker_active: bool


class DeterministicRiskEngine:
    """
    Hard Gating Layer wrapping agent action execution.
    """

    def __init__(
        self,
        max_stop_loss_pct: float = 0.02,        # 2.0% max loss per position
        max_daily_drawdown_pct: float = 0.05,    # 5.0% 24h portfolio drawdown cap
        circuit_breaker_cooldown: int = 24,      # 24 steps cooldown period
        min_cash_ratio_for_buy: float = 0.01,    # Minimum cash needed to initiate buy
    ) -> None:
        self.max_stop_loss_pct = float(max_stop_loss_pct)
        self.max_daily_drawdown_pct = float(max_daily_drawdown_pct)
        self.circuit_breaker_cooldown = int(circuit_breaker_cooldown)
        self.min_cash_ratio_for_buy = float(min_cash_ratio_for_buy)

        # Circuit Breaker tracking
        self.portfolio_window = deque(maxlen=24)
        self.circuit_breaker_timer = 0

    def evaluate_action(
        self,
        proposed_action: int,
        env_info: Dict[str, Any],
    ) -> RiskEvaluationResult:
        """
        Evaluates proposed action against hard risk rules and returns safe execution action.

        Args:
            proposed_action: Action integer proposed by RL Agent (0: Hold, 1: Buy, 2: Sell)
            env_info: Current environment telemetry dictionary (cash, position, prices, etc.)

        Returns:
            RiskEvaluationResult object containing verdict and reason.
        """
        safe_action = proposed_action
        is_overridden = False
        reason = "PASSED_RISK_GATES"

        close_price = env_info.get("close_price", 0.0)
        position = env_info.get("position", 0.0)
        entry_price = env_info.get("position_entry_price", 0.0)
        portfolio_value = env_info.get("portfolio_value", 1.0)
        cash = env_info.get("cash", 0.0)

        # Update rolling portfolio history for daily circuit breaker
        self.portfolio_window.append(portfolio_value)

        # 1. Update Circuit Breaker Cooldown Timer
        if self.circuit_breaker_timer > 0:
            self.circuit_breaker_timer -= 1

        # Check 24-step peak-to-trough drawdown for Circuit Breaker
        if len(self.portfolio_window) > 1:
            window_peak = max(self.portfolio_window)
            window_drawdown = (window_peak - portfolio_value) / (window_peak + 1e-8)
            if window_drawdown >= self.max_daily_drawdown_pct:
                self.circuit_breaker_timer = self.circuit_breaker_cooldown

        circuit_breaker_active = self.circuit_breaker_timer > 0

        # 2. Rule 1: HARD STOP-LOSS CHECK
        if position > 1e-8 and entry_price > 0.0:
            pos_drawdown = (entry_price - close_price) / entry_price
            if pos_drawdown >= self.max_stop_loss_pct:
                safe_action = 2  # Override to SELL immediately
                is_overridden = True
                reason = f"HARD_STOP_LOSS_TRIGGERED (Position Drawdown {pos_drawdown*100:.2f}% >= {self.max_stop_loss_pct*100:.2f}%)"
                return RiskEvaluationResult(
                    original_action=proposed_action,
                    safe_action=safe_action,
                    is_overridden=is_overridden,
                    override_reason=reason,
                    circuit_breaker_active=circuit_breaker_active,
                )

        # 3. Rule 2: DAILY CIRCUIT BREAKER CHECK
        if circuit_breaker_active:
            if proposed_action == 1:  # Cease Buy orders during circuit breaker
                safe_action = 0  # Override to HOLD
                is_overridden = True
                reason = f"CIRCUIT_BREAKER_ACTIVE ({self.circuit_breaker_timer} steps remaining). BUY blocked."
                return RiskEvaluationResult(
                    original_action=proposed_action,
                    safe_action=safe_action,
                    is_overridden=is_overridden,
                    override_reason=reason,
                    circuit_breaker_active=circuit_breaker_active,
                )

        # 4. Rule 3: CAPITAL & ORDER VALIDATION CHECK
        if proposed_action == 1:  # BUY request
            cash_ratio = cash / (portfolio_value + 1e-8)
            if cash_ratio < self.min_cash_ratio_for_buy or cash < 1.0:
                safe_action = 0  # Override to HOLD
                is_overridden = True
                reason = f"INSUFFICIENT_CAPITAL_FOR_BUY (Cash Ratio {cash_ratio*100:.2f}% < {self.min_cash_ratio_for_buy*100:.2f}%)"
                return RiskEvaluationResult(
                    original_action=proposed_action,
                    safe_action=safe_action,
                    is_overridden=is_overridden,
                    override_reason=reason,
                    circuit_breaker_active=circuit_breaker_active,
                )

        return RiskEvaluationResult(
            original_action=proposed_action,
            safe_action=safe_action,
            is_overridden=is_overridden,
            override_reason=reason,
            circuit_breaker_active=circuit_breaker_active,
        )
