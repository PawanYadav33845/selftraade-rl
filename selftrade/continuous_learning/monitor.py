"""
Performance and Market Regime Monitor.
Tracks rolling Sharpe Ratio, volatility/ATR shift detection, and periodic timestep triggers.
"""

from typing import List, Tuple, Optional
from enum import Enum
from collections import deque
import numpy as np


class TriggerEvent(Enum):
    NONE = "NONE"
    SHARPE_DROP = "SHARPE_DROP"
    REGIME_SHIFT = "REGIME_SHIFT"
    PERIODIC_TIMESTEP = "PERIODIC_TIMESTEP"


class PerformanceMonitor:
    """
    Monitors agent returns and market volatility to detect when re-training is required.
    """

    def __init__(
        self,
        sharpe_window: int = 100,
        sharpe_threshold: float = 0.5,
        volatility_window: int = 50,
        volatility_shift_threshold: float = 0.25,  # 25% change
        periodic_interval: int = 2048,
    ) -> None:
        self.sharpe_window = sharpe_window
        self.sharpe_threshold = sharpe_threshold
        self.volatility_window = volatility_window
        self.volatility_shift_threshold = volatility_shift_threshold
        self.periodic_interval = periodic_interval

        self.returns_history = deque(maxlen=sharpe_window)
        self.volatility_history = deque(maxlen=volatility_window)

        self.baseline_volatility: Optional[float] = None
        self.total_timesteps = 0
        self.last_training_step = 0

    def step(self, reward: float, market_atr_norm: float) -> Tuple[TriggerEvent, str]:
        """
        Updates monitor state with latest step data and checks for trigger conditions.

        Args:
            reward: Step reward / return.
            market_atr_norm: Current normalized market ATR volatility measure.

        Returns:
            (TriggerEvent, trigger_reason_description)
        """
        self.total_timesteps += 1
        self.returns_history.append(reward)
        self.volatility_history.append(market_atr_norm)

        # Establish baseline volatility
        if self.baseline_volatility is None and len(self.volatility_history) >= 20:
            self.baseline_volatility = float(np.mean(self.volatility_history))

        # Check Trigger 1: Periodic Timestep Interval
        steps_since_last = self.total_timesteps - self.last_training_step
        if steps_since_last >= self.periodic_interval:
            self.last_training_step = self.total_timesteps
            return TriggerEvent.PERIODIC_TIMESTEP, f"Periodic interval elapsed ({steps_since_last} steps)"

        # Check Trigger 2: Rolling Sharpe Ratio Drop
        if len(self.returns_history) >= 30:
            returns_arr = np.array(self.returns_history)
            mean_ret = float(np.mean(returns_arr))
            std_ret = float(np.std(returns_arr)) + 1e-8
            sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252.0)  # Annualized approximation

            if sharpe_ratio < self.sharpe_threshold and steps_since_last >= 100:
                self.last_training_step = self.total_timesteps
                return TriggerEvent.SHARPE_DROP, f"Rolling Sharpe Ratio dropped ({sharpe_ratio:.2f} < {self.sharpe_threshold:.2f})"

        # Check Trigger 3: Regime Shift (Volatility/ATR Change > 25%)
        if self.baseline_volatility is not None and len(self.volatility_history) >= 20:
            current_vol = float(np.mean(self.volatility_history))
            vol_change = abs(current_vol - self.baseline_volatility) / (self.baseline_volatility + 1e-8)

            if vol_change >= self.volatility_shift_threshold and steps_since_last >= 100:
                self.baseline_volatility = current_vol  # Reset baseline to new regime level
                self.last_training_step = self.total_timesteps
                return TriggerEvent.REGIME_SHIFT, f"Market Regime Shift detected (Volatility changed by {vol_change*100:.1f}% >= 25%)"

        return TriggerEvent.NONE, "NO_TRIGGER"

    def compute_sharpe_ratio(self) -> float:
        """Computes current rolling Sharpe Ratio."""
        if len(self.returns_history) < 5:
            return 0.0
        returns_arr = np.array(self.returns_history)
        std = np.std(returns_arr) + 1e-8
        return float((np.mean(returns_arr) / std) * np.sqrt(252.0))
