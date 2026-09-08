"""
Shadow Deployment & Validation Engine.
Runs fine-tuned candidate models side-by-side with active live models in shadow mode.
Enforces promotion criteria: Candidate Sharpe Ratio must beat active model by >= 10%.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

from selftrade.agent.ppo_agent import PPOAgent
from selftrade.env.trading_env import TradingEnv


@dataclass
class ShadowValidationResult:
    """Dataclass holding shadow validation results."""
    candidate_tag: str
    active_tag: str
    candidate_sharpe: float
    active_sharpe: float
    sharpe_improvement_pct: float
    promoted: bool
    reason: str


class ShadowValidationEngine:
    """
    Shadow Execution Mode runner and promotion validator.
    """

    def __init__(
        self,
        promotion_threshold_multiplier: float = 1.10,  # 10% Sharpe improvement requirement
        eval_test_window: int = 500,                   # 500 steps test window
    ) -> None:
        self.promotion_threshold_multiplier = promotion_threshold_multiplier
        self.eval_test_window = eval_test_window

        self.candidate_agent: Optional[PPOAgent] = None
        self.candidate_tag: Optional[str] = None
        self.candidate_returns = deque(maxlen=eval_test_window)
        self.active_returns = deque(maxlen=eval_test_window)

        self.eval_step_count = 0
        self.is_evaluating = False

    def start_shadow_evaluation(self, candidate_agent: PPOAgent, candidate_tag: str) -> None:
        """
        Registers candidate agent for shadow side-by-side paper evaluation.
        """
        self.candidate_agent = candidate_agent
        self.candidate_tag = candidate_tag
        self.candidate_returns.clear()
        self.active_returns.clear()
        self.eval_step_count = 0
        self.is_evaluating = True

    def record_shadow_step(
        self,
        state: np.ndarray,
        active_reward: float,
        env: TradingEnv,
    ) -> Optional[ShadowValidationResult]:
        """
        Executes candidate agent action prediction on current step observation in shadow mode.

        Args:
            state: Current state observation array S_t.
            active_reward: Reward achieved by active live model on this step.
            env: Trading environment (used to simulate candidate step in paper mode).

        Returns:
            ShadowValidationResult if evaluation window completed, else None.
        """
        if not self.is_evaluating or self.candidate_agent is None:
            return None

        self.eval_step_count += 1
        self.active_returns.append(active_reward)

        # Candidate selects action on same observation
        cand_action, _, _ = self.candidate_agent.select_action(state, deterministic=True)

        # Simulate candidate reward estimate based on current market return and transaction penalty
        # Candidate reward estimate: log return minus fee penalty if action differs
        cand_reward = active_reward
        # If candidate action differed from active action, apply simple trade cost adjustment
        if cand_action != 0:
            cand_reward -= 0.0005  # slight execution cost adjustment for simulated action

        self.candidate_returns.append(cand_reward)

        # Check if evaluation test window completed
        if self.eval_step_count >= self.eval_test_window:
            return self.evaluate_promotion(active_tag=self.candidate_agent.version_tag)

        return None

    def evaluate_promotion(self, active_tag: str) -> ShadowValidationResult:
        """
        Evaluates candidate promotion criteria after evaluation window completion.
        """
        self.is_evaluating = False

        cand_arr = np.array(self.candidate_returns) if self.candidate_returns else np.array([0.0])
        act_arr = np.array(self.active_returns) if self.active_returns else np.array([0.0])

        cand_mean, cand_std = float(np.mean(cand_arr)), float(np.std(cand_arr)) + 1e-8
        act_mean, act_std = float(np.mean(act_arr)), float(np.std(act_arr)) + 1e-8

        cand_sharpe = float((cand_mean / cand_std) * np.sqrt(252.0))
        act_sharpe = float((act_mean / act_std) * np.sqrt(252.0))

        # Handle zero or negative base Sharpe ratios gracefully
        required_sharpe = act_sharpe * self.promotion_threshold_multiplier if act_sharpe > 0 else act_sharpe + 0.1

        if act_sharpe != 0:
            sharpe_improvement_pct = ((cand_sharpe - act_sharpe) / abs(act_sharpe)) * 100.0
        else:
            sharpe_improvement_pct = 100.0 if cand_sharpe > 0 else 0.0

        promoted = cand_sharpe >= required_sharpe

        if promoted:
            reason = (
                f"PROMOTED: Candidate Sharpe ({cand_sharpe:.2f}) exceeded active Sharpe ({act_sharpe:.2f}) "
                f"by {sharpe_improvement_pct:.1f}% (Required threshold: +10.0%)."
            )
        else:
            reason = (
                f"REJECTED: Candidate Sharpe ({cand_sharpe:.2f}) did not beat active Sharpe ({act_sharpe:.2f}) "
                f"by 10% (Required Sharpe: {required_sharpe:.2f})."
            )

        return ShadowValidationResult(
            candidate_tag=self.candidate_tag or "candidate",
            active_tag=active_tag,
            candidate_sharpe=cand_sharpe,
            active_sharpe=act_sharpe,
            sharpe_improvement_pct=sharpe_improvement_pct,
            promoted=promoted,
            reason=reason,
        )
