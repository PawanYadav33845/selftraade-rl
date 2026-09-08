"""
Self-Improving Engine Module.
Orchestrates transition logging, trigger detection, and online fine-tuning using 70/30 mixed batching.
"""

from typing import Dict, Any, Optional, Callable, Tuple
import os
import numpy as np

from selftrade.agent.ppo_agent import PPOAgent
from selftrade.agent.buffer import Transition, LiveExperienceBuffer, HistoricalBuffer, MixedBatchSampler
from selftrade.continuous_learning.monitor import PerformanceMonitor, TriggerEvent


class SelfImprovingEngine:
    """
    Continuous Learning Engine managing online experience collection and model fine-tuning.
    """

    def __init__(
        self,
        agent: PPOAgent,
        live_buffer: LiveExperienceBuffer,
        historical_buffer: HistoricalBuffer,
        monitor: PerformanceMonitor,
        fine_tune_epochs: int = 5,
        batch_size: int = 64,
        models_dir: str = "checkpoints",
    ) -> None:
        self.agent = agent
        self.live_buffer = live_buffer
        self.historical_buffer = historical_buffer
        self.monitor = monitor
        self.fine_tune_epochs = fine_tune_epochs
        self.batch_size = batch_size
        self.models_dir = models_dir

        self.version_counter = 1
        self.sampler = MixedBatchSampler(self.live_buffer, self.historical_buffer, live_ratio=0.70)

        os.makedirs(self.models_dir, exist_ok=True)

    def record_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
    ) -> Transition:
        """
        Logs live transition into the circular replay buffer.
        """
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
        )
        self.live_buffer.add(transition)
        return transition

    def check_and_fine_tune(
        self,
        reward: float,
        market_atr_norm: float,
        on_candidate_created_cb: Optional[Callable[[PPOAgent, str], None]] = None,
    ) -> Tuple[bool, Optional[PPOAgent], str]:
        """
        Evaluates trigger conditions and executes fine-tuning if triggered.

        Args:
            reward: Step reward.
            market_atr_norm: Volatility metric.
            on_candidate_created_cb: Callback function invoked when candidate agent is ready.

        Returns:
            (triggered_bool, candidate_agent_or_none, trigger_description)
        """
        trigger, reason = self.monitor.step(reward, market_atr_norm)

        if trigger == TriggerEvent.NONE:
            return False, None, reason

        # Trigger fired! Create candidate agent clone for fine-tuning
        self.version_counter += 1
        candidate_tag = f"model_v1.{self.version_counter}_candidate"
        candidate_agent = self.agent.clone(new_version_tag=candidate_tag)

        # Fine-tune candidate agent using 70/30 mixed batch sampling
        fine_tune_stats = self._execute_fine_tuning(candidate_agent)

        # Save versioned candidate checkpoint
        checkpoint_path = os.path.join(self.models_dir, f"{candidate_tag}.pt")
        candidate_agent.save(checkpoint_path)

        desc = f"TRIGGER [{trigger.value}]: {reason}. Fine-tuned candidate model {candidate_tag} saved."

        if on_candidate_created_cb is not None:
            on_candidate_created_cb(candidate_agent, candidate_tag)

        return True, candidate_agent, desc

    def _execute_fine_tuning(self, candidate_agent: PPOAgent) -> Dict[str, float]:
        """
        Executes PPO update epochs using mixed sampling buffer (70% live + 30% historical).
        """
        if len(self.live_buffer) == 0 and len(self.historical_buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        accum_stats = []
        for _ in range(self.fine_tune_epochs):
            batch = self.sampler.sample_batch(batch_size=self.batch_size)
            stats = candidate_agent.update(batch, ppo_epochs=2)
            accum_stats.append(stats)

        avg_policy_loss = float(np.mean([s["policy_loss"] for s in accum_stats]))
        avg_value_loss = float(np.mean([s["value_loss"] for s in accum_stats]))
        avg_entropy = float(np.mean([s["entropy"] for s in accum_stats]))

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
        }
