"""
Unit Tests for Performance Monitor and Self-Improving Engine.
"""

import pytest
import tempfile
import numpy as np

from selftrade.agent.ppo_agent import PPOAgent
from selftrade.agent.buffer import LiveExperienceBuffer, HistoricalBuffer, Transition
from selftrade.continuous_learning.monitor import PerformanceMonitor, TriggerEvent
from selftrade.continuous_learning.self_improver import SelfImprovingEngine


def test_performance_monitor_triggers():
    monitor = PerformanceMonitor(
        sharpe_window=30,
        sharpe_threshold=0.5,
        volatility_shift_threshold=0.25,
        periodic_interval=50,
    )

    # 1. Test Periodic Interval Trigger
    for step in range(49):
        trigger, _ = monitor.step(reward=0.01, market_atr_norm=0.02)
        assert trigger == TriggerEvent.NONE

    trigger, reason = monitor.step(reward=0.01, market_atr_norm=0.02)
    assert trigger == TriggerEvent.PERIODIC_TIMESTEP
    assert "Periodic interval" in reason


def test_self_improver_fine_tuning_trigger():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = PPOAgent(state_dim=7, action_dim=3, version_tag="v1.0_base")
        live_buffer = LiveExperienceBuffer(capacity=100)
        hist_buffer = HistoricalBuffer()

        # Fill historical buffer with seed data
        for _ in range(50):
            hist_buffer.add(
                Transition(
                    state=np.random.randn(7),
                    action=0,
                    reward=0.0,
                    next_state=np.random.randn(7),
                    done=False,
                )
            )

        monitor = PerformanceMonitor(periodic_interval=20)
        improver = SelfImprovingEngine(
            agent=agent,
            live_buffer=live_buffer,
            historical_buffer=hist_buffer,
            monitor=monitor,
            models_dir=tmpdir,
        )

        # Log steps up to trigger
        candidate_created = False
        for step in range(20):
            state = np.random.randn(7).astype(np.float32)
            improver.record_transition(
                state=state,
                action=1,
                reward=0.01,
                next_state=state,
                done=False,
            )
            triggered, candidate_agent, desc = improver.check_and_fine_tune(reward=0.01, market_atr_norm=0.02)
            if triggered:
                candidate_created = True
                assert candidate_agent is not None
                assert "candidate" in candidate_agent.version_tag

        assert candidate_created is True
