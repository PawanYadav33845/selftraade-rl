"""
Unit Tests for Shadow Deployment & Validation Engine.
"""

import pytest
import numpy as np

from selftrade.agent.ppo_agent import PPOAgent
from selftrade.data.generator import MarketDataGenerator
from selftrade.env.trading_env import TradingEnv
from selftrade.shadow.shadow_engine import ShadowValidationEngine, ShadowValidationResult


def test_shadow_validation_engine_promotion():
    gen = MarketDataGenerator(seed=42)
    df = gen.generate_ohlcv(num_steps=100)
    env = TradingEnv(df=df)

    active_agent = PPOAgent(version_tag="v1.0_active")
    candidate_agent = PPOAgent(version_tag="v1.1_candidate")

    shadow_engine = ShadowValidationEngine(
        promotion_threshold_multiplier=1.10,
        eval_test_window=20,  # Short window for test
    )

    shadow_engine.start_shadow_evaluation(candidate_agent=candidate_agent, candidate_tag="v1.1_candidate")
    assert shadow_engine.is_evaluating is True

    result = None
    for step in range(20):
        state = np.random.randn(7).astype(np.float32)
        result = shadow_engine.record_shadow_step(state=state, active_reward=0.01, env=env)

    assert result is not None
    assert isinstance(result, ShadowValidationResult)
    assert result.candidate_tag == "v1.1_candidate"
    assert shadow_engine.is_evaluating is False
