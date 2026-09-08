"""
Unit Tests for PPO Agent and Dual Memory Buffer.
"""

import pytest
import tempfile
import os
import numpy as np
import torch

from selftrade.agent.ppo_agent import PPOAgent, ActorCriticNetwork
from selftrade.agent.buffer import LiveExperienceBuffer, HistoricalBuffer, MixedBatchSampler, Transition


def test_ppo_agent_select_action():
    agent = PPOAgent(state_dim=7, action_dim=3, version_tag="v1.0_test")
    state = np.random.randn(7).astype(np.float32)

    action, log_prob, val = agent.select_action(state, deterministic=False)
    assert action in [0, 1, 2]
    assert isinstance(log_prob, float)
    assert isinstance(val, float)

    det_action, _, _ = agent.select_action(state, deterministic=True)
    assert det_action in [0, 1, 2]


def test_dual_memory_buffer_mixed_sampling():
    live_buffer = LiveExperienceBuffer(capacity=100)
    hist_buffer = HistoricalBuffer()

    # Fill buffers with sample transitions
    for i in range(50):
        t_live = Transition(
            state=np.random.randn(7),
            action=np.random.choice([0, 1, 2]),
            reward=0.1,
            next_state=np.random.randn(7),
            done=False,
        )
        live_buffer.add(t_live)

        t_hist = Transition(
            state=np.random.randn(7),
            action=np.random.choice([0, 1, 2]),
            reward=-0.05,
            next_state=np.random.randn(7),
            done=False,
        )
        hist_buffer.add(t_hist)

    assert len(live_buffer) == 50
    assert len(hist_buffer) == 50

    sampler = MixedBatchSampler(live_buffer, hist_buffer, live_ratio=0.70)
    batch = sampler.sample_batch(batch_size=32)

    assert batch["states"].shape == (32, 7)
    assert batch["actions"].shape == (32,)
    assert batch["rewards"].shape == (32,)


def test_ppo_agent_save_load():
    agent = PPOAgent(state_dim=7, action_dim=3, version_tag="v1.0_save_test")
    state = np.zeros(7, dtype=np.float32)
    act_orig, _, _ = agent.select_action(state, deterministic=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model_test.pt")
        agent.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        loaded_agent = PPOAgent(state_dim=7, action_dim=3, version_tag="v_loaded")
        loaded_agent.load(ckpt_path)

        act_loaded, _, _ = loaded_agent.select_action(state, deterministic=True)
        assert act_orig == act_loaded
        assert loaded_agent.version_tag == "v1.0_save_test"
