"""
Unit Tests for OpenAI Gymnasium Trading Environment (TradingEnv).
"""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym

from selftrade.data.generator import MarketDataGenerator
from selftrade.env.trading_env import TradingEnv


@pytest.fixture
def sample_market_data():
    gen = MarketDataGenerator(seed=42)
    return gen.generate_ohlcv(num_steps=150, initial_price=100.0)


def test_trading_env_initialization(sample_market_data):
    env = TradingEnv(df=sample_market_data, initial_capital=10000.0)
    assert env.observation_space.shape == (7,)
    assert env.action_space.n == 3

    obs, info = env.reset()
    assert obs.shape == (7,)
    assert obs.dtype == np.float32
    assert info["portfolio_value"] == 10000.0
    assert info["cash"] == 10000.0
    assert info["position"] == 0.0


def test_trading_env_step_buy_sell(sample_market_data):
    env = TradingEnv(df=sample_market_data, initial_capital=10000.0)
    obs, info = env.reset()

    # Step 1: Execute BUY action (1)
    obs, reward, terminated, truncated, info = env.step(1)
    assert info["position"] > 0.0
    assert info["cash"] == 0.0
    assert isinstance(reward, float)

    # Step 2: Execute HOLD action (0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["position"] > 0.0

    # Step 3: Execute SELL action (2)
    obs, reward, terminated, truncated, info = env.step(2)
    assert info["position"] == 0.0
    assert info["cash"] > 0.0


def test_trading_env_hard_drawdown_termination():
    gen = MarketDataGenerator(seed=42)
    df = gen.generate_ohlcv(num_steps=100, initial_price=100.0)

    # Simulate catastrophic price crash in dataframe
    df.loc[30:, "close"] = 10.0  # 90% price crash

    env = TradingEnv(df=df, initial_capital=10000.0, max_drawdown_limit=0.50)
    obs, info = env.reset()

    # Buy position before crash
    for step in range(30):
        obs, reward, terminated, truncated, info = env.step(1 if step == 0 else 0)
        if terminated:
            break

    # After step 30, portfolio value will drop by > 50%, triggering hard termination
    assert terminated is True
    assert info["portfolio_value"] <= 5000.0
