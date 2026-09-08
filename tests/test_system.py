"""
Integration Tests for Complete SelfTradeSystem Orchestration.
"""

import pytest
import numpy as np

from selftrade.data.generator import MarketDataGenerator
from selftrade.system import SelfTradeSystem


def test_selftrade_system_end_to_end():
    gen = MarketDataGenerator(seed=42)
    hist_df = gen.generate_ohlcv(num_steps=100, initial_price=100.0, symbol="BTC/USDT")
    live_df = gen.generate_ohlcv(num_steps=150, initial_price=100.0, symbol="BTC/USDT")

    system = SelfTradeSystem(
        df=live_df,
        initial_capital=10000.0,
        periodic_interval=30,  # Short interval to force trigger during simulation
        eval_test_window=20,
    )

    # 1. Seed historical buffer
    seeded_count = system.seed_historical_buffer(hist_df)
    assert seeded_count > 0
    assert len(system.historical_buffer) == seeded_count

    # 2. Run simulation loop
    metrics = system.run_simulation(max_steps=100)
    assert len(metrics) > 0
    assert "portfolio_value" in metrics[0]
    assert "safe_action" in metrics[0]
    assert "active_model_version" in metrics[0]
