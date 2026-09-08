"""
Run Demo: End-to-End Simulation Script for SelfTrade-RL.
Demonstrates market data streaming, PPO trading execution, hard risk gating, online regime-shift fine-tuning, shadow deployment, and candidate promotion.
"""

import sys
import logging
import numpy as np
import pandas as pd

from selftrade.data.generator import MarketDataGenerator
from selftrade.system import SelfTradeSystem

# Setup pretty log output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RunDemo")


def main():
    print("=" * 80)
    print("      SELFTRADE-RL: AUTONOMOUS SELF-IMPROVING TRADING BOT DEMO      ")
    print("=" * 80)

    # 1. Synthesize Historical & Live Market Data Series
    data_gen = MarketDataGenerator(seed=42)

    logger.info("Generating Historical Market Data (1,000 candles)...")
    historical_df = data_gen.generate_ohlcv(
        num_steps=1000,
        initial_price=50000.0,  # BTC/USDT scale
        base_volatility=0.015,
        regime_shift_freq=250,
        symbol="BTC/USDT",
    )

    logger.info("Generating Live Market Streaming Data (1,200 candles with Regime Shifts)...")
    live_df = data_gen.generate_ohlcv(
        num_steps=1200,
        initial_price=52000.0,
        base_volatility=0.020,
        regime_shift_freq=300,
        symbol="BTC/USDT",
    )

    # 2. Instantiate SelfTrade-RL System
    initial_capital = 10000.0
    system = SelfTradeSystem(
        df=live_df,
        initial_capital=initial_capital,
        max_stop_loss_pct=0.02,        # 2.0% hard stop-loss limit
        max_daily_drawdown_pct=0.05,    # 5.0% daily circuit breaker
        sharpe_threshold=0.5,           # Sharpe drop trigger
        volatility_shift_threshold=0.25,# 25% regime volatility jump trigger
        periodic_interval=300,         # Fine-tune every 300 steps
        eval_test_window=100,           # Shadow validation window
    )

    # 3. Seed Historical Buffer (30% portion of mixed sampling)
    system.seed_historical_buffer(historical_df)

    # 4. Run System Execution Loop
    metrics = system.run_simulation(max_steps=1000)

    # 5. Output Summary Statistics & Performance Analysis
    final_portfolio_value = metrics[-1]["portfolio_value"] if metrics else initial_capital
    net_return_pct = ((final_portfolio_value - initial_capital) / initial_capital) * 100.0
    total_steps_executed = len(metrics)

    risk_overrides = sum(1 for m in metrics if m["risk_overridden"])
    circuit_breaker_steps = sum(1 for m in metrics if m["circuit_breaker_active"])
    buys = sum(1 for m in metrics if m["safe_action"] == 1)
    sells = sum(1 for m in metrics if m["safe_action"] == 2)
    holds = sum(1 for m in metrics if m["safe_action"] == 0)

    final_model_version = metrics[-1]["active_model_version"] if metrics else "v1.0_active"

    print("\n" + "=" * 80)
    print("                        SIMULATION RESULTS SUMMARY                      ")
    print("=" * 80)
    print(f" Initial Portfolio Value  : ${initial_capital:,.2f}")
    print(f" Final Portfolio Value    : ${final_portfolio_value:,.2f}")
    print(f" Net Portfolio Return (%) : {net_return_pct:+.2f}%")
    print(f" Total Steps Executed     : {total_steps_executed}")
    print("-" * 80)
    print(" ACTION BREAKDOWN:")
    print(f"   - Buy Actions (1)      : {buys}")
    print(f"   - Sell Actions (2)     : {sells}")
    print(f"   - Hold Actions (0)     : {holds}")
    print("-" * 80)
    print(" RISK & SAFETY ENGINE METRICS:")
    print(f"   - Hard Risk Overrides  : {risk_overrides}")
    print(f"   - Circuit Breaker Steps: {circuit_breaker_steps}")
    print("-" * 80)
    print(" CONTINUOUS LEARNING & SHADOW DEPLOYMENT:")
    print(f"   - Final Model Weight Tag: {final_model_version}")
    print(f"   - Replay Buffer Count  : {len(system.live_buffer)}")
    print(f"   - Historical Buffer    : {len(system.historical_buffer)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
