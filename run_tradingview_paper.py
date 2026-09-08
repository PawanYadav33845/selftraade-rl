"""
TradingView Paper Trading & Online RL Training Runner.
Starts the SelfTrade-RL Webhook Server to receive live candle data from TradingView Paper Trading, train policy weights online, enforce 1:2.0 R:R risk gates, and log trades to the Trade Journal.
"""

from typing import Dict, Any
import logging
import sys

from selftrade.data.generator import MarketDataGenerator
from selftrade.system import SelfTradeSystem
from selftrade.live.tradingview_bridge import TradingViewBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingViewPaperRunner")


def main():
    print("=" * 80)
    print("      SELFTRADE-RL: TRADINGVIEW PAPER TRADING & ONLINE TRAINING SERVER      ")
    print("=" * 80)

    # Initialize baseline system orchestrator with historical warmup data
    data_gen = MarketDataGenerator(seed=42)
    warmup_df = data_gen.generate_ohlcv(num_steps=500, initial_price=2500.0, symbol="XAUUSD")

    system = SelfTradeSystem(
        df=warmup_df,
        initial_capital=10000.0,
        max_stop_loss_pct=0.02,
        max_daily_drawdown_pct=0.05,
        sharpe_threshold=0.5,
        periodic_interval=100,
        eval_test_window=50,
    )

    # Seed historical dataset for 70/30 mixed batch training
    system.seed_historical_buffer(warmup_df)

    # Start Webhook Bridge Listener on http://localhost:5000/tradingview-webhook
    bridge = TradingViewBridge(system=system, host="0.0.0.0", port=5000)

    logger.info("\nStarting TradingView Paper Trading Webhook Listener...")
    logger.info("Listening on http://localhost:5000/tradingview-webhook")
    logger.info("Connect your TradingView Paper Trading chart alerts to this URL to train online!\n")

    bridge.run()


if __name__ == "__main__":
    main()
