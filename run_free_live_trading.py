"""
100% FREE Live Paper Trading & Real-Time Training Runner.
Connects directly to free live market feeds (Gold, Silver, Forex, Crypto), trades in paper mode, trains the RL agent online, enforces 1:2.0 R:R risk gates, and logs trades to the Trade Journal.
No TradingView subscription or paid API key needed!
"""

from typing import Dict, Any
import logging
import time
import sys
import pandas as pd

from selftrade.live.free_live_streamer import FreeLiveStreamer
from selftrade.data.ingestion import DataIngestion
from selftrade.env.trading_env import TradingEnv
from selftrade.agent.ppo_agent import PPOAgent
from selftrade.agent.buffer import LiveExperienceBuffer, HistoricalBuffer
from selftrade.analysis.pattern_engine import PatternTrendEngine
from selftrade.reasoning.trade_reasoner import TradeReasoner
from selftrade.risk.risk_engine import DeterministicRiskEngine
from selftrade.continuous_learning.monitor import PerformanceMonitor
from selftrade.continuous_learning.self_improver import SelfImprovingEngine
from selftrade.shadow.shadow_engine import ShadowValidationEngine
from selftrade.journal.trade_journal import TradeJournal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("FreeLiveTrader")


def run_free_live_trader(symbol: str = "XAUUSD", duration_minutes: int = 10) -> None:
    print("=" * 85)
    print(f"      SELFTRADE-RL: 100% FREE LIVE PAPER TRADER & ONLINE TRAINER ({symbol})      ")
    print("=" * 85)

    ingestion = DataIngestion()
    pattern_engine = PatternTrendEngine()
    reasoner = TradeReasoner(target_risk_reward=2.0)
    risk_engine = DeterministicRiskEngine()
    agent = PPOAgent(version_tag="v1.0_free_live")
    journal = TradeJournal(journal_filepath="journal/trade_journal.json")

    live_buffer = LiveExperienceBuffer(capacity=5000)
    hist_buffer = HistoricalBuffer()
    monitor = PerformanceMonitor(periodic_interval=50)

    improver = SelfImprovingEngine(agent, live_buffer, hist_buffer, monitor)

    logger.info(f"Connecting to FREE Live Stream for {symbol}...")
    df_live = FreeLiveStreamer.fetch_live_data(symbol=symbol, interval="1m", period="1d")
    logger.info(f"Successfully fetched {len(df_live)} live market candles!")

    env = TradingEnv(df=df_live, initial_capital=10000.0)
    obs, info = env.reset()

    # Pre-seed historical buffer
    improver.historical_buffer.add_bulk([])

    start_time = time.time()
    step = 0
    trade_counter = 1
    last_buy_entry = 0.0

    print("\n" + "=" * 85)
    print("                      LIVE PAPER TRADING TICK EXECUTION                        ")
    print("=" * 85)

    while (time.time() - start_time) < (duration_minutes * 60) and step < len(df_live) - 1:
        current_price = info["close_price"]
        current_df = env.df.iloc[: env.current_step + 1]

        # 1. Real-time Chart Analysis
        market_struct = pattern_engine.analyze_chart(current_df)

        # 2. Agent Action
        action_net, log_prob, val = agent.select_action(obs, deterministic=False)

        # 3. Multi-factor Trade Reasoner ("Think before trading")
        atr = float(obs[4]) * current_price
        decision = reasoner.evaluate_trade(action_net, current_price, atr, market_struct)

        # 4. Hard Risk Engine Check
        risk_verdict = risk_engine.evaluate_action(decision.approved_action, info)
        safe_action = risk_verdict.safe_action

        # 5. Environment Step
        next_obs, reward, terminated, truncated, info = env.step(safe_action)
        done = terminated or truncated

        # Log transition for online RL training
        improver.record_transition(obs, safe_action, reward, next_obs, done, log_prob, val)

        # Check continuous learning triggers
        triggered, candidate_agent, trigger_desc = improver.check_and_fine_tune(reward, float(obs[4]))
        if triggered and candidate_agent is not None:
            logger.info(f"Step {step}: {trigger_desc}")
            agent = candidate_agent

        # Journal Logging on Completed Trade
        if safe_action == 1:
            last_buy_entry = current_price
            logger.info(f"Step {step} | EXECUTED BUY at ${current_price:,.2f} | TP: ${decision.take_profit_price:,.2f} | SL: ${decision.stop_loss_price:,.2f} (1:2.0 R:R)")

        elif safe_action == 2 and last_buy_entry > 0.0:
            pnl_usd = (current_price - last_buy_entry) * (10000.0 / last_buy_entry)
            pnl_pct = ((current_price - last_buy_entry) / last_buy_entry) * 100.0

            journal.log_trade(
                trade_id=f"TRD-{trade_counter:04d}",
                timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol=symbol,
                side="BUY",
                entry_price=last_buy_entry,
                exit_price=current_price,
                units=10000.0 / last_buy_entry,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                take_profit=decision.take_profit_price,
                stop_loss=decision.stop_loss_price,
                risk_reward_ratio=2.0,
                confluence_score=decision.confluence_score,
                trend_condition=market_struct.trend.value,
                candlestick_pattern=market_struct.candlestick_pattern.value,
                reasoner_summary=decision.reasoning_summary,
                risk_verdict=risk_verdict.override_reason,
                model_version=agent.version_tag,
            )

            logger.info(f"Step {step} | EXECUTED SELL at ${current_price:,.2f} | Realized PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)")
            trade_counter += 1
            last_buy_entry = 0.0

        if risk_verdict.is_overridden and step % 20 == 0:
            logger.warning(f"Step {step}: Risk Override -> {risk_verdict.override_reason}")

        obs = next_obs
        step += 1
        time.sleep(0.02)  # Tick speed simulation

        if done:
            break

    print("\n" + "=" * 85)
    print("                       FREE LIVE TRADING SESSION SUMMARY                        ")
    print("=" * 85)
    print(f" Symbol Executed       : {symbol}")
    print(f" Final Portfolio Value : ${info['portfolio_value']:,.2f}")
    print(f" Net Session PnL (%)   : {((info['portfolio_value'] - 10000.0)/10000.0)*100.0:+.2f}%")
    print(f" Total Trades Logged   : {len(journal.entries)}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_free_live_trader("XAUUSD", duration_minutes=1)
