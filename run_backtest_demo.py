"""
Interactive Chart Reasoning & Strategy Backtest Demo.
Demonstrates pattern analysis, multi-factor reasoning ("thinking before trading"), safe TP/SL placement (1:2.5 R:R), and backtesting performance reporting.
"""

import sys
import logging

from selftrade.data.generator import MarketDataGenerator
from selftrade.agent.ppo_agent import PPOAgent
from selftrade.analysis.pattern_engine import PatternTrendEngine
from selftrade.reasoning.trade_reasoner import TradeReasoner
from selftrade.backtest.backtester import StrategyBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestDemo")


def main():
    print("=" * 85)
    print("      SELFTRADE-RL: ADVANCED CHART REASONER & STRATEGY BACKTESTING DEMO      ")
    print("=" * 85)

    data_gen = MarketDataGenerator(seed=42)

    # 1. Backtest across symbols
    symbols = ["XAUUSD", "XAGUSD", "GBPUSD", "EURUSD", "BTCUSD"]
    agent = PPOAgent(version_tag="v1.0_trained")
    backtester = StrategyBacktester(initial_capital=10000.0)

    print("\n" + "=" * 85)
    print("                      STRATEGY BACKTEST PERFORMANCE TABLE                       ")
    print("=" * 85)
    print(f"{'SYMBOL':<10} | {'NET RETURN':<11} | {'WIN RATE (%)':<13} | {'PROFIT FACTOR':<14} | {'MAX DRAWDOWN':<14} | {'EXPECTANCY':<12}")
    print("-" * 85)

    for sym in symbols:
        df = data_gen.generate_symbol_data(symbol=sym, num_steps=600)
        report = backtester.run_backtest(df, agent)

        print(
            f"{report.symbol:<10} | {report.total_return_pct:>+9.2f}% | "
            f"{report.win_rate_pct:>11.1f}% | {report.profit_factor:>12.2f} | "
            f"{report.max_drawdown_pct:>12.2f}% | ${report.expectancy:>+10.2f}"
        )

    print("=" * 85)

    # 2. Demonstrate Real-Time Chart Reading & "Thinking Before Trading"
    logger.info("\nDemonstrating Real-Time Chart Reading & Cognitive Reasoning for XAUUSD (Gold)...")
    gold_df = data_gen.generate_symbol_data(symbol="XAUUSD", num_steps=150)

    pattern_engine = PatternTrendEngine()
    reasoner = TradeReasoner(min_confluence_threshold=0.60, target_risk_reward=2.0)

    market_struct = pattern_engine.analyze_chart(gold_df)
    current_price = float(gold_df["close"].iloc[-1])
    atr = current_price * 0.01

    print("\n" + "-" * 85)
    print("                        REAL-TIME CHART ANALYSIS REPORT                        ")
    print("-" * 85)
    print(f" Asset Symbol              : XAUUSD (Gold Spot)")
    print(f" Current Market Price      : ${current_price:,.2f}")
    print(f" Market Trend              : {market_struct.trend.value}")
    print(f" Candlestick Pattern       : {market_struct.candlestick_pattern.value}")
    print(f" Support Level             : ${market_struct.support_level:,.2f}")
    print(f" Resistance Level          : ${market_struct.resistance_level:,.2f}")
    print(f" Fast EMA (20) / Slow (50) : ${market_struct.ema_fast:,.2f} / ${market_struct.ema_slow:,.2f}")
    print("-" * 85)

    # Evaluate Trade Decision for proposed BUY action
    proposed_action = 1  # BUY
    decision = reasoner.evaluate_trade(
        proposed_action=proposed_action,
        current_price=current_price,
        atr=atr,
        market_structure=market_struct,
    )

    print("                     COGNITIVE TRADE REASONING & TP/SL GEOMETRY                 ")
    print("-" * 85)
    print(f" Proposed Action           : BUY (1)")
    print(f" Approved Safe Action      : {'BUY (1)' if decision.approved_action == 1 else 'HOLD (0)'}")
    print(f" Confluence Score          : {decision.confluence_score:.2f} / 1.00")
    print(f" Dynamic Stop-Loss (SL)    : ${decision.stop_loss_price:,.2f} (Risk: ${abs(current_price - decision.stop_loss_price):.2f})")
    print(f" Dynamic Take-Profit (TP)  : ${decision.take_profit_price:,.2f} (Reward: ${abs(decision.take_profit_price - current_price):.2f})")
    print(f" Target Risk:Reward Ratio  : 1:{decision.risk_reward_ratio:.1f}")
    print(f" Reasoning Summary         : {decision.reasoning_summary}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
