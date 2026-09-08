"""
Strategy Backtesting Engine.
Executes historical strategy backtests and generates comprehensive quantitative performance reports.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

from selftrade.env.trading_env import TradingEnv
from selftrade.agent.ppo_agent import PPOAgent
from selftrade.analysis.pattern_engine import PatternTrendEngine
from selftrade.reasoning.trade_reasoner import TradeReasoner
from selftrade.risk.risk_engine import DeterministicRiskEngine


@dataclass
class BacktestReport:
    """Dataclass holding complete backtest analytics."""
    symbol: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    expectancy: float

    def print_summary() -> None:
        print("\n" + "=" * 80)
        print(f"            HISTORICAL BACKTEST PERFORMANCE REPORT: {self.symbol}            ")
        print("=" * 80)
        print(f" Initial Capital      : ${self.initial_capital:,.2f}")
        print(f" Final Capital        : ${self.final_capital:,.2f}")
        print(f" Total Net Return (%) : {self.total_return_pct:+.2f}%")
        print(f" Win Rate (%)         : {self.win_rate_pct:.1f}% ({self.winning_trades} Wins / {self.losing_trades} Losses)")
        print(f" Profit Factor        : {self.profit_factor:.2f}")
        print(f" Max Drawdown (%)     : {self.max_drawdown_pct:.2f}%")
        print(f" Sharpe Ratio         : {self.sharpe_ratio:.2f}")
        print(f" Trade Expectancy     : ${self.expectancy:+.2f} per trade")
        print("=" * 80 + "\n")


class StrategyBacktester:
    """
    Backtesting Engine evaluating historical performance of SelfTrade-RL.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        pattern_engine: PatternTrendEngine = None,
        trade_reasoner: TradeReasoner = None,
        risk_engine: DeterministicRiskEngine = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.pattern_engine = pattern_engine or PatternTrendEngine()
        self.trade_reasoner = trade_reasoner or TradeReasoner()
        self.risk_engine = risk_engine or DeterministicRiskEngine()

    def run_backtest(self, df: pd.DataFrame, agent: PPOAgent) -> BacktestReport:
        """
        Runs strategy backtest over historical DataFrame using Agent, Pattern Engine, Reasoner, and Risk Engine.
        """
        symbol = str(df.get("symbol", pd.Series(["ASSET"])).iloc[0])
        env = TradingEnv(df=df, initial_capital=self.initial_capital)

        obs, info = env.reset()
        done = False
        step = 0

        trade_pnls: List[float] = []

        while not done:
            current_df = env.df.iloc[: env.current_step + 1]

            # 1. Real-time Chart Analysis (Pattern & Trend)
            market_struct = self.pattern_engine.analyze_chart(current_df)

            # 2. Agent Action Prediction
            action_net, _, _ = agent.select_action(obs, deterministic=True)

            # 3. Multi-Factor Reasoner Evaluation ("Think before taking a trade")
            close_price = info["close_price"]
            atr = float(obs[4]) * close_price
            reasoned_decision = self.trade_reasoner.evaluate_trade(
                proposed_action=action_net,
                current_price=close_price,
                atr=atr,
                market_structure=market_struct,
            )

            # 4. Hard Risk Engine Check
            risk_verdict = self.risk_engine.evaluate_action(
                proposed_action=reasoned_decision.approved_action,
                env_info=info,
            )

            safe_action = risk_verdict.safe_action

            # 5. Environment Step
            next_obs, reward, terminated, truncated, info = env.step(safe_action)
            done = terminated or truncated

            obs = next_obs
            step += 1

        # Calculate Trades Metrics
        final_capital = info["portfolio_value"]
        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100.0

        # Extract PnLs from trades history
        pnl_list = []
        buy_price = 0.0
        for t in env.trades_history:
            if t["type"] == "BUY":
                buy_price = t["price"]
            elif t["type"] == "SELL" and buy_price > 0.0:
                sell_pnl = (t["price"] - buy_price) * t["units"]
                pnl_list.append(sell_pnl)
                buy_price = 0.0

        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]

        total_trades = len(pnl_list)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-8
        profit_factor = float(gross_profit / gross_loss)

        expectancy = float(np.mean(pnl_list)) if pnl_list else 0.0

        # Equity Curve Drawdown & Sharpe Ratio
        eq_curve = np.array(env.portfolio_history)
        peak = np.maximum.accumulate(eq_curve)
        dd = (peak - eq_curve) / peak
        max_dd_pct = float(np.max(dd)) * 100.0 if len(dd) > 0 else 0.0

        returns = np.diff(eq_curve) / eq_curve[:-1]
        std_ret = np.std(returns) + 1e-8
        sharpe_ratio = float((np.mean(returns) / std_ret) * np.sqrt(252.0)) if len(returns) > 5 else 0.0

        report = BacktestReport(
            symbol=symbol,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            expectancy=expectancy,
        )

        return report
