"""
Custom OpenAI Gymnasium Trading Environment for SelfTrade-RL.
Tracks market state, portfolio values, cash/position ratios, execution fees, and log-return rewards.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from selftrade.data.ingestion import DataIngestion, MarketFeatureEngine


class TradingEnv(gym.Env):
    """
    OpenAI Gymnasium Environment for RL Trading Agent.

    State Space (S_t): Vector of size N=7:
        0: Price return (close_t / close_{t-1} - 1)
        1: Volume percentage change
        2: Normalized RSI (0 to 1)
        3: Normalized MACD histogram
        4: Normalized ATR
        5: Position ratio (holdings_value / total_portfolio_value)
        6: Cash ratio (unallocated_cash / total_portfolio_value)

    Action Space (A_t): Discrete(3) mapping to:
        0: Hold
        1: Buy
        2: Sell

    Reward Function (R_t):
        R_t = ln(W_t / (W_{t-1} + eps)) - lambda * C_trans
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        transaction_fee: float = 0.001,  # 0.1% maker/taker fee
        slippage: float = 0.0005,        # 0.05% execution slippage
        cost_penalty_lambda: float = 1.0,
        max_drawdown_limit: float = 0.50, # 50% termination cap
        feature_engine: Optional[MarketFeatureEngine] = None,
    ) -> None:
        super().__init__()

        self.initial_capital = float(initial_capital)
        self.transaction_fee = float(transaction_fee)
        self.slippage = float(slippage)
        self.cost_penalty_lambda = float(cost_penalty_lambda)
        self.max_drawdown_limit = float(max_drawdown_limit)

        # Prepare market dataset with technical indicators
        self.ingestion = DataIngestion(feature_engine or MarketFeatureEngine())
        self.df = self.ingestion.prepare_data(df)
        self.num_steps = len(self.df)

        if self.num_steps < 20:
            raise ValueError(f"Insufficient data rows ({self.num_steps}). At least 20 rows required.")

        # Define Observation & Action Spaces according to Gymnasium standard
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(7,),
            dtype=np.float32,
        )

        # Environment State Variables
        self.current_step = 0
        self.cash = self.initial_capital
        self.position = 0.0  # units held
        self.position_entry_price = 0.0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital

        # History tracking
        self.trades_history = []
        self.portfolio_history = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_capital
        self.position = 0.0
        self.position_entry_price = 0.0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital

        self.trades_history.clear()
        self.portfolio_history = [self.initial_capital]

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= self.num_steps - 1:
            # End of dataset reached
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info

        current_row = self.df.iloc[self.current_step]
        current_price = float(current_row["close"])

        # Execute Action
        c_trans_ratio = 0.0
        executed_trade = False

        if action == 1:  # BUY
            if self.cash > 1e-4:
                # Buy asset with available cash accounting for slippage & fee
                buy_price = current_price * (1.0 + self.slippage)
                available_cash = self.cash * (1.0 - self.transaction_fee)
                units_bought = available_cash / buy_price

                self.position += units_bought
                self.position_entry_price = buy_price
                c_trans_ratio = (self.cash * self.transaction_fee + available_cash * self.slippage) / self.portfolio_value
                self.cash = 0.0
                executed_trade = True

                self.trades_history.append({
                    "step": self.current_step,
                    "type": "BUY",
                    "price": buy_price,
                    "units": units_bought,
                    "cost_ratio": c_trans_ratio,
                })

        elif action == 2:  # SELL
            if self.position > 1e-8:
                # Sell held asset units accounting for slippage & fee
                sell_price = current_price * (1.0 - self.slippage)
                gross_proceeds = self.position * sell_price
                net_proceeds = gross_proceeds * (1.0 - self.transaction_fee)
                c_trans_ratio = (gross_proceeds * self.transaction_fee + self.position * current_price * self.slippage) / self.portfolio_value

                self.cash += net_proceeds
                executed_trade = True

                self.trades_history.append({
                    "step": self.current_step,
                    "type": "SELL",
                    "price": sell_price,
                    "units": self.position,
                    "cost_ratio": c_trans_ratio,
                })

                self.position = 0.0
                self.position_entry_price = 0.0

        # Advance timestep
        self.current_step += 1
        next_row = self.df.iloc[self.current_step]
        next_price = float(next_row["close"])

        # Update Portfolio Net Worth W_t
        holdings_value = self.position * next_price
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = max(self.cash + holdings_value, 1e-8)
        self.portfolio_history.append(self.portfolio_value)

        # Logarithmic return reward
        eps = 1e-8
        log_return = np.log((self.portfolio_value + eps) / (self.prev_portfolio_value + eps))
        reward = float(log_return - self.cost_penalty_lambda * c_trans_ratio)

        # Check Termination / Truncation
        terminated = False
        truncated = False

        # Hard termination if portfolio value falls by >= 50%
        drawdown_from_initial = (self.initial_capital - self.portfolio_value) / self.initial_capital
        if drawdown_from_initial >= self.max_drawdown_limit:
            terminated = True
            reward -= 2.0  # Terminal failure penalty

        # End of dataset truncation
        if self.current_step >= self.num_steps - 1:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        current_row = self.df.iloc[self.current_step]
        current_price = float(current_row["close"])
        holdings_val = self.position * current_price
        total_val = max(self.cash + holdings_val, 1e-8)

        pos_ratio = float(np.clip(holdings_val / total_val, 0.0, 1.0))
        cash_ratio = float(np.clip(self.cash / total_val, 0.0, 1.0))

        return self.ingestion.extract_market_state(current_row, pos_ratio=pos_ratio, cash_ratio=cash_ratio)

    def _get_info(self) -> Dict[str, Any]:
        current_row = self.df.iloc[self.current_step]
        current_price = float(current_row["close"])
        holdings_val = self.position * current_price
        total_val = self.cash + holdings_val

        position_drawdown = 0.0
        if self.position > 0.0 and self.position_entry_price > 0.0:
            position_drawdown = (self.position_entry_price - current_price) / self.position_entry_price

        peak_val = max(self.portfolio_history) if self.portfolio_history else self.initial_capital
        portfolio_drawdown = (peak_val - total_val) / peak_val

        return {
            "step": self.current_step,
            "close_price": current_price,
            "cash": self.cash,
            "position": self.position,
            "position_entry_price": self.position_entry_price,
            "portfolio_value": total_val,
            "position_drawdown": position_drawdown,
            "portfolio_drawdown": portfolio_drawdown,
            "num_trades": len(self.trades_history),
        }
