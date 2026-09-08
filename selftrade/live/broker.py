"""
Live Exchange Adapter Interface for Connecting SelfTrade-RL to Real Trading Exchanges.
Supports REST/WebSocket order placement, account balance fetching, and market data streaming.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class BaseBrokerAdapter(ABC):
    """
    Abstract Interface for Live Exchange / Broker Connectors.
    Implements standard methods required by SelfTrade-RL to interface with live platforms.
    """

    @abstractmethod
    def fetch_latest_candles(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> pd.DataFrame:
        """Fetch real-time OHLCV candles from exchange REST API."""
        pass

    @abstractmethod
    def fetch_account_balance(self) -> Tuple[float, float, float]:
        """
        Fetch real-time account balances.
        Returns: (total_portfolio_value, available_cash, holdings_value)
        """
        pass

    @abstractmethod
    def execute_order(self, symbol: str, action: int, amount: float, current_price: float) -> Dict[str, Any]:
        """
        Executes safe action on live exchange (1: BUY, 2: SELL, 0: HOLD).
        """
        pass


class PaperBrokerAdapter(BaseBrokerAdapter):
    """
    Simulated Paper Broker Adapter for live streaming test environment.
    """

    def __init__(self, initial_capital: float = 10000.0) -> None:
        self.cash = initial_capital
        self.position_units = 0.0
        self.entry_price = 0.0

    def fetch_latest_candles(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> pd.DataFrame:
        # Placeholder returns simulated dataframe
        pass

    def fetch_account_balance(self) -> Tuple[float, float, float]:
        holdings_val = self.position_units * self.entry_price
        total_val = self.cash + holdings_val
        return total_val, self.cash, holdings_val

    def execute_order(self, symbol: str, action: int, amount: float, current_price: float) -> Dict[str, Any]:
        if action == 1:  # BUY
            if self.cash > 0:
                units = (self.cash * 0.999) / current_price
                self.position_units += units
                self.entry_price = current_price
                self.cash = 0.0
                return {"status": "FILLED", "side": "BUY", "units": units, "price": current_price}
        elif action == 2:  # SELL
            if self.position_units > 0:
                proceeds = self.position_units * current_price * 0.999
                self.cash += proceeds
                sold_units = self.position_units
                self.position_units = 0.0
                self.entry_price = 0.0
                return {"status": "FILLED", "side": "SELL", "units": sold_units, "price": current_price}

        return {"status": "HOLD", "side": "NONE", "units": 0.0, "price": current_price}
