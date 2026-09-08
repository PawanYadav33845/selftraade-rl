"""
Synthetic Market Data Generator for Crypto, Forex, and Commodities (Gold, Silver, FX Pairs).
Simulates market regimes including bullish trends, bearish trends, high volatility, and sideways consolidation.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


SYMBOL_PRESETS: Dict[str, Dict[str, Any]] = {
    "XAUUSD": {"initial_price": 2500.0, "base_volatility": 0.008, "asset_type": "Gold Spot / USD"},
    "XAGUSD": {"initial_price": 30.0, "base_volatility": 0.014, "asset_type": "Silver Spot / USD"},
    "GBPUSD": {"initial_price": 1.30, "base_volatility": 0.004, "asset_type": "GBP / USD Forex"},
    "EURUSD": {"initial_price": 1.08, "base_volatility": 0.0035, "asset_type": "EUR / USD Forex"},
    "BTCUSD": {"initial_price": 60000.0, "base_volatility": 0.018, "asset_type": "Bitcoin / USD Crypto"},
    "BTC/USDT": {"initial_price": 50000.0, "base_volatility": 0.015, "asset_type": "BTC / USDT Crypto"},
}


class MarketDataGenerator:
    """
    Generates realistic OHLCV price series for testing, simulation, and historical buffer initialization.
    Supports multi-symbol assets: XAUUSD (Gold), XAGUSD (Silver), GBPUSD, EURUSD, BTCUSD, BTC/USDT.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_symbol_data(
        self,
        symbol: str = "XAUUSD",
        num_steps: int = 1000,
        regime_shift_freq: int = 250,
    ) -> pd.DataFrame:
        """
        Convenience method to generate OHLCV data for a specific asset symbol preset.
        Supported symbols: XAUUSD, XAGUSD, GBPUSD, EURUSD, BTCUSD, BTC/USDT.
        """
        symbol_upper = symbol.upper().replace("-", "/").replace("_", "/")
        preset = SYMBOL_PRESETS.get(
            symbol_upper,
            {"initial_price": 100.0, "base_volatility": 0.01, "asset_type": "Custom Asset"},
        )

        return self.generate_ohlcv(
            num_steps=num_steps,
            initial_price=preset["initial_price"],
            base_volatility=preset["base_volatility"],
            regime_shift_freq=regime_shift_freq,
            symbol=symbol_upper,
        )

    def generate_ohlcv(
        self,
        num_steps: int = 2000,
        initial_price: float = 100.0,
        base_volatility: float = 0.015,
        regime_shift_freq: int = 400,
        symbol: str = "BTC/USDT",
    ) -> pd.DataFrame:
        """
        Generates synthetic OHLCV market data series with regime shifts.
        """
        prices = np.zeros(num_steps, dtype=np.float64)
        prices[0] = initial_price

        volumes = np.zeros(num_steps, dtype=np.float64)
        volumes[0] = 1000.0

        current_drift = 0.0005  # Slight upward drift initially
        current_vol = base_volatility

        for i in range(1, num_steps):
            # Regime shift logic
            if i % regime_shift_freq == 0:
                regime_type = self.rng.choice(["bullish", "bearish", "high_vol", "sideways"])
                if regime_type == "bullish":
                    current_drift = self.rng.uniform(0.001, 0.003)
                    current_vol = base_volatility * 0.8
                elif regime_type == "bearish":
                    current_drift = self.rng.uniform(-0.003, -0.001)
                    current_vol = base_volatility * 1.5
                elif regime_type == "high_vol":
                    current_drift = self.rng.uniform(-0.001, 0.001)
                    current_vol = base_volatility * 3.0
                else:  # sideways
                    current_drift = 0.0
                    current_vol = base_volatility * 0.6

            # Return generation with t-distribution noise for heavy tails
            shock = self.rng.standard_t(df=5) * current_vol
            ret = current_drift + shock
            prices[i] = max(prices[i - 1] * (1.0 + ret), 0.0001)

            # Volume simulation correlated with volatility and return magnitude
            vol_noise = self.rng.lognormal(mean=0.0, sigma=0.3)
            volumes[i] = max(100.0, volumes[i - 1] * (1.0 + 5.0 * abs(ret)) * vol_noise)

        # Generate High/Low/Open from Close
        highs = prices * (1.0 + self.rng.uniform(0.001, 0.01, size=num_steps))
        lows = prices * (1.0 - self.rng.uniform(0.001, 0.01, size=num_steps))
        opens = np.roll(prices, 1)
        opens[0] = initial_price

        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        highs = np.maximum(highs, np.maximum(opens, prices))
        lows = np.minimum(lows, np.minimum(opens, prices))

        start_time = pd.Timestamp("2026-01-01 00:00:00")
        timestamps = [start_time + pd.Timedelta(minutes=5 * i) for i in range(num_steps)]

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
                "symbol": symbol,
            }
        )
        return df
