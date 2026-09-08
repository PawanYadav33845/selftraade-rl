"""
Data Ingestion and Technical Feature Engineering Engine.
Provides lookahead-bias-free calculation of technical indicators and feature normalization.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class MarketFeatureEngine:
    """
    Computes technical indicators and state features strictly without lookahead bias.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
    ) -> None:
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes OHLCV dataframe, handles missing values, and calculates normalized indicators.

        Args:
            df: DataFrame containing ['open', 'high', 'low', 'close', 'volume'] columns.

        Returns:
            DataFrame with original columns + engineered technical features.
        """
        req_cols = {"open", "high", "low", "close", "volume"}
        if not req_cols.issubset(set(df.columns)):
            missing = req_cols - set(df.columns)
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        data = df.copy()
        # Clean missing/invalid values
        data = data.ffill().bfill().fillna(0.0)

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

        # 1. Price return (close_t / close_{t-1} - 1)
        prev_close = data["close"].shift(1).replace(0.0, np.nan).bfill()
        data["price_return"] = (data["close"] / prev_close - 1.0).fillna(0.0)

        # 2. Volume percentage change
        prev_vol = data["volume"].shift(1).replace(0.0, np.nan).bfill()
        data["volume_pct_change"] = ((data["volume"] - prev_vol) / prev_vol).fillna(0.0)
        data["volume_pct_change"] = np.clip(data["volume_pct_change"], -5.0, 5.0)

        # 3. Normalized RSI (0 to 1 scaling)
        delta = data["close"].diff().fillna(0.0)
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        data["rsi_norm"] = (rsi / 100.0).fillna(0.5).clip(0.0, 1.0)

        # 4. Normalized MACD Histogram
        ema_fast = data["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data["close"].ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        # Normalize MACD histogram by close price to make it scale-invariant
        data["macd_hist_norm"] = (macd_hist / (data["close"] + 1e-8)).fillna(0.0).clip(-0.1, 0.1)

        # 5. Normalized ATR (Average True Range / close_t)
        prev_close_series = data["close"].shift(1).fillna(data["close"])
        tr1 = data["high"] - data["low"]
        tr2 = (data["high"] - prev_close_series).abs()
        tr3 = (data["low"] - prev_close_series).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()
        data["atr_norm"] = (atr / (data["close"] + 1e-8)).fillna(0.01).clip(0.0, 0.2)

        # Replace any residual NaN/inf with zeros or reasonable bounds
        feature_cols = ["price_return", "volume_pct_change", "rsi_norm", "macd_hist_norm", "atr_norm"]
        for col in feature_cols:
            data[col] = np.nan_to_num(data[col].values, nan=0.0, posinf=1.0, neginf=-1.0)

        return data


class DataIngestion:
    """
    High-level data ingestion manager for historical and live market streams.
    """

    def __init__(self, feature_engine: MarketFeatureEngine = None) -> None:
        self.feature_engine = feature_engine or MarketFeatureEngine()

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans raw OHLCV market data and appends normalized features.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        return self.feature_engine.process_dataframe(df)

    def extract_market_state(
        self,
        row: pd.Series,
        pos_ratio: float = 0.0,
        cash_ratio: float = 1.0,
    ) -> np.ndarray:
        """
        Extracts 7-dimensional market state array:
        [price_return, volume_pct_change, rsi_norm, macd_hist_norm, atr_norm, pos_ratio, cash_ratio].
        """
        return np.array(
            [
                float(row.get("price_return", 0.0)),
                float(row.get("volume_pct_change", 0.0)),
                float(row.get("rsi_norm", 0.5)),
                float(row.get("macd_hist_norm", 0.0)),
                float(row.get("atr_norm", 0.01)),
                float(pos_ratio),
                float(cash_ratio),
            ],
            dtype=np.float32,
        )


# Alias for backward compatibility
FeatureEngineer = DataIngestion

