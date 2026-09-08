"""
Unit Tests for Data Ingestion, Technical Feature Engineering, and Synthetic Market Data Generator.
"""

import pytest
import pandas as pd
import numpy as np

from selftrade.data.generator import MarketDataGenerator
from selftrade.data.ingestion import DataIngestion, MarketFeatureEngine


def test_market_data_generator():
    gen = MarketDataGenerator(seed=42)
    df = gen.generate_ohlcv(num_steps=100, initial_price=100.0, symbol="BTC/USDT")

    assert len(df) == 100
    assert set(["timestamp", "open", "high", "low", "close", "volume", "symbol"]).issubset(set(df.columns))
    assert (df["high"] >= df["low"]).all()
    assert (df["close"] > 0).all()
    assert not df.isnull().any().any()


def test_multi_symbol_presets():
    gen = MarketDataGenerator(seed=42)
    for symbol in ["XAUUSD", "XAGUSD", "GBPUSD", "EURUSD", "BTCUSD"]:
        df = gen.generate_symbol_data(symbol=symbol, num_steps=50)
        assert len(df) == 50
        assert df["symbol"].iloc[0] == symbol
        assert (df["close"] > 0).all()


def test_data_ingestion_and_feature_engineering():
    gen = MarketDataGenerator(seed=123)
    raw_df = gen.generate_ohlcv(num_steps=200)

    # Introduce some missing NaN values to verify robustness
    raw_df.loc[10:15, "close"] = np.nan
    raw_df.loc[20, "volume"] = np.nan

    ingestion = DataIngestion()
    processed_df = ingestion.prepare_data(raw_df)

    assert not processed_df.isnull().any().any()
    feature_cols = ["price_return", "volume_pct_change", "rsi_norm", "macd_hist_norm", "atr_norm"]
    for col in feature_cols:
        assert col in processed_df.columns
        assert not np.isinf(processed_df[col]).any()

    # Verify RSI normalized strictly in [0, 1]
    assert (processed_df["rsi_norm"] >= 0.0).all() and (processed_df["rsi_norm"] <= 1.0).all()

    # Verify extract_market_state returns 7-dim float32 observation state
    sample_row = processed_df.iloc[50]
    market_state = ingestion.extract_market_state(sample_row)
    assert isinstance(market_state, np.ndarray)
    assert market_state.shape == (7,)
    assert market_state.dtype == np.float32
