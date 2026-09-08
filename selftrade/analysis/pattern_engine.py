"""
Pattern Recognition, Market Structure, Historical Trend Memory Engine.
Analyzes current and historical price action to identify previous trends, trend continuations, trend reversals, candlestick patterns, and support/resistance levels.
"""

from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd


class TrendType(Enum):
    BULLISH_UPTREND = "BULLISH_UPTREND"
    BEARISH_DOWNTREND = "BEARISH_DOWNTREND"
    SIDEWAYS_CONSOLIDATION = "SIDEWAYS_CONSOLIDATION"


class CandlestickPattern(Enum):
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    HAMMER_PINBAR = "HAMMER_PINBAR"
    SHOOTING_STAR = "SHOOTING_STAR"
    DOJI = "DOJI"
    NONE = "NONE"


class MarketStructure:
    """Dataclass holding real-time chart analysis and historical trend memory findings."""

    def __init__(
        self,
        trend: TrendType,
        previous_trend: TrendType,
        candlestick_pattern: CandlestickPattern,
        support_level: float,
        resistance_level: float,
        ema_fast: float,
        ema_slow: float,
        ema_baseline: float,
        trend_score: float,  # +1.0 for strong bull, -1.0 for strong bear
        trend_continuation_signal: bool = False,
        trend_reversal_signal: bool = False,
    ) -> None:
        self.trend = trend
        self.previous_trend = previous_trend
        self.candlestick_pattern = candlestick_pattern
        self.support_level = support_level
        self.resistance_level = resistance_level
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_baseline = ema_baseline
        self.trend_score = trend_score
        self.trend_continuation_signal = trend_continuation_signal
        self.trend_reversal_signal = trend_reversal_signal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend": self.trend.value,
            "previous_trend": self.previous_trend.value,
            "candlestick_pattern": self.candlestick_pattern.value,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "ema_baseline": self.ema_baseline,
            "trend_score": self.trend_score,
            "trend_continuation_signal": self.trend_continuation_signal,
            "trend_reversal_signal": self.trend_reversal_signal,
        }


class PatternTrendEngine:
    """
    Real-time chart pattern scanner, historical trend memory, and breakout engine.
    """

    def __init__(
        self,
        fast_ema_period: int = 20,
        slow_ema_period: int = 50,
        baseline_ema_period: int = 200,
        pivot_lookback: int = 20,
        historical_trend_lookback: int = 50,
    ) -> None:
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.baseline_ema_period = baseline_ema_period
        self.pivot_lookback = pivot_lookback
        self.historical_trend_lookback = historical_trend_lookback

    def analyze_chart(self, df: pd.DataFrame) -> MarketStructure:
        """
        Analyzes OHLCV dataset up to the current bar to extract current and historical trend memory, patterns, and levels.
        """
        if len(df) < 10:
            return MarketStructure(
                trend=TrendType.SIDEWAYS_CONSOLIDATION,
                previous_trend=TrendType.SIDEWAYS_CONSOLIDATION,
                candlestick_pattern=CandlestickPattern.NONE,
                support_level=0.0,
                resistance_level=0.0,
                ema_fast=0.0,
                ema_slow=0.0,
                ema_baseline=0.0,
                trend_score=0.0,
                trend_continuation_signal=False,
                trend_reversal_signal=False,
            )

        data = df.copy()

        # 1. Calculate EMAs (20, 50, 200)
        ema20 = data["close"].ewm(span=self.fast_ema_period, adjust=False).mean()
        ema50 = data["close"].ewm(span=self.slow_ema_period, adjust=False).mean()
        ema200 = data["close"].ewm(span=self.baseline_ema_period, adjust=False).mean()

        curr_close = float(data["close"].iloc[-1])
        curr_ema20 = float(ema20.iloc[-1])
        curr_ema50 = float(ema50.iloc[-1])
        curr_ema200 = float(ema200.iloc[-1])

        # 2. Determine Current Trend & Trend Score
        trend_score = 0.0
        if curr_close > curr_ema20 > curr_ema50 > curr_ema200:
            trend = TrendType.BULLISH_UPTREND
            trend_score = 1.0
        elif curr_close > curr_ema20 > curr_ema50:
            trend = TrendType.BULLISH_UPTREND
            trend_score = 0.6
        elif curr_close < curr_ema20 < curr_ema50 < curr_ema200:
            trend = TrendType.BEARISH_DOWNTREND
            trend_score = -1.0
        elif curr_close < curr_ema20 < curr_ema50:
            trend = TrendType.BEARISH_DOWNTREND
            trend_score = -0.6
        else:
            trend = TrendType.SIDEWAYS_CONSOLIDATION
            trend_score = 0.0

        # 3. Determine Historical Previous Trend (Lookback Window)
        hist_idx = max(0, len(data) - self.historical_trend_lookback)
        prev_close = float(data["close"].iloc[hist_idx])
        prev_ema20 = float(ema20.iloc[hist_idx])
        prev_ema50 = float(ema50.iloc[hist_idx])

        if prev_close > prev_ema20 > prev_ema50:
            previous_trend = TrendType.BULLISH_UPTREND
        elif prev_close < prev_ema20 < prev_ema50:
            previous_trend = TrendType.BEARISH_DOWNTREND
        else:
            previous_trend = TrendType.SIDEWAYS_CONSOLIDATION

        # 4. Detect Candlestick Pattern on latest bar
        curr_bar = data.iloc[-1]
        prev_bar = data.iloc[-2]

        c_open, c_high, c_low, c_close = float(curr_bar["open"]), float(curr_bar["high"]), float(curr_bar["low"]), float(curr_bar["close"])
        p_open, p_high, p_low, p_close = float(prev_bar["open"]), float(prev_bar["high"]), float(prev_bar["low"]), float(prev_bar["close"])

        body = abs(c_close - c_open)
        rng = max(c_high - c_low, 1e-8)
        upper_wick = c_high - max(c_open, c_close)
        lower_wick = min(c_open, c_close) - c_low

        pattern = CandlestickPattern.NONE

        if p_close < p_open and c_close > c_open and c_close >= p_open and c_open <= p_close:
            pattern = CandlestickPattern.BULLISH_ENGULFING
        elif p_close > p_open and c_close < c_open and c_close <= p_open and c_open >= p_close:
            pattern = CandlestickPattern.BEARISH_ENGULFING
        elif lower_wick >= 2.0 * body and upper_wick <= 0.5 * body:
            pattern = CandlestickPattern.HAMMER_PINBAR
        elif upper_wick >= 2.0 * body and lower_wick <= 0.5 * body:
            pattern = CandlestickPattern.SHOOTING_STAR
        elif body / rng < 0.10:
            pattern = CandlestickPattern.DOJI

        # 5. Support and Resistance via Pivot Highs / Lows
        recent_data = data.tail(min(self.pivot_lookback, len(data)))
        lows = recent_data["low"].values
        highs = recent_data["high"].values

        support = float(np.min(lows))
        resistance = float(np.max(highs))

        # 6. Trend Continuation & Reversal Signals based on Historical Trend Memory
        trend_continuation_signal = False
        trend_reversal_signal = False

        # Continuation: Previous Trend was Bullish, price pulled back near EMA 20/50 and forms Bullish pattern
        if previous_trend == TrendType.BULLISH_UPTREND and trend == TrendType.BULLISH_UPTREND:
            if pattern in [CandlestickPattern.BULLISH_ENGULFING, CandlestickPattern.HAMMER_PINBAR] or (curr_close >= curr_ema20):
                trend_continuation_signal = True

        # Reversal: Previous Trend was Bearish, but current price breaks out above previous swing resistance
        if previous_trend == TrendType.BEARISH_DOWNTREND and curr_close > resistance:
            trend_reversal_signal = True

        return MarketStructure(
            trend=trend,
            previous_trend=previous_trend,
            candlestick_pattern=pattern,
            support_level=support,
            resistance_level=resistance,
            ema_fast=curr_ema20,
            ema_slow=curr_ema50,
            ema_baseline=curr_ema200,
            trend_score=trend_score,
            trend_continuation_signal=trend_continuation_signal,
            trend_reversal_signal=trend_reversal_signal,
        )
