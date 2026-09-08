"""
Multi-Strategy Synthesis Engine.
Combines 6 Famous and Price-Action Rudimentary Strategies into a unified Strategy Confluence Score:
1. EMA Trend Crossover (20/50/200)
2. RSI Overbought / Oversold
3. MACD Momentum
4. Bollinger Bands Volatility
5. Support & Resistance Pivot Rejection
6. Price Action Candlestick Patterns
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

from selftrade.analysis.pattern_engine import MarketStructure, TrendType, CandlestickPattern


@dataclass
class EnsembleEvaluationResult:
    """Dataclass holding multi-strategy ensemble evaluation votes and overall confluence score."""
    overall_score: float                # -1.0 (Strong Bear) to +1.0 (Strong Bull)
    ensemble_bias: str                 # BULLISH, BEARISH, or NEUTRAL
    strategy_votes: Dict[str, float]   # Individual strategy scores
    active_strategies_count: int       # Number of confirming strategies


class StrategyEnsemble:
    """
    Ensemble engine synthesizing famous technical indicator strategies and price action signals.
    """

    def __init__(self, bb_window: int = 20, bb_std: float = 2.0) -> None:
        self.bb_window = bb_window
        self.bb_std = bb_std

    def evaluate_strategies(self, df: pd.DataFrame, market_struct: MarketStructure) -> EnsembleEvaluationResult:
        """
        Evaluates 6 famous and rudimentary strategies across recent price series.
        """
        if len(df) < 20:
            return EnsembleEvaluationResult(
                overall_score=0.0,
                ensemble_bias="NEUTRAL",
                strategy_votes={},
                active_strategies_count=0,
            )

        data = df.tail(50).copy()
        curr_close = float(data["close"].iloc[-1])

        votes: Dict[str, float] = {}

        # 1. Strategy 1: EMA Trend Crossover (20/50/200)
        votes["EMA_Trend_Cross"] = float(market_struct.trend_score)

        # 2. Strategy 2: RSI Overbought / Oversold (14)
        delta = data["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = float(100.0 - (100.0 / (1.0 + rs.iloc[-1])))

        rsi_score = 0.0
        if rsi < 35.0:
            rsi_score = 1.0  # Oversold Buy
        elif rsi > 65.0:
            rsi_score = -1.0  # Overbought Sell
        votes["RSI_Oversold_Overbought"] = rsi_score

        # 3. Strategy 3: MACD Momentum
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd - signal).iloc[-1])

        macd_score = 1.0 if macd_hist > 0 else (-1.0 if macd_hist < 0 else 0.0)
        votes["MACD_Momentum"] = macd_score

        # 4. Strategy 4: Bollinger Bands Reversion
        ma = data["close"].rolling(window=self.bb_window).mean()
        std = data["close"].rolling(window=self.bb_window).std()
        upper_bb = float((ma + self.bb_std * std).iloc[-1])
        lower_bb = float((ma - self.bb_std * std).iloc[-1])

        bb_score = 0.0
        if curr_close <= lower_bb:
            bb_score = 1.0  # Touch lower band -> Bullish bounce
        elif curr_close >= upper_bb:
            bb_score = -1.0  # Touch upper band -> Bearish rejection
        votes["Bollinger_Bands_Reversion"] = bb_score

        # 5. Strategy 5: Support & Resistance Pivot Rejection
        sr_score = 0.0
        if market_struct.support_level > 0.0:
            dist_sup = (curr_close - market_struct.support_level) / curr_close
            dist_res = (market_struct.resistance_level - curr_close) / curr_close

            if dist_sup < 0.01:
                sr_score = 1.0  # Near Support -> Bullish bounce
            elif dist_res < 0.01:
                sr_score = -1.0  # Near Resistance -> Bearish rejection
        votes["Support_Resistance_Pivot"] = sr_score

        # 6. Strategy 6: Price Action Candlestick Patterns
        pa_score = 0.0
        if market_struct.candlestick_pattern in [CandlestickPattern.BULLISH_ENGULFING, CandlestickPattern.HAMMER_PINBAR]:
            pa_score = 1.0
        elif market_struct.candlestick_pattern in [CandlestickPattern.BEARISH_ENGULFING, CandlestickPattern.SHOOTING_STAR]:
            pa_score = -1.0
        votes["Price_Action_Patterns"] = pa_score

        # Calculate Overall Ensemble Confluence Score
        scores = list(votes.values())
        overall_score = float(np.mean(scores))

        active_count = sum(1 for s in scores if abs(s) > 0.2)

        if overall_score > 0.25:
            bias = "BULLISH"
        elif overall_score < -0.25:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return EnsembleEvaluationResult(
            overall_score=overall_score,
            ensemble_bias=bias,
            strategy_votes=votes,
            active_strategies_count=active_count,
        )
