"""
Multi-Strategy Synthesis Engine.
Combines 8 Famous and Price-Action Rudimentary Strategies simultaneously into a unified Strategy Confluence Score:
1. EMA Trend Crossover (20/50/200)
2. RSI Overbought / Oversold
3. MACD Momentum
4. Bollinger Bands Volatility Reversion
5. Support & Resistance Pivot Rejection
6. Price Action Candlestick Patterns
7. Stochastic Oscillator (%K/%D)
8. Keltner Volatility Channel Breakout
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

    def evaluate_strategies(
        self,
        df: pd.DataFrame,
        market_struct: MarketStructure,
        symbol: str = "",
    ) -> EnsembleEvaluationResult:
        """
        Evaluates 8 famous and rudimentary strategies simultaneously across recent price series,
        applying category-optimized indicator weights for Commodities (Gold, Silver) and Crypto (BTCUSD).
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
        votes["EMA_Trend_Cross"] = float(market_struct.trend_score) if market_struct is not None else 0.0

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
        if market_struct is not None and market_struct.support_level > 0.0:
            dist_sup = (curr_close - market_struct.support_level) / curr_close
            dist_res = (market_struct.resistance_level - curr_close) / curr_close

            if dist_sup < 0.01:
                sr_score = 1.0  # Near Support -> Bullish bounce
            elif dist_res < 0.01:
                sr_score = -1.0  # Near Resistance -> Bearish rejection
        votes["Support_Resistance_Pivot"] = sr_score

        # 6. Strategy 6: Price Action Candlestick Patterns
        pa_score = 0.0
        if market_struct is not None:
            if market_struct.candlestick_pattern in [CandlestickPattern.BULLISH_ENGULFING, CandlestickPattern.HAMMER_PINBAR]:
                pa_score = 1.0
            elif market_struct.candlestick_pattern in [CandlestickPattern.BEARISH_ENGULFING, CandlestickPattern.SHOOTING_STAR]:
                pa_score = -1.0
        votes["Price_Action_Patterns"] = pa_score

        # 7. Strategy 7: Stochastic Oscillator (%K/%D 14)
        low_14 = data["low"].rolling(14).min()
        high_14 = data["high"].rolling(14).max()
        stoch_k = 100.0 * (data["close"] - low_14) / (high_14 - low_14 + 1e-8)
        k_val = float(stoch_k.iloc[-1])

        stoch_score = 0.0
        if k_val < 20.0:
            stoch_score = 1.0  # Oversold
        elif k_val > 80.0:
            stoch_score = -1.0  # Overbought
        votes["Stochastic_Oscillator"] = stoch_score

        # 8. Strategy 8: Keltner Volatility Channel Breakout
        tr = np.maximum(
            data["high"] - data["low"],
            np.maximum(
                abs(data["high"] - data["close"].shift(1)),
                abs(data["low"] - data["close"].shift(1)),
            ),
        )
        atr_14 = float(tr.rolling(14).mean().iloc[-1])
        ema_20 = float(data["close"].ewm(span=20, adjust=False).mean().iloc[-1])

        keltner_upper = ema_20 + 1.5 * atr_14
        keltner_lower = ema_20 - 1.5 * atr_14

        keltner_score = 0.0
        if curr_close > keltner_upper:
            keltner_score = 1.0  # Upper breakout -> Bullish momentum
        elif curr_close < keltner_lower:
            keltner_score = -1.0  # Lower breakout -> Bearish momentum
        votes["Keltner_Channel_Breakout"] = keltner_score

        # Category-Specific Strategy Optimization Weighting Matrix
        category_weights = {
            "XAUUSD": {  # Gold: Trend & Volatility Breakout Priority
                "EMA_Trend_Cross": 1.3,
                "Keltner_Channel_Breakout": 1.4,
                "Support_Resistance_Pivot": 1.2,
                "MACD_Momentum": 1.1,
                "Price_Action_Patterns": 1.1,
                "RSI_Oversold_Overbought": 0.9,
                "Bollinger_Bands_Reversion": 0.8,
                "Stochastic_Oscillator": 0.8,
            },
            "XAGUSD": {  # Silver: Mean Reversion & Oscillator Priority
                "Bollinger_Bands_Reversion": 1.4,
                "Stochastic_Oscillator": 1.3,
                "RSI_Oversold_Overbought": 1.2,
                "Support_Resistance_Pivot": 1.2,
                "Price_Action_Patterns": 1.1,
                "EMA_Trend_Cross": 0.9,
                "MACD_Momentum": 0.9,
                "Keltner_Channel_Breakout": 0.8,
            },
            "BTCUSD": {  # Crypto: Momentum & Trend Continuation Priority
                "MACD_Momentum": 1.4,
                "EMA_Trend_Cross": 1.3,
                "Price_Action_Patterns": 1.3,
                "Keltner_Channel_Breakout": 1.2,
                "Stochastic_Oscillator": 1.0,
                "RSI_Oversold_Overbought": 0.9,
                "Support_Resistance_Pivot": 0.9,
                "Bollinger_Bands_Reversion": 0.8,
            },
        }

        weights_dict = category_weights.get(symbol.upper(), {})
        weighted_scores = []
        weight_sum = 0.0

        for strat_name, raw_vote in votes.items():
            w = weights_dict.get(strat_name, 1.0)
            weighted_scores.append(raw_vote * w)
            weight_sum += w

        overall_score = float(np.sum(weighted_scores) / (weight_sum + 1e-8))
        active_count = sum(1 for s in votes.values() if abs(s) > 0.2)

        if overall_score > 0.20:
            bias = "BULLISH"
        elif overall_score < -0.20:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return EnsembleEvaluationResult(
            overall_score=overall_score,
            ensemble_bias=bias,
            strategy_votes=votes,
            active_strategies_count=active_count,
        )
