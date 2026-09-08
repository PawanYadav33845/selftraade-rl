"""
Multi-Factor Trade Reasoning Engine.
"Thinks before taking a trade", evaluates Market Opening Sessions, Strategy Ensemble Confluence, enforces minimum 1:2.0 Risk-to-Reward ratio, and sets dynamic Take-Profit (TP) and Stop-Loss (SL) targets.
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from selftrade.analysis.pattern_engine import MarketStructure, TrendType
from selftrade.sessions.market_sessions import MarketSessionsEngine, MarketSessionInfo
from selftrade.strategies.ensemble import StrategyEnsemble, EnsembleEvaluationResult


@dataclass
class ReasonedTradeDecision:
    """Dataclass holding trade reasoning verdict, confluence score, and safe TP/SL prices."""
    approved_action: int               # 0: Hold, 1: Buy, 2: Sell
    confluence_score: float            # 0.0 to 1.0 confidence score
    stop_loss_price: float             # Calculated safe stop-loss price
    take_profit_price: float           # Calculated safe take-profit price
    risk_reward_ratio: float           # R:R ratio (e.g., 2.0 means 1:2.0 R:R)
    market_session_name: str           # Current market session name
    reasoning_summary: str             # Explanatory text log


class TradeReasoner:
    """
    Cognitive reasoning engine synthesizing market sessions, strategy ensembles, and dynamic TP/SL geometry.
    """

    def __init__(
        self,
        min_confluence_threshold: float = 0.60,
        target_risk_reward: float = 2.0,  # 1:2.0 R:R ratio
        atr_sl_multiplier: float = 1.5,
    ) -> None:
        self.min_confluence_threshold = min_confluence_threshold
        self.target_risk_reward = target_risk_reward
        self.atr_sl_multiplier = atr_sl_multiplier
        self.strategy_ensemble = StrategyEnsemble()

    def evaluate_trade(
        self,
        proposed_action: int,
        current_price: float,
        atr: float,
        market_structure: MarketStructure,
        df: Optional[Any] = None,
        timestamp: Optional[Any] = None,
    ) -> ReasonedTradeDecision:
        """
        Evaluates proposed trade action against Market Session, Strategy Ensemble, and TP/SL geometry.
        """
        # 1. Market Session Analysis
        session_info: MarketSessionInfo = MarketSessionsEngine.get_session_info(timestamp)

        if proposed_action == 0:  # HOLD
            return ReasonedTradeDecision(
                approved_action=0,
                confluence_score=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                risk_reward_ratio=0.0,
                market_session_name=session_info.session_name,
                reasoning_summary="HOLD_ACTION: No trade proposed.",
            )

        # 2. Strategy Ensemble Evaluation
        ensemble_res: Optional[EnsembleEvaluationResult] = None
        if df is not None:
            ensemble_res = self.strategy_ensemble.evaluate_strategies(df, market_structure)

        confluence_score = 0.50  # Base score
        reasons = [f"Session: {session_info.session_name}"]

        # Apply Session Volatility Bias
        confluence_score += (session_info.trading_bias_score - 0.5) * 0.2

        # Apply Trend Memory & Pattern Signal Confluence
        if market_structure.trend_continuation_signal:
            if (proposed_action == 1 and market_structure.trend == TrendType.BULLISH_UPTREND) or \
               (proposed_action == 2 and market_structure.trend == TrendType.BEARISH_DOWNTREND):
                confluence_score += 0.15
                reasons.append(f"Trend Memory: {market_structure.trend.value} Continuation")
        elif market_structure.trend_reversal_signal:
            if proposed_action == 1:
                confluence_score += 0.20
                reasons.append("Trend Memory: Bullish Reversal Breakout")
            elif proposed_action == 2:
                confluence_score += 0.20
                reasons.append("Trend Memory: Bearish Reversal Breakout")

        # Apply Strategy Ensemble Bias
        if ensemble_res is not None:
            ens_score = ensemble_res.overall_score
            if proposed_action == 1 and ens_score > 0.1:  # BUY
                confluence_score += ens_score * 0.3
                reasons.append(f"Ensemble: {ensemble_res.ensemble_bias} ({ens_score:+.2f})")
            elif proposed_action == 2 and ens_score < -0.1:  # SELL
                confluence_score += abs(ens_score) * 0.3
                reasons.append(f"Ensemble: {ensemble_res.ensemble_bias} ({ens_score:+.2f})")
            elif (proposed_action == 1 and ens_score < -0.2) or (proposed_action == 2 and ens_score > 0.2):
                confluence_score -= 0.3
                reasons.append(f"Ensemble Conflict: Opposite bias ({ens_score:+.2f})")

        confluence_score = float(np.clip(confluence_score, 0.0, 1.0))

        # 3. Calculate Dynamic Stop-Loss & Take-Profit Levels (1:2.0 R:R)
        safe_atr = max(atr, current_price * 0.005)

        if proposed_action == 1:  # BUY
            raw_sl = current_price - (self.atr_sl_multiplier * safe_atr)
            if market_structure.support_level > 0.0 and market_structure.support_level < current_price:
                sl_price = min(raw_sl, market_structure.support_level * 0.998)
            else:
                sl_price = raw_sl

            risk_dist = current_price - sl_price
            tp_price = current_price + (self.target_risk_reward * risk_dist)

        else:  # SELL
            raw_sl = current_price + (self.atr_sl_multiplier * safe_atr)
            if market_structure.resistance_level > 0.0 and market_structure.resistance_level > current_price:
                sl_price = max(raw_sl, market_structure.resistance_level * 1.002)
            else:
                sl_price = raw_sl

            risk_dist = sl_price - current_price
            tp_price = current_price - (self.target_risk_reward * risk_dist)

        # 4. Final Approval Verdict
        if confluence_score >= self.min_confluence_threshold:
            approved = proposed_action
            summary = f"APPROVED trade (Score: {confluence_score:.2f} >= {self.min_confluence_threshold:.2f}). Reasons: {'; '.join(reasons)}"
        else:
            approved = 0  # REJECT / HOLD
            summary = f"REJECTED trade (Score: {confluence_score:.2f} < {self.min_confluence_threshold:.2f}). Overridden to HOLD. Reasons: {'; '.join(reasons)}"

        return ReasonedTradeDecision(
            approved_action=approved,
            confluence_score=confluence_score,
            stop_loss_price=float(sl_price),
            take_profit_price=float(tp_price),
            risk_reward_ratio=self.target_risk_reward,
            market_session_name=session_info.session_name,
            reasoning_summary=summary,
        )
