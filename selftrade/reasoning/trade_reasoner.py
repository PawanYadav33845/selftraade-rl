"""
Multi-Factor Trade Reasoning Engine.
"Thinks before taking a trade", evaluates Market Opening Sessions, Strategy Ensemble Confluence, enforces dynamic Risk-to-Reward scaling, and sets dual Take-Profit (TP1 scalp / TP2 trend) and Break-Even targets.
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from selftrade.analysis.pattern_engine import MarketStructure, TrendType
from selftrade.sessions.market_sessions import MarketSessionsEngine, MarketSessionInfo
from selftrade.strategies.ensemble import StrategyEnsemble, EnsembleEvaluationResult


@dataclass
class ReasonedTradeDecision:
    """Dataclass holding trade reasoning verdict, confluence score, dual TP levels, and holding mode."""
    approved_action: int               # 0: Hold, 1: Buy, 2: Sell
    confluence_score: float            # 0.0 to 1.0 confidence score
    stop_loss_price: float             # Calculated safe stop-loss price
    take_profit_price: float           # Primary take-profit target
    take_profit_tp1: float = 0.0       # Quick Scalp Target (1.2x - 1.5x Risk)
    take_profit_tp2: float = 0.0       # Big Trend Runner Target (2.5x - 4.0x Risk)
    break_even_trigger_price: float = 0.0 # Price level activating Break-Even SL shift (+1.0 R)
    holding_mode: str = "SCALP_QUICK"  # "SCALP_QUICK" vs "TREND_BIG_PROFIT"
    risk_reward_ratio: float = 2.0     # Adaptive R:R ratio
    market_session_name: str = ""      # Current market session name
    reasoning_summary: str = ""        # Explanatory text log


class TradeReasoner:
    """
    Cognitive reasoning engine synthesizing market sessions, strategy ensembles, dynamic TP/SL geometry,
    and adaptive holding modes (Scalp Quick Profits vs Big Trend Runners).
    """

    def __init__(
        self,
        min_confluence_threshold: float = 0.65,
        target_risk_reward: float = 2.0,  # Base 1:2.0 R:R ratio
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
        symbol: str = "",
    ) -> ReasonedTradeDecision:
        """
        Evaluates proposed trade action against Market Session, Strategy Ensemble, and adaptive TP/SL geometry,
        applying category-tailored volatility and risk parameters for Gold (XAUUSD), Silver (XAGUSD), and Crypto (BTCUSD).
        """
        # 1. Market Session Analysis
        session_info: MarketSessionInfo = MarketSessionsEngine.get_session_info(timestamp)

        if proposed_action == 0:  # HOLD
            return ReasonedTradeDecision(
                approved_action=0,
                confluence_score=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                take_profit_tp1=0.0,
                take_profit_tp2=0.0,
                break_even_trigger_price=0.0,
                holding_mode="HOLD",
                risk_reward_ratio=0.0,
                market_session_name=session_info.session_name,
                reasoning_summary="HOLD_ACTION: No trade proposed.",
            )

        # Category-Specific Parameter Optimization
        sym_upper = symbol.upper()
        category_atr_mult = self.atr_sl_multiplier
        if "BTC" in sym_upper:
            category_atr_mult = 2.0
        elif "XAU" in sym_upper:
            category_atr_mult = 1.8
        elif "XAG" in sym_upper:
            category_atr_mult = 1.6

        # 2. Strategy Ensemble Evaluation
        ensemble_res: Optional[EnsembleEvaluationResult] = None
        if df is not None:
            ensemble_res = self.strategy_ensemble.evaluate_strategies(df, market_structure, symbol=symbol)

        confluence_score = 0.50  # Base score
        reasons = [f"Session: {session_info.session_name}"]

        # Apply Session Volatility Bias
        confluence_score += (session_info.trading_bias_score - 0.5) * 0.2

        # Apply Trend Memory & Pattern Signal Confluence
        is_trend_continuation = False
        is_reversal = False

        if market_structure is not None and market_structure.trend_continuation_signal:
            if (proposed_action == 1 and market_structure.trend == TrendType.BULLISH_UPTREND) or \
               (proposed_action == 2 and market_structure.trend == TrendType.BEARISH_DOWNTREND):
                confluence_score += 0.15
                is_trend_continuation = True
                reasons.append(f"Trend Memory: {market_structure.trend.value} Continuation")
        elif market_structure is not None and market_structure.trend_reversal_signal:
            if proposed_action == 1:
                confluence_score += 0.20
                is_reversal = True
                reasons.append("Trend Memory: Bullish Reversal Breakout")
            elif proposed_action == 2:
                confluence_score += 0.20
                is_reversal = True
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

        # 3. Determine Adaptive Holding Mode & Scaled R:R Ratio
        if is_trend_continuation or is_reversal or confluence_score >= 0.70:
            holding_mode = "TREND_BIG_PROFIT"
            adaptive_rr = max(self.target_risk_reward, 3.0)  # Scale target R:R up to 1:3.0+ for big trend runs
            reasons.append("Holding Strategy: BIG TREND RUNNER (Scaled R:R 1:3.0)")
        else:
            holding_mode = "SCALP_QUICK"
            adaptive_rr = self.target_risk_reward
            reasons.append(f"Holding Strategy: QUICK SCALP (R:R 1:{adaptive_rr:.1f})")

        # 4. Calculate Dynamic Stop-Loss & Dual Take-Profit Levels
        safe_atr = max(atr, current_price * 0.005)

        if proposed_action == 1:  # BUY
            raw_sl = current_price - (category_atr_mult * safe_atr)
            if market_structure is not None and market_structure.support_level > 0.0 and market_structure.support_level < current_price:
                sl_price = min(raw_sl, market_structure.support_level * 0.998)
            else:
                sl_price = raw_sl

            risk_dist = current_price - sl_price
            tp1_price = current_price + (1.5 * risk_dist)
            tp2_price = current_price + (adaptive_rr * risk_dist)
            be_trigger = current_price + (1.0 * risk_dist)

        else:  # SELL
            raw_sl = current_price + (category_atr_mult * safe_atr)
            if market_structure is not None and market_structure.resistance_level > 0.0 and market_structure.resistance_level > current_price:
                sl_price = max(raw_sl, market_structure.resistance_level * 1.002)
            else:
                sl_price = raw_sl

            risk_dist = sl_price - current_price
            tp1_price = current_price - (1.5 * risk_dist)
            tp2_price = current_price - (adaptive_rr * risk_dist)
            be_trigger = current_price - (1.0 * risk_dist)

        primary_tp = tp2_price if holding_mode == "TREND_BIG_PROFIT" else tp1_price

        # 5. Final Approval Verdict
        if confluence_score >= self.min_confluence_threshold:
            approved = proposed_action
            summary = f"APPROVED [{holding_mode}] trade (Score: {confluence_score:.2f} >= {self.min_confluence_threshold:.2f}). Reasons: {'; '.join(reasons)}"
        else:
            approved = 0  # REJECT / HOLD
            summary = f"REJECTED trade (Score: {confluence_score:.2f} < {self.min_confluence_threshold:.2f}). Overridden to HOLD. Reasons: {'; '.join(reasons)}"

        return ReasonedTradeDecision(
            approved_action=approved,
            confluence_score=confluence_score,
            stop_loss_price=float(sl_price),
            take_profit_price=float(primary_tp),
            take_profit_tp1=float(tp1_price),
            take_profit_tp2=float(tp2_price),
            break_even_trigger_price=float(be_trigger),
            holding_mode=holding_mode,
            risk_reward_ratio=adaptive_rr,
            market_session_name=session_info.session_name,
            reasoning_summary=summary,
        )
