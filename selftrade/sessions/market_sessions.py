"""
Market Opening Sessions Engine.
Evaluates global financial market trading sessions (Sydney/Asian, London, New York, London/NY Overlap) and their effects on volatility and trading volume.
"""

from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime, timezone
import pandas as pd


class MarketSessionType(Enum):
    ASIAN_SESSION = "ASIAN_SESSION"            # 00:00 - 08:00 UTC (Range-bound)
    LONDON_SESSION = "LONDON_SESSION"          # 08:00 - 13:00 UTC (Trend initiation)
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"    # 13:00 - 16:00 UTC (Peak volume & liquidity)
    NEW_YORK_SESSION = "NEW_YORK_SESSION"      # 16:00 - 21:00 UTC (High volatility)
    OFF_PEAK = "OFF_PEAK"                      # 21:00 - 00:00 UTC (Low liquidity)


class MarketSessionInfo:
    """Dataclass holding real-time market session telemetry."""

    def __init__(
        self,
        session_type: MarketSessionType,
        session_name: str,
        volatility_multiplier: float,
        is_high_liquidity: bool,
        trading_bias_score: float,  # +1.0 for trend continuation, 0.0 for range
    ) -> None:
        self.session_type = session_type
        self.session_name = session_name
        self.volatility_multiplier = volatility_multiplier
        self.is_high_liquidity = is_high_liquidity
        self.trading_bias_score = trading_bias_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_type": self.session_type.value,
            "session_name": self.session_name,
            "volatility_multiplier": self.volatility_multiplier,
            "is_high_liquidity": self.is_high_liquidity,
            "trading_bias_score": self.trading_bias_score,
        }


class MarketSessionsEngine:
    """
    Evaluates market session opening hours and their characteristic trading behavior.
    """

    @staticmethod
    def get_session_info(timestamp: Optional[pd.Timestamp] = None) -> MarketSessionInfo:
        """
        Determines market session characteristics for a given UTC timestamp or current system time.
        """
        if timestamp is None:
            utc_now = datetime.now(timezone.utc)
            hour = utc_now.hour
        else:
            ts = pd.to_datetime(timestamp)
            if ts.tzinfo is None:
                hour = ts.hour
            else:
                hour = ts.tz_convert("UTC").hour

        # Determine session window
        if 13 <= hour < 16:
            return MarketSessionInfo(
                session_type=MarketSessionType.LONDON_NY_OVERLAP,
                session_name="London / New York Overlap",
                volatility_multiplier=1.5,
                is_high_liquidity=True,
                trading_bias_score=1.0,  # Optimal trend & breakout trading
            )
        elif 8 <= hour < 13:
            return MarketSessionInfo(
                session_type=MarketSessionType.LONDON_SESSION,
                session_name="London Session",
                volatility_multiplier=1.2,
                is_high_liquidity=True,
                trading_bias_score=0.8,  # Strong momentum
            )
        elif 16 <= hour < 21:
            return MarketSessionInfo(
                session_type=MarketSessionType.NEW_YORK_SESSION,
                session_name="New York Session",
                volatility_multiplier=1.3,
                is_high_liquidity=True,
                trading_bias_score=0.7,  # High volatility
            )
        elif 0 <= hour < 8:
            return MarketSessionInfo(
                session_type=MarketSessionType.ASIAN_SESSION,
                session_name="Sydney / Asian Session",
                volatility_multiplier=0.7,
                is_high_liquidity=False,
                trading_bias_score=0.3,  # Mean reversion / range-bound
            )
        else:
            return MarketSessionInfo(
                session_type=MarketSessionType.OFF_PEAK,
                session_name="Off-Peak / Asian Evening",
                volatility_multiplier=0.5,
                is_high_liquidity=False,
                trading_bias_score=0.1,  # Low liquidity caution
            )
