"""
Trade Journal & Cognitive Diary Engine.
Logs every trade execution along with entry/exit prices, realized PnL, confluence scores, technical patterns, and reasoner verdicts.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
import os
import pandas as pd


@dataclass
class TradeJournalEntry:
    """Dataclass representing a complete trade diary record."""
    trade_id: str
    timestamp: str
    symbol: str
    side: str                          # BUY / SELL
    entry_price: float
    exit_price: float
    units: float
    pnl_usd: float
    pnl_pct: float
    take_profit: float
    stop_loss: float
    risk_reward_ratio: float
    confluence_score: float
    trend_condition: str
    candlestick_pattern: str
    reasoner_summary: str
    risk_verdict: str
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeJournal:
    """
    Persistent Trade Journal & Diary Manager.
    Saves trade records to JSON/CSV files and provides query APIs for GUI visualization.
    """

    def __init__(self, journal_filepath: str = "journal/trade_journal.json") -> None:
        self.journal_filepath = journal_filepath
        self.entries: List[TradeJournalEntry] = []
        os.makedirs(os.path.dirname(os.path.abspath(journal_filepath)), exist_ok=True)
        self.load()

    def log_trade(
        self,
        trade_id: str,
        timestamp: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        units: float,
        pnl_usd: float,
        pnl_pct: float,
        take_profit: float,
        stop_loss: float,
        risk_reward_ratio: float,
        confluence_score: float,
        trend_condition: str,
        candlestick_pattern: str,
        reasoner_summary: str,
        risk_verdict: str,
        model_version: str,
    ) -> TradeJournalEntry:
        """
        Creates and logs a new trade diary record.
        """
        entry = TradeJournalEntry(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            units=units,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            take_profit=take_profit,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            confluence_score=confluence_score,
            trend_condition=trend_condition,
            candlestick_pattern=candlestick_pattern,
            reasoner_summary=reasoner_summary,
            risk_verdict=risk_verdict,
            model_version=model_version,
        )
        self.entries.append(entry)
        self.save()
        return entry

    def save(self) -> None:
        """Saves trade entries to JSON file."""
        data = [e.to_dict() for e in self.entries]
        with open(self.journal_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Loads trade entries from JSON file if exists."""
        if not os.path.exists(self.journal_filepath):
            return
        try:
            with open(self.journal_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entries = [TradeJournalEntry(**item) for item in data]
        except Exception:
            self.entries = []

    load_journal = load

    def export_csv(self, csv_path: str = "journal/trade_journal.csv") -> str:
        """Exports trade diary records to CSV format."""
        if not self.entries:
            return csv_path
        df = pd.DataFrame([e.to_dict() for e in self.entries])
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        df.to_csv(csv_path, index=False)
        return csv_path

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates journal summary statistics."""
        if not self.entries:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl_usd": 0.0,
                "profit_factor": 0.0,
            }
        total = len(self.entries)
        wins = [e for e in self.entries if e.pnl_usd > 0]
        losses = [e for e in self.entries if e.pnl_usd < 0]
        win_rate = (len(wins) / total) * 100.0 if total > 0 else 0.0

        gross_profit = sum(e.pnl_usd for e in wins)
        gross_loss = abs(sum(e.pnl_usd for e in losses)) or 1e-8
        profit_factor = gross_profit / gross_loss
        total_pnl = sum(e.pnl_usd for e in self.entries)

        return {
            "total_trades": total,
            "win_rate": win_rate,
            "total_pnl_usd": total_pnl,
            "profit_factor": profit_factor,
        }
