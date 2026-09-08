"""
Unit Tests for Trade Journal and Diary Logger.
"""

import pytest
import tempfile
import os

from selftrade.journal.trade_journal import TradeJournal, TradeJournalEntry


def test_trade_journal_logging_and_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_file = os.path.join(tmpdir, "trade_journal.json")
        journal = TradeJournal(journal_filepath=journal_file)

        entry = journal.log_trade(
            trade_id="TRD-0001",
            timestamp="2026-09-08 05:00:00",
            symbol="XAUUSD",
            side="BUY",
            entry_price=2500.0,
            exit_price=2550.0,
            units=4.0,
            pnl_usd=200.0,
            pnl_pct=2.0,
            take_profit=2550.0,
            stop_loss=2475.0,
            risk_reward_ratio=2.0,
            confluence_score=0.75,
            trend_condition="BULLISH_UPTREND",
            candlestick_pattern="BULLISH_ENGULFING",
            reasoner_summary="APPROVED trade",
            risk_verdict="PASSED_RISK_GATES",
            model_version="v1.0_active",
        )

        assert entry.trade_id == "TRD-0001"
        assert len(journal.entries) == 1
        assert os.path.exists(journal_file)

        # Export CSV check
        csv_file = os.path.join(tmpdir, "trade_journal.csv")
        journal.export_csv(csv_file)
        assert os.path.exists(csv_file)

        # Reload check
        reloaded_journal = TradeJournal(journal_filepath=journal_file)
        assert len(reloaded_journal.entries) == 1
        assert reloaded_journal.entries[0].symbol == "XAUUSD"
