"""
Unit Tests for MultiSymbolLiveManager (Concurrent Multi-Asset Execution Manager).
"""

from unittest.mock import MagicMock, patch
import time
import pytest

from selftrade.live.multi_symbol_manager import MultiSymbolLiveManager, SymbolTelemetry


def test_multi_symbol_manager_initialization(tmp_path):
    journal_file = str(tmp_path / "test_journal.json")
    manager = MultiSymbolLiveManager(target_rr_ratio=2.0, journal_path=journal_file)

    assert manager.target_rr_ratio == 2.0
    assert manager.is_running is False
    assert manager.is_force_stopped is False
    assert len(manager.active_symbols) == 0


def test_multi_symbol_manager_simulated_start_stop(tmp_path):
    journal_file = str(tmp_path / "test_journal.json")
    manager = MultiSymbolLiveManager(target_rr_ratio=2.0, journal_path=journal_file)

    symbols = ["XAUUSD", "EURUSD", "BTCUSD"]
    started = manager.start_trading(symbols=symbols, mode="Simulated Paper")

    assert started is True
    assert manager.is_running is True
    assert len(manager.worker_threads) == 3

    # Allow workers to run briefly
    time.sleep(0.4)

    with manager.lock:
        for sym in symbols:
            telem = manager.telemetry.get(sym)
            assert telem is not None
            assert len(telem.price_history) > 0
            assert telem.last_price > 0.0

    manager.stop_trading()
    assert manager.is_running is False


def test_multi_symbol_manager_force_stop(tmp_path):
    journal_file = str(tmp_path / "test_journal.json")
    manager = MultiSymbolLiveManager(target_rr_ratio=2.0, journal_path=journal_file)

    symbols = ["XAGUSD", "GBPUSD"]
    manager.start_trading(symbols=symbols, mode="Simulated Paper")
    time.sleep(0.2)

    manager.force_stop_all()

    assert manager.is_running is False
    assert manager.is_force_stopped is True


@patch("selftrade.live.multi_symbol_manager.MT5BrokerAdapter")
def test_multi_symbol_manager_mt5_mode(mock_mt5_adapter_class, tmp_path):
    mock_adapter = MagicMock()
    mock_adapter.connect.return_value = True
    mock_mt5_adapter_class.return_value = mock_adapter

    journal_file = str(tmp_path / "test_journal.json")
    manager = MultiSymbolLiveManager(
        target_rr_ratio=2.0,
        journal_path=journal_file,
        mt5_credentials={"account": 12345, "password": "pwd", "server": "srv"},
    )

    started = manager.start_trading(symbols=["XAUUSD"], mode="MT5 Demo Live")
    assert started is True
    mock_adapter.connect.assert_called_once()

    manager.stop_trading()
    mock_adapter.disconnect.assert_called_once()
