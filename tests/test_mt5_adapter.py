"""
Unit Tests for MetaTrader 5 (MT5) Live Broker Adapter using mock MT5 interface.
"""

from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import pandas as pd

from selftrade.live.mt5_adapter import MT5BrokerAdapter


@pytest.fixture
def mock_mt5():
    with patch("selftrade.live.mt5_adapter.mt5") as mock, patch("selftrade.live.mt5_adapter.save_credentials"):
        mock.TIMEFRAME_M1 = 1
        mock.TIMEFRAME_M5 = 5
        mock.TIMEFRAME_M15 = 15
        mock.TIMEFRAME_H1 = 16385
        mock.TIMEFRAME_D1 = 16408

        mock.ORDER_TYPE_BUY = 0
        mock.ORDER_TYPE_SELL = 1
        mock.TRADE_ACTION_DEAL = 1
        mock.ORDER_TIME_GTC = 0
        mock.ORDER_FILLING_IOC = 1
        mock.ORDER_FILLING_FOK = 2
        mock.ORDER_FILLING_RETURN = 4
        mock.TRADE_RETCODE_DONE = 10009

        yield mock


def test_mt5_adapter_connect_success(mock_mt5):
    mock_mt5.initialize.return_value = True
    mock_mt5.login.return_value = True
    acc_info = MagicMock()
    acc_info.login = 998877
    acc_info.server = "MetaQuotes-Demo"
    acc_info.company = "MetaQuotes"
    acc_info.equity = 10000.0
    acc_info.margin_free = 10000.0
    acc_info.margin = 0.0
    mock_mt5.account_info.side_effect = [None, acc_info]

    adapter = MT5BrokerAdapter(account=998877, password="password", server="MetaQuotes-Demo")
    res = adapter.connect()

    assert res is True
    assert adapter.is_connected is True
    mock_mt5.initialize.assert_called_once()
    mock_mt5.login.assert_called_once_with(login=998877, password="password", server="MetaQuotes-Demo")


def test_mt5_adapter_fetch_account_balance(mock_mt5):
    mock_mt5.initialize.return_value = True
    mock_mt5.login.return_value = True
    acc_info = MagicMock()
    acc_info.equity = 10500.0
    acc_info.margin_free = 9500.0
    acc_info.margin = 1000.0
    mock_mt5.account_info.return_value = acc_info

    adapter = MT5BrokerAdapter(account=123456, password="password", server="MetaQuotes-Demo")
    adapter.is_connected = True

    eq, free, used = adapter.fetch_account_balance()
    assert eq == 10500.0
    assert free == 9500.0
    assert used == 1000.0


def test_mt5_adapter_fetch_latest_candles(mock_mt5):
    mock_mt5.initialize.return_value = True
    mock_mt5.symbol_select.return_value = True

    dummy_rates = np.array(
        [
            (1600000000, 1.1000, 1.1050, 1.0990, 1.1020, 100, 0, 0),
            (1600000300, 1.1020, 1.1080, 1.1010, 1.1070, 150, 0, 0),
        ],
        dtype=[
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i4"),
            ("real_volume", "i8"),
        ],
    )
    mock_mt5.copy_rates_from_pos.return_value = dummy_rates

    adapter = MT5BrokerAdapter()
    adapter.is_connected = True

    df = adapter.fetch_latest_candles(symbol="EURUSD", timeframe="5m", limit=2)
    assert not df.empty
    assert len(df) == 2
    assert "close" in df.columns
    assert df["close"].iloc[-1] == 1.1070


def test_mt5_adapter_execute_order(mock_mt5):
    mock_mt5.initialize.return_value = True
    mock_mt5.symbol_select.return_value = True

    sym_info = MagicMock()
    sym_info.visible = True
    sym_info.volume_min = 0.01
    sym_info.volume_max = 100.0
    sym_info.filling_mode = 1  # IOC
    mock_mt5.symbol_info.return_value = sym_info

    tick = MagicMock()
    tick.ask = 1.1050
    tick.bid = 1.1048
    mock_mt5.symbol_info_tick.return_value = tick

    order_result = MagicMock()
    order_result.retcode = 10009  # TRADE_RETCODE_DONE
    order_result.order = 998877
    order_result.price = 1.1050
    mock_mt5.order_send.return_value = order_result

    adapter = MT5BrokerAdapter()
    adapter.is_connected = True

    res = adapter.execute_order(
        symbol="EURUSD",
        action=1,  # BUY
        amount=0.0,
        current_price=1.1050,
        stop_loss=1.0950,
        take_profit=1.1250,
        lot_size=0.01,
    )

    assert res["status"] == "FILLED"
    assert res["side"] == "BUY"
    assert res["ticket"] == 998877
    assert res["price"] == 1.1050
    mock_mt5.order_send.assert_called_once()
