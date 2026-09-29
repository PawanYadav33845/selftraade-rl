"""
MetaTrader 5 (MT5) Live Broker Adapter for SelfTrade-RL.
Enables real-time candle streaming, robust symbol auto-discovery, persistent credential storage, live account balance tracking, open position monitoring, and market order execution with dynamic TP/SL geometry.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

import MetaTrader5 as mt5
from selftrade.live.broker import BaseBrokerAdapter

logger = logging.getLogger("SelfTrade.MT5Adapter")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
CREDENTIALS_FILE = os.path.join(CONFIG_DIR, "mt5_credentials.json")


TIMEFRAME_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


def load_saved_credentials() -> Dict[str, Any]:
    """Loads saved MT5 credentials from local config file if present."""
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load MT5 credentials config: {e}")
    return {}


def save_credentials(account: int, password: str, server: str) -> None:
    """Saves MT5 credentials to local config file (skips dummy test accounts)."""
    if not account or int(account) == 123456:
        return
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = {
        "account": int(account),
        "password": str(password),
        "server": str(server),
    }
    try:
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved MT5 credentials for account {account} to {CREDENTIALS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save MT5 credentials: {e}")


class MT5BrokerAdapter(BaseBrokerAdapter):
    """
    Adapter connecting SelfTrade-RL to native MetaTrader 5 desktop terminal.
    Supports symbol resolution, persistent credential storage, live data streaming, and automated execution.
    """

    def __init__(
        self,
        account: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        default_lot_size: float = 0.01,
    ) -> None:
        saved = load_saved_credentials()
        self.account = account if account is not None else saved.get("account")
        self.password = password if password is not None else saved.get("password")
        self.server = server if server is not None else saved.get("server", "MetaQuotes-Demo")
        self.path = path
        self.default_lot_size = default_lot_size
        self.is_connected = False
        self._symbol_cache: Dict[str, str] = {}
        self.last_connect_attempt = 0.0

    def connect(self) -> bool:
        """
        Initializes MT5 terminal and connects to demo account with safe error handling.
        Re-uses existing MT5 session automatically if terminal is already running and logged in.
        """
        init_kwargs = {}
        if self.path:
            init_kwargs["path"] = self.path

        try:
            if not mt5.initialize(**init_kwargs):
                err_code, err_msg = mt5.last_error()
                logger.error(f"MT5 initialize failed: {err_code} - {err_msg}")
                self.is_connected = False
                return False

            account_info = mt5.account_info()
            acc_num = None
            if self.account is not None:
                try:
                    acc_num = int(self.account)
                except (ValueError, TypeError):
                    acc_num = None

            # 1. If terminal is already running & authorized to an active account in MT5 GUI, use active account!
            if account_info is not None:
                if acc_num is None or acc_num == 123456 or account_info.login == acc_num:
                    logger.info(f"Connected to active MT5 terminal session. Account: {account_info.login}, Server: {account_info.server}")
                    self.is_connected = True
                    return True

            # 2. If specific non-test account credentials were provided, log in
            if acc_num and self.password and self.server and acc_num != 123456:
                authorized = mt5.login(
                    login=acc_num,
                    password=str(self.password),
                    server=str(self.server),
                )
                if authorized:
                    save_credentials(acc_num, str(self.password), str(self.server))
                    logger.info(f"MT5 Logged in successfully to account {acc_num}")
                    self.is_connected = True
                    return True
                else:
                    err_code, err_msg = mt5.last_error()
                    logger.error(f"MT5 login failed for account {acc_num}: {err_code} - {err_msg}")

            # 3. Fallback check: if already connected to any session
            account_info = mt5.account_info()
            if account_info is not None:
                logger.info(f"Using active MT5 session. Account: {account_info.login}, Server: {account_info.server}")
                self.is_connected = True
                return True

        except Exception as e:
            logger.error(f"Exception during MT5 connection attempt: {e}")

        self.is_connected = False
        return False

    def ensure_connected(self, max_retries: int = 3) -> bool:
        """
        Ensures MT5 terminal is connected. Attempts auto-reconnect if connection drops.
        """
        if self.is_connected:
            account_info = mt5.account_info()
            if account_info is not None:
                return True

        # Connection dropped or not initialized - attempt reconnection with rate limiting
        now = time.time()
        if now - self.last_connect_attempt < 2.0:
            return self.is_connected

        self.last_connect_attempt = now
        for attempt in range(1, max_retries + 1):
            logger.info(f"Attempting MT5 auto-reconnect ({attempt}/{max_retries})...")
            if self.connect():
                return True
            time.sleep(0.3)

        return False

    def disconnect(self) -> None:
        """Shuts down MT5 terminal connection."""
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            logger.info("MT5 Connection shutdown.")

    def resolve_symbol(self, symbol: str) -> str:
        """
        Auto-discovers exact symbol name used by user's broker in MT5.
        Maps generic names like EURUSD, XAUUSD, BTCUSD to broker-specific symbols like GOLD, EURUSD.m, etc.
        """
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        if not self.is_connected:
            if not self.connect():
                return symbol

        # Check exact symbol select
        if mt5.symbol_select(symbol, True):
            self._symbol_cache[symbol] = symbol
            return symbol

        # Search broker symbol database
        all_symbols = mt5.symbols_get()
        if all_symbols:
            s_upper = symbol.upper()

            # 1. Exact match case-insensitive
            for s in all_symbols:
                if s.name.upper() == s_upper:
                    mt5.symbol_select(s.name, True)
                    self._symbol_cache[symbol] = s.name
                    return s.name

            # 2. Known alias mappings
            alias_map = {
                "XAUUSD": ["GOLD", "XAU", "GOLD.M", "XAUUSD.M"],
                "XAGUSD": ["SILVER", "XAG", "SILVER.M"],
                "BTCUSD": ["BITCOIN", "BTC", "BTCUSD.M", "BTC/USD"],
                "EURUSD": ["EURUSD", "EURUSD.M", "EURUSD_I"],
                "GBPUSD": ["GBPUSD", "GBPUSD.M", "GBPUSD_I"],
            }
            candidates = alias_map.get(s_upper, [s_upper])
            for cand in candidates:
                for s in all_symbols:
                    if cand in s.name.upper():
                        mt5.symbol_select(s.name, True)
                        logger.info(f"Resolved symbol '{symbol}' -> '{s.name}' on MT5 Broker.")
                        self._symbol_cache[symbol] = s.name
                        return s.name

        self._symbol_cache[symbol] = symbol
        return symbol

    def fetch_latest_candles(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> pd.DataFrame:
        """
        Fetches historical and real-time OHLCV candle rates from MT5 terminal.
        """
        if not self.is_connected:
            if not self.connect():
                raise ConnectionError("MT5 is not connected.")

        resolved_symbol = self.resolve_symbol(symbol)
        tf_enum = TIMEFRAME_MAP.get(timeframe.lower(), mt5.TIMEFRAME_M5)

        # Select symbol in Market Watch
        mt5.symbol_select(resolved_symbol, True)

        rates = mt5.copy_rates_from_pos(resolved_symbol, tf_enum, 0, limit)
        if rates is None or len(rates) == 0:
            # Fallback retry with copy_rates_from
            rates = mt5.copy_rates_from(resolved_symbol, tf_enum, datetime.now(), limit)

        if rates is None or len(rates) == 0:
            err_code, err_msg = mt5.last_error()
            logger.error(f"Failed to fetch rates for {symbol} (resolved: {resolved_symbol}): {err_code} - {err_msg}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df[["time", "open", "high", "low", "close", "volume"]]

    def fetch_account_balance(self) -> Tuple[float, float, float]:
        """
        Retrieves real-time account balances from MT5.
        Returns: (equity, free_margin, used_margin)
        """
        if not self.is_connected:
            if not self.connect():
                return 10000.0, 10000.0, 0.0

        account_info = mt5.account_info()
        if account_info is None:
            return 10000.0, 10000.0, 0.0

        equity = float(account_info.equity)
        free_margin = float(account_info.margin_free)
        used_margin = float(account_info.margin)
        return equity, free_margin, used_margin

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches open live positions from MT5.
        """
        if not self.is_connected:
            if not self.connect():
                return []

        pos_kwargs = {}
        if symbol:
            resolved_symbol = self.resolve_symbol(symbol)
            pos_kwargs["symbol"] = resolved_symbol

        positions = mt5.positions_get(**pos_kwargs)
        if positions is None:
            return []

        results = []
        for pos in positions:
            results.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "time": pd.to_datetime(pos.time, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            })
        return results

    fetch_open_positions = get_open_positions

    def execute_order(
        self,
        symbol: str,
        action: int,
        amount: float,
        current_price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        lot_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes a market order on MT5 Demo account with explicit SL and TP prices.
        action: 1 (BUY), 2 (SELL), 0 (HOLD)
        """
        if action == 0:
            return {"status": "HOLD", "side": "NONE", "units": 0.0, "price": current_price}

        if not self.is_connected:
            if not self.connect():
                return {"status": "FAILED", "reason": "MT5 Not Connected"}

        resolved_symbol = self.resolve_symbol(symbol)
        mt5.symbol_select(resolved_symbol, True)

        symbol_info = mt5.symbol_info(resolved_symbol)
        if symbol_info is None:
            return {"status": "FAILED", "reason": f"Symbol info for {resolved_symbol} not found"}

        lots = lot_size if lot_size is not None else self.default_lot_size
        lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

        order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(resolved_symbol)
        price = tick.ask if (action == 1 and tick) else (tick.bid if tick else current_price)

        # Determine filling mode
        filling_mode = mt5.ORDER_FILLING_IOC
        if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK
        elif symbol_info.filling_mode & mt5.ORDER_FILLING_IOC:
            filling_mode = mt5.ORDER_FILLING_IOC
        elif symbol_info.filling_mode & mt5.ORDER_FILLING_RETURN:
            filling_mode = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": resolved_symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": float(stop_loss),
            "tp": float(take_profit),
            "deviation": 20,
            "magic": 202609,
            "comment": "SelfTrade-RL AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result is None:
            err_code, err_msg = mt5.last_error()
            return {"status": "FAILED", "reason": f"order_send failed: {err_code} - {err_msg}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "status": "REJECTED",
                "retcode": result.retcode,
                "reason": f"MT5 Retcode: {result.retcode} ({result.comment})",
            }

        logger.info(f"MT5 Order Executed Successfully! Ticket: {result.order}, Symbol: {resolved_symbol}, Side: {'BUY' if action == 1 else 'SELL'}, Price: {result.price}, SL: {stop_loss}, TP: {take_profit}")
        return {
            "status": "FILLED",
            "ticket": result.order,
            "side": "BUY" if action == 1 else "SELL",
            "units": lots,
            "price": result.price,
            "sl": stop_loss,
            "tp": take_profit,
        }

    def modify_position_sltp(self, symbol: str, ticket: int, stop_loss: float, take_profit: float) -> bool:
        """Modifies SL/TP for an active MT5 position."""
        if not self.is_connected:
            if not self.connect():
                return False
        resolved_symbol = self.resolve_symbol(symbol)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": resolved_symbol,
            "sl": float(stop_loss),
            "tp": float(take_profit),
        }
        result = mt5.order_send(request)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"MT5 Position {ticket} ({resolved_symbol}) SL/TP modified: SL=${stop_loss:,.2f}, TP=${take_profit:,.2f}")
            return True
        return False
