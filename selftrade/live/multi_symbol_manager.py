"""
Multi-Symbol Concurrent Live Manager for SelfTrade-RL.
Manages concurrent execution threads across Gold (XAUUSD), Silver (XAGUSD), Forex (EURUSD, GBPUSD), and Crypto (BTCUSD).
"""

import threading
import time
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from selftrade.data.generator import MarketDataGenerator
from selftrade.data.ingestion import DataIngestion
from selftrade.analysis.pattern_engine import PatternTrendEngine, MarketStructure
from selftrade.sessions.market_sessions import MarketSessionsEngine, MarketSessionInfo
from selftrade.reasoning.trade_reasoner import TradeReasoner, ReasonedTradeDecision
from selftrade.risk.risk_engine import DeterministicRiskEngine, RiskEvaluationResult
from selftrade.agent.ppo_agent import PPOAgent
from selftrade.journal.trade_journal import TradeJournal
from selftrade.live.mt5_adapter import MT5BrokerAdapter
from selftrade.live.broker import BaseBrokerAdapter, PaperBrokerAdapter

logger = logging.getLogger("SelfTrade.MultiSymbolManager")


class SymbolTelemetry:
    """Dataclass holding thread-safe real-time telemetry for an individual symbol."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.price_history: List[float] = []
        self.step_history: List[int] = []
        self.last_price: float = 0.0
        self.support_level: float = 0.0
        self.resistance_level: float = 0.0
        self.last_decision: Optional[ReasonedTradeDecision] = None
        self.last_market_struct: Optional[MarketStructure] = None
        self.last_session_info: Optional[MarketSessionInfo] = None
        self.open_position: float = 0.0
        self.position_entry_price: float = 0.0
        self.is_active: bool = True


class MultiSymbolLiveManager:
    """
    Orchestrates parallel trading execution loops for multiple assets (Crypto, Forex, Commodities).
    """

    def __init__(
        self,
        target_rr_ratio: float = 2.0,
        journal_path: str = "journal/trade_journal.json",
        mt5_credentials: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.target_rr_ratio = target_rr_ratio
        self.journal = TradeJournal(journal_filepath=journal_path)
        self.mt5_credentials = mt5_credentials or {}

        self.active_symbols: List[str] = []
        self.execution_mode: str = "Simulated Paper"
        self.is_running: bool = False
        self.is_force_stopped: bool = False

        self.telemetry: Dict[str, SymbolTelemetry] = {}
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

        # Shared Agents & Engines
        self.agent = PPOAgent(version_tag="v1.0_multi_active")
        self.pattern_engine = PatternTrendEngine()
        self.reasoner = TradeReasoner(target_risk_reward=target_rr_ratio)
        self.risk_engine = DeterministicRiskEngine()
        self.ingestion = DataIngestion()
        self.mt5_adapter: Optional[MT5BrokerAdapter] = None

        self.overall_portfolio_value: float = 10000.0
        self.trade_counter: int = 1

    def start_trading(self, symbols: List[str], mode: str = "Simulated Paper") -> bool:
        """
        Launches concurrent worker threads for each selected asset.
        """
        if self.is_running:
            return False

        self.active_symbols = symbols
        self.execution_mode = mode
        self.is_running = True
        self.is_force_stopped = False

        self.telemetry.clear()
        self.worker_threads.clear()

        for sym in symbols:
            self.telemetry[sym] = SymbolTelemetry(symbol=sym)

        if mode == "MT5 Demo Live":
            acc = self.mt5_credentials.get("account")
            pwd = self.mt5_credentials.get("password")
            srv = self.mt5_credentials.get("server", "MetaQuotes-Demo")

            self.mt5_adapter = MT5BrokerAdapter(account=acc, password=pwd, server=srv)
            if not self.mt5_adapter.connect():
                logger.error("Failed to connect MT5 Broker Adapter in MultiSymbolLiveManager.")
                self.is_running = False
                return False

        # Launch thread per symbol
        for sym in symbols:
            thread = threading.Thread(
                target=self._symbol_worker_loop,
                args=(sym,),
                daemon=True,
            )
            self.worker_threads[sym] = thread
            thread.start()

        logger.info(f"MultiSymbolLiveManager started for symbols: {symbols} in {mode} mode.")
        return True

    def stop_trading(self) -> None:
        """
        Gracefully stops all active symbol worker loops.
        """
        self.is_running = False
        if self.mt5_adapter:
            self.mt5_adapter.disconnect()
        logger.info("MultiSymbolLiveManager stopped cleanly.")

    def force_stop_all(self) -> None:
        """
        EMERGENCY FORCE STOP: Immediately halts all symbol worker loops and cancels trades.
        """
        self.is_running = False
        self.is_force_stopped = True
        if self.mt5_adapter:
            self.mt5_adapter.disconnect()
        logger.warning("🚨 EMERGENCY FORCE STOP: All multi-asset trading threads halted.")

    def _symbol_worker_loop(self, symbol: str) -> None:
        """
        Execution loop for an individual asset thread.
        """
        try:
            step = 0
            last_buy_price = 0.0

            if self.execution_mode == "Simulated Paper":
                seed_val = (int(time.time() * 1000) + hash(symbol)) % 10000
                data_gen = MarketDataGenerator(seed=seed_val)
                df_source = data_gen.generate_symbol_data(symbol=symbol, num_steps=800)

                while self.is_running and not self.is_force_stopped and step < len(df_source) - 1:
                    current_df = df_source.iloc[: step + 2]
                    current_price = float(current_df["close"].iloc[-1])

                    market_struct = self.pattern_engine.analyze_chart(current_df)
                    session_info = MarketSessionsEngine.get_session_info()

                    df_feat = self.ingestion.prepare_data(current_df)
                    row = df_feat.iloc[-1]

                    with self.lock:
                        telem = self.telemetry[symbol]
                        pos_ratio = 1.0 if telem.open_position > 0 else 0.0
                        cash_ratio = 0.0 if telem.open_position > 0 else 1.0

                    obs = self.ingestion.extract_market_state(row, pos_ratio=pos_ratio, cash_ratio=cash_ratio)
                    atr = float(row.get("atr_norm", 0.01)) * current_price

                    action_net, _, _ = self.agent.select_action(obs, deterministic=False)
                    self.reasoner.target_risk_reward = self.target_rr_ratio
                    decision = self.reasoner.evaluate_trade(action_net, current_price, atr, market_struct, current_df)

                    mock_info = {
                        "close_price": current_price,
                        "portfolio_value": self.overall_portfolio_value,
                        "position": telem.open_position,
                        "position_entry_price": last_buy_price,
                        "cash": self.overall_portfolio_value if telem.open_position == 0 else 0.0,
                    }
                    risk_verdict = self.risk_engine.evaluate_action(decision.approved_action, mock_info)
                    safe_action = risk_verdict.safe_action

                    # Update Telemetry State
                    with self.lock:
                        telem.last_price = current_price
                        telem.support_level = market_struct.support_level
                        telem.resistance_level = market_struct.resistance_level
                        telem.last_decision = decision
                        telem.last_market_struct = market_struct
                        telem.last_session_info = session_info
                        telem.step_history.append(step)
                        telem.price_history.append(current_price)

                        if safe_action == 1 and telem.open_position == 0:  # BUY
                            telem.open_position = 1000.0 / current_price
                            telem.position_entry_price = current_price
                            last_buy_price = current_price

                        elif safe_action == 2 and telem.open_position > 0:  # SELL
                            pnl_usd = (current_price - last_buy_price) * telem.open_position
                            pnl_pct = ((current_price - last_buy_price) / last_buy_price) * 100.0
                            self.overall_portfolio_value += pnl_usd

                            self.journal.log_trade(
                                trade_id=f"TRD-M-{self.trade_counter:04d}",
                                timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                symbol=symbol,
                                side="BUY",
                                entry_price=last_buy_price,
                                exit_price=current_price,
                                units=telem.open_position,
                                pnl_usd=pnl_usd,
                                pnl_pct=pnl_pct,
                                take_profit=decision.take_profit_price,
                                stop_loss=decision.stop_loss_price,
                                risk_reward_ratio=self.target_rr_ratio,
                                confluence_score=decision.confluence_score,
                                trend_condition=market_struct.trend.value,
                                candlestick_pattern=market_struct.candlestick_pattern.value,
                                reasoner_summary=decision.reasoning_summary,
                                risk_verdict=risk_verdict.override_reason,
                                model_version=self.agent.version_tag,
                            )
                            self.trade_counter += 1
                            telem.open_position = 0.0
                            telem.position_entry_price = 0.0

                    step += 1
                    time.sleep(0.08)

            else:  # MT5 Demo Live Mode
                while self.is_running and not self.is_force_stopped:
                    rates_df = self.mt5_adapter.fetch_latest_candles(symbol, timeframe="5m", limit=100)
                    if rates_df.empty or len(rates_df) < 10:
                        time.sleep(3.0)
                        continue

                    current_price = float(rates_df["close"].iloc[-1])
                    equity, free_margin, used_margin = self.mt5_adapter.fetch_account_balance()
                    self.overall_portfolio_value = equity

                    market_struct = self.pattern_engine.analyze_chart(rates_df)
                    session_info = MarketSessionsEngine.get_session_info()

                    df_feat = self.ingestion.prepare_data(rates_df)
                    row = df_feat.iloc[-1]

                    mt5_positions = self.mt5_adapter.fetch_open_positions(symbol)
                    open_pos_vol = sum(p["volume"] for p in mt5_positions) if mt5_positions else 0.0
                    open_pos_entry = mt5_positions[0]["price_open"] if mt5_positions else 0.0

                    pos_ratio = 1.0 if open_pos_vol > 0 else 0.0
                    cash_ratio = 0.0 if open_pos_vol > 0 else 1.0

                    with self.lock:
                        telem = self.telemetry[symbol]
                        if len(telem.price_history) == 0 and not rates_df.empty:
                            telem.price_history.extend(rates_df["close"].tolist()[-50:])
                        telem.open_position = open_pos_vol
                        telem.position_entry_price = open_pos_entry

                    obs = self.ingestion.extract_market_state(row, pos_ratio=pos_ratio, cash_ratio=cash_ratio)
                    atr = float(row.get("atr_norm", 0.01)) * current_price

                    action_net, _, _ = self.agent.select_action(obs, deterministic=False)
                    self.reasoner.target_risk_reward = self.target_rr_ratio
                    decision = self.reasoner.evaluate_trade(action_net, current_price, atr, market_struct, rates_df)

                    mock_info = {
                        "close_price": current_price,
                        "portfolio_value": equity,
                        "position": open_pos_vol,
                        "position_entry_price": open_pos_entry,
                        "cash": free_margin,
                    }
                    risk_verdict = self.risk_engine.evaluate_action(decision.approved_action, mock_info)
                    safe_action = risk_verdict.safe_action

                    if safe_action != 0 and open_pos_vol == 0:
                        logger.info(f"⚡ MT5 MULTI-ASSET ORDER: {safe_action} on {symbol} at ${current_price:,.2f}")
                        order_res = self.mt5_adapter.execute_order(
                            symbol=symbol,
                            action=safe_action,
                            amount=0.0,
                            current_price=current_price,
                            stop_loss=decision.stop_loss_price,
                            take_profit=decision.take_profit_price,
                        )
                        logger.info(f"   MT5 Order Status for {symbol}: {order_res}")

                        if order_res.get("status") == "SUCCESS" or order_res.get("ticket"):
                            self.journal.log_trade(
                                trade_id=f"TRD-MT5-{self.trade_counter:04d}",
                                timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                symbol=symbol,
                                side="BUY" if safe_action == 1 else "SELL",
                                entry_price=current_price,
                                exit_price=0.0,
                                units=order_res.get("units", 0.01),
                                pnl_usd=0.0,
                                pnl_pct=0.0,
                                take_profit=decision.take_profit_price,
                                stop_loss=decision.stop_loss_price,
                                risk_reward_ratio=self.target_rr_ratio,
                                confluence_score=decision.confluence_score,
                                trend_condition=market_struct.trend.value,
                                candlestick_pattern=market_struct.candlestick_pattern.value,
                                reasoner_summary=decision.reasoning_summary,
                                risk_verdict=risk_verdict.override_reason,
                                model_version=self.agent.version_tag,
                            )
                            self.trade_counter += 1

                    with self.lock:
                        telem.last_price = current_price
                        telem.support_level = market_struct.support_level
                        telem.resistance_level = market_struct.resistance_level
                        telem.last_decision = decision
                        telem.last_market_struct = market_struct
                        telem.last_session_info = session_info
                        telem.step_history.append(step)
                        telem.price_history.append(current_price)

                    step += 1
                    time.sleep(3.0)

        except Exception as e:
            logger.error(f"Worker loop exception for {symbol}: {e}")
        finally:
            with self.lock:
                if symbol in self.telemetry:
                    self.telemetry[symbol].is_active = False
