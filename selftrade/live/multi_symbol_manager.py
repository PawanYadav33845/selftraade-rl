"""
Multi-Symbol Concurrent Live Manager for SelfTrade-RL.
Manages concurrent execution threads across Gold (XAUUSD), Silver (XAGUSD), Forex (EURUSD, GBPUSD), and Crypto (BTCUSD)
with Adaptive Holding Logic (Scalp Quick Profits vs Big Trend Runners), Break-Even Protection, and Dynamic Trailing Stops.
"""

import threading
import time
import logging
import gc
from datetime import datetime
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
from selftrade.agent.buffer import LiveExperienceBuffer, HistoricalBuffer
from selftrade.continuous_learning.monitor import PerformanceMonitor
from selftrade.continuous_learning.self_improver import SelfImprovingEngine
from selftrade.journal.trade_journal import TradeJournal
from selftrade.reports.learning_report import SessionLearningReportGenerator
from selftrade.live.mt5_adapter import MT5BrokerAdapter
from selftrade.live.broker import BaseBrokerAdapter, PaperBrokerAdapter

logger = logging.getLogger("SelfTrade.MultiSymbolManager")


class PositionRecord:
    """Dataclass holding an individual trade position's state."""

    def __init__(
        self,
        pos_id: str,
        units: float,
        entry_price: float,
        sl: float,
        tp: float,
        holding_mode: str = "SCALP_QUICK",
        be_trigger_price: float = 0.0,
    ) -> None:
        self.pos_id = pos_id
        self.units = units
        self.entry_price = entry_price
        self.current_sl = sl
        self.current_tp = tp
        self.holding_mode = holding_mode
        self.be_trigger_price = be_trigger_price
        self.is_be_active = False


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
        self.open_positions: List[PositionRecord] = []
        self.is_active: bool = True

    @property
    def open_position(self) -> float:
        return sum(p.units for p in self.open_positions)

    @open_position.setter
    def open_position(self, value: float) -> None:
        if value <= 0:
            self.open_positions.clear()

    @property
    def position_entry_price(self) -> float:
        if self.open_positions:
            return self.open_positions[-1].entry_price
        return 0.0

    @position_entry_price.setter
    def position_entry_price(self, value: float) -> None:
        pass

    @property
    def current_sl(self) -> float:
        if self.open_positions:
            return self.open_positions[-1].current_sl
        return 0.0

    @current_sl.setter
    def current_sl(self, value: float) -> None:
        if self.open_positions:
            self.open_positions[-1].current_sl = value

    @property
    def current_tp(self) -> float:
        if self.open_positions:
            return self.open_positions[-1].current_tp
        return 0.0

    @current_tp.setter
    def current_tp(self, value: float) -> None:
        if self.open_positions:
            self.open_positions[-1].current_tp = value

    @property
    def holding_mode(self) -> str:
        if self.open_positions:
            return self.open_positions[-1].holding_mode
        return "SCALP_QUICK"

    @holding_mode.setter
    def holding_mode(self, value: str) -> None:
        if self.open_positions:
            self.open_positions[-1].holding_mode = value

    @property
    def is_be_active(self) -> bool:
        if self.open_positions:
            return any(p.is_be_active for p in self.open_positions)
        return False

    @is_be_active.setter
    def is_be_active(self, value: bool) -> None:
        if self.open_positions:
            self.open_positions[-1].is_be_active = value


class MultiSymbolLiveManager:
    """
    Orchestrates parallel trading execution loops for multiple assets (Crypto, Forex, Commodities).
    """

    def __init__(
        self,
        target_rr_ratio: float = 2.0,
        journal_path: str = "journal/trade_journal.json",
        mt5_credentials: Optional[Dict[str, Any]] = None,
        max_trades_per_symbol: int = 2,
    ) -> None:
        self.target_rr_ratio = target_rr_ratio
        self.journal = TradeJournal(journal_filepath=journal_path)
        self.mt5_credentials = mt5_credentials or {}
        self.max_trades_per_symbol = max_trades_per_symbol

        self.active_symbols: List[str] = []
        self.execution_mode: str = "Simulated Paper"
        self.is_running: bool = False
        self.is_force_stopped: bool = False

        self.telemetry: Dict[str, SymbolTelemetry] = {}
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self.training_lock = threading.Lock()

        # Shared Agents & Engines
        self.agent = PPOAgent(version_tag="v1.0_multi_active")
        self.pattern_engine = PatternTrendEngine()
        self.reasoner = TradeReasoner(target_risk_reward=target_rr_ratio)
        self.risk_engine = DeterministicRiskEngine()
        self.ingestion = DataIngestion()
        self.mt5_adapter: Optional[MT5BrokerAdapter] = None
        self.report_generator = SessionLearningReportGenerator()

        # Continuous Learning Online Self-Improvement Engine
        self.live_buffer = LiveExperienceBuffer(capacity=5000)
        self.historical_buffer = HistoricalBuffer()
        self.monitor = PerformanceMonitor(periodic_interval=30, sharpe_window=20)
        self.self_improver = SelfImprovingEngine(
            agent=self.agent,
            live_buffer=self.live_buffer,
            historical_buffer=self.historical_buffer,
            monitor=self.monitor,
            fine_tune_epochs=3,
            batch_size=32,
        )

        self.overall_portfolio_value: float = 10000.0
        self.initial_session_capital: float = 10000.0
        self.trade_counter: int = 1

        self.session_start_time: Optional[datetime] = None
        self.session_end_time: Optional[datetime] = None
        self.trained_models_list: List[str] = []
        self.latest_report_data: Optional[Dict[str, Any]] = None

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
        self.session_start_time = datetime.now()
        self.session_end_time = None
        self.trained_models_list.clear()
        self.latest_report_data = None
        self.initial_session_capital = self.overall_portfolio_value

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

    def stop_trading(self) -> Optional[Dict[str, Any]]:
        """
        Gracefully stops all active symbol worker loops and generates a Session Learning Report.
        """
        if not self.is_running and self.session_end_time is not None:
            return self.latest_report_data

        self.is_running = False
        self.session_end_time = datetime.now()

        if self.mt5_adapter:
            self.mt5_adapter.disconnect()

        logger.info("MultiSymbolLiveManager stopped cleanly.")
        return self._generate_learning_report()

    def force_stop_all(self) -> Optional[Dict[str, Any]]:
        """
        EMERGENCY FORCE STOP: Immediately halts all symbol worker loops and generates report.
        """
        self.is_running = False
        self.is_force_stopped = True
        self.session_end_time = datetime.now()

        if self.mt5_adapter:
            self.mt5_adapter.disconnect()

        logger.warning("🚨 EMERGENCY FORCE STOP: All multi-asset trading threads halted.")
        return self._generate_learning_report()

    def _generate_learning_report(self) -> Dict[str, Any]:
        """
        Compiles trade statistics and generates a formatted Markdown & Text Session Learning Report.
        """
        start_time = self.session_start_time or datetime.now()
        end_time = self.session_end_time or datetime.now()
        trades = self.journal.get_all_trades()

        report_data = self.report_generator.generate_report(
            session_start_time=start_time,
            session_end_time=end_time,
            execution_mode=self.execution_mode,
            active_symbols=self.active_symbols,
            initial_capital=self.initial_session_capital,
            final_capital=self.overall_portfolio_value,
            trades_history=trades,
            model_versions_trained=list(self.trained_models_list),
            active_model_version=self.agent.version_tag,
        )
        self.latest_report_data = report_data
        return report_data

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
                    start_idx = max(0, step - 100)
                    current_df = df_source.iloc[start_idx : step + 2]
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

                    action_net, log_prob, val_est = self.agent.select_action(obs, deterministic=False)
                    self.reasoner.target_risk_reward = self.target_rr_ratio
                    decision = self.reasoner.evaluate_trade(action_net, current_price, atr, market_struct, current_df, symbol=symbol)

                    total_open_trades_all = sum(len(t.open_positions) for t in self.telemetry.values())
                    mock_info = {
                        "close_price": current_price,
                        "portfolio_value": self.overall_portfolio_value,
                        "position": telem.open_position,
                        "position_entry_price": telem.position_entry_price,
                        "cash": self.overall_portfolio_value if len(telem.open_positions) == 0 else self.overall_portfolio_value * 0.5,
                        "total_open_trades": total_open_trades_all,
                    }
                    risk_verdict = self.risk_engine.evaluate_action(decision.approved_action, mock_info)
                    safe_action = risk_verdict.safe_action

                    # Update Telemetry State & Trailing SL / Break-Even across active positions
                    with self.lock:
                        telem.last_price = current_price
                        telem.support_level = market_struct.support_level
                        telem.resistance_level = market_struct.resistance_level
                        telem.last_decision = decision
                        telem.last_market_struct = market_struct
                        telem.last_session_info = session_info
                        telem.step_history.append(step)
                        telem.price_history.append(current_price)
                        if len(telem.step_history) > 200:
                            telem.step_history = telem.step_history[-200:]
                            telem.price_history = telem.price_history[-200:]

                        closed_positions = []
                        step_reward = 0.0

                        # Evaluate Break-Even & Dynamic Trailing Stop for Open Positions
                        for pos in list(telem.open_positions):
                            new_sl, modified, reason_str = self.risk_engine.update_trailing_stop_and_be(
                                symbol=symbol,
                                side="BUY",
                                current_price=current_price,
                                entry_price=pos.entry_price,
                                current_sl=pos.current_sl,
                                current_tp=pos.current_tp,
                                be_trigger_price=pos.be_trigger_price,
                                atr=atr,
                                holding_mode=pos.holding_mode,
                            )
                            if modified:
                                pos.current_sl = new_sl
                                if reason_str == "BREAK_EVEN_PROTECTION_ACTIVATED":
                                    pos.is_be_active = True
                                logger.info(f"🛡️ [{symbol}] ({pos.pos_id}) {reason_str}: SL updated to ${new_sl:,.2f}")

                            # Check explicit TP or SL hits or Agent Exit Signal in Paper Mode
                            pos_exit = False
                            pos_exit_price = current_price
                            pos_exit_reason = "AGENT_SELL_SIGNAL"

                            if pos.current_tp > 0.0 and current_price >= pos.current_tp:
                                pos_exit = True
                                pos_exit_price = pos.current_tp
                                pos_exit_reason = f"TAKE_PROFIT_HIT (R:R 1:{decision.risk_reward_ratio:.1f})"
                                step_reward += 1.0
                            elif pos.current_sl > 0.0 and current_price <= pos.current_sl:
                                pos_exit = True
                                pos_exit_price = pos.current_sl
                                pos_exit_reason = f"STOP_LOSS_HIT (R:R 1:{decision.risk_reward_ratio:.1f})"
                                step_reward -= 0.8
                            elif safe_action == 2:
                                pos_exit = True
                                pos_exit_price = current_price
                                pos_exit_reason = "AGENT_SELL_SIGNAL"

                            if pos_exit:
                                pnl_usd = (pos_exit_price - pos.entry_price) * pos.units
                                pnl_pct = ((pos_exit_price - pos.entry_price) / pos.entry_price) * 100.0
                                self.overall_portfolio_value += pnl_usd
                                step_reward += 0.5 if pnl_usd > 0 else -0.5

                                self.journal.log_trade(
                                    trade_id=pos.pos_id,
                                    timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    symbol=symbol,
                                    side="BUY",
                                    entry_price=pos.entry_price,
                                    exit_price=pos_exit_price,
                                    units=pos.units,
                                    pnl_usd=pnl_usd,
                                    pnl_pct=pnl_pct,
                                    take_profit=pos.current_tp,
                                    stop_loss=pos.current_sl,
                                    risk_reward_ratio=decision.risk_reward_ratio,
                                    confluence_score=decision.confluence_score,
                                    trend_condition=market_struct.trend.value,
                                    candlestick_pattern=market_struct.candlestick_pattern.value,
                                    reasoning_summary=f"[{pos_exit_reason}] {decision.reasoning_summary}",
                                    risk_verdict=risk_verdict.override_reason,
                                    model_version=self.agent.version_tag,
                                )
                                self.trade_counter += 1
                                closed_positions.append(pos)

                        for pos in closed_positions:
                            telem.open_positions.remove(pos)

                        # Check BUY ENTRY if open positions count < max_trades_per_symbol (at least 2 trades per category)
                        if safe_action == 1 and len(telem.open_positions) < self.max_trades_per_symbol:
                            new_pos = PositionRecord(
                                pos_id=f"TRD-M-{self.trade_counter:04d}",
                                units=1000.0 / current_price,
                                entry_price=current_price,
                                sl=decision.stop_loss_price,
                                tp=decision.take_profit_price,
                                holding_mode=decision.holding_mode,
                                be_trigger_price=decision.break_even_trigger_price,
                            )
                            telem.open_positions.append(new_pos)
                            step_reward += 0.1
                            logger.info(f"📈 [{symbol}] NEW CATEGORY POSITION OPENED ({len(telem.open_positions)}/{self.max_trades_per_symbol}): ${current_price:,.2f}")

                        # Record live experience transition for continuous self-learning
                        self.self_improver.record_transition(
                            state=obs,
                            action=safe_action,
                            reward=step_reward,
                            next_state=obs,
                            done=len(closed_positions) > 0,
                            log_prob=log_prob,
                            value=val_est,
                        )

                        # Trigger Online GPU Fine-Tuning Check
                        if step > 0 and step % 25 == 0:
                            with self.training_lock:
                                triggered, candidate_agent, desc = self.self_improver.check_and_fine_tune(
                                    reward=step_reward,
                                    market_atr_norm=float(row.get("atr_norm", 0.01)),
                                )
                                if triggered and candidate_agent:
                                    self.agent = candidate_agent
                                    if self.agent.version_tag not in self.trained_models_list:
                                        self.trained_models_list.append(self.agent.version_tag)
                                    logger.info(f"🧠 [PPO Continuous Learning] {desc}")
                                gc.collect()

                    step += 1
                    time.sleep(0.20)

            else:  # MT5 Demo Live Mode
                while self.is_running and not self.is_force_stopped:
                    if self.mt5_adapter:
                        self.mt5_adapter.ensure_connected()

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
                    open_pos_count = len(mt5_positions) if mt5_positions else 0
                    open_pos_entry = mt5_positions[0]["price_open"] if mt5_positions else 0.0

                    pos_ratio = 1.0 if open_pos_vol > 0 else 0.0
                    cash_ratio = 0.0 if open_pos_vol > 0 else 1.0

                    with self.lock:
                        telem = self.telemetry[symbol]
                        if len(telem.price_history) == 0 and not rates_df.empty:
                            telem.price_history.extend(rates_df["close"].tolist()[-50:])

                    obs = self.ingestion.extract_market_state(row, pos_ratio=pos_ratio, cash_ratio=cash_ratio)
                    atr = float(row.get("atr_norm", 0.01)) * current_price

                    action_net, _, _ = self.agent.select_action(obs, deterministic=False)
                    self.reasoner.target_risk_reward = self.target_rr_ratio
                    decision = self.reasoner.evaluate_trade(action_net, current_price, atr, market_struct, rates_df, symbol=symbol)

                    total_open_trades_all = sum(len(t.open_positions) for t in self.telemetry.values()) + open_pos_count
                    mock_info = {
                        "close_price": current_price,
                        "portfolio_value": equity,
                        "position": open_pos_vol,
                        "position_entry_price": open_pos_entry,
                        "cash": free_margin,
                        "total_open_trades": total_open_trades_all,
                    }
                    risk_verdict = self.risk_engine.evaluate_action(decision.approved_action, mock_info)
                    safe_action = risk_verdict.safe_action

                    # Break-Even & Dynamic Trailing Stop for MT5 open positions
                    if open_pos_count > 0 and telem.current_sl > 0.0:
                        new_sl, modified, reason_str = self.risk_engine.update_trailing_stop_and_be(
                            symbol=symbol,
                            side="BUY",
                            current_price=current_price,
                            entry_price=open_pos_entry,
                            current_sl=telem.current_sl,
                            current_tp=telem.current_tp,
                            be_trigger_price=decision.break_even_trigger_price,
                            atr=atr,
                            holding_mode=decision.holding_mode,
                        )
                        if modified:
                            telem.current_sl = new_sl
                            if reason_str == "BREAK_EVEN_PROTECTION_ACTIVATED":
                                telem.is_be_active = True
                            if mt5_positions:
                                for p in mt5_positions:
                                    self.mt5_adapter.modify_position_sltp(symbol, p["ticket"], new_sl, telem.current_tp)
                            logger.info(f"🛡️ MT5 [{symbol}] {reason_str}: Updated SL to ${new_sl:,.2f}")

                    if safe_action != 0 and open_pos_count < self.max_trades_per_symbol:
                        logger.info(f"⚡ MT5 CATEGORY ORDER [{decision.holding_mode}] ({open_pos_count+1}/{self.max_trades_per_symbol}): {safe_action} on {symbol} at ${current_price:,.2f}")
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
                            telem.current_sl = decision.stop_loss_price
                            telem.current_tp = decision.take_profit_price
                            telem.holding_mode = decision.holding_mode
                            telem.is_be_active = False

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
                                risk_reward_ratio=decision.risk_reward_ratio,
                                confluence_score=decision.confluence_score,
                                trend_condition=market_struct.trend.value,
                                candlestick_pattern=market_struct.candlestick_pattern.value,
                                reasoning_summary=f"[{decision.holding_mode}] {decision.reasoning_summary}",
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
                        if len(telem.step_history) > 200:
                            telem.step_history = telem.step_history[-200:]
                            telem.price_history = telem.price_history[-200:]

                    step += 1
                    time.sleep(3.0)

        except Exception as e:
            logger.error(f"Worker loop exception for {symbol}: {e}")
        finally:
            with self.lock:
                if symbol in self.telemetry:
                    self.telemetry[symbol].is_active = False
