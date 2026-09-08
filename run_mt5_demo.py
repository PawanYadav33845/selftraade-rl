"""
MetaTrader 5 (MT5) Demo Live Trading Runner for SelfTrade-RL.

Run this script to connect SelfTrade-RL directly to your MetaTrader 5 Demo Account on Windows.

Usage:
  python run_mt5_demo.py --account 12345678 --password mypassword --server MetaQuotes-Demo --symbol EURUSD
"""

import argparse
import sys
import time
import logging

from selftrade.live.mt5_adapter import MT5BrokerAdapter
from selftrade.analysis.pattern_engine import PatternTrendEngine
from selftrade.reasoning.trade_reasoner import TradeReasoner
from selftrade.agent.ppo_agent import PPOAgent
from selftrade.data.ingestion import DataIngestion
from selftrade.journal.trade_journal import TradeJournal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SelfTrade.MT5Runner")


def run_mt5_live_trading(
    account: int,
    password: str,
    server: str,
    symbol: str = "EURUSD",
    timeframe: str = "5m",
    lot_size: float = 0.01,
    loop_interval_sec: float = 5.0,
):
    print("\n" + "=" * 70)
    print(f"🚀 SelfTrade-RL MetaTrader 5 (MT5) Demo Live Trader")
    print(f"   Account: {account} | Server: {server} | Symbol: {symbol} | TF: {timeframe}")
    print("=" * 70 + "\n")

    # 1. Initialize MT5 Adapter
    adapter = MT5BrokerAdapter(
        account=account,
        password=password,
        server=server,
        default_lot_size=lot_size,
    )

    print("Connecting to MetaTrader 5 Terminal...")
    if not adapter.connect():
        print("❌ Failed to connect to MetaTrader 5. Please ensure MT5 terminal is installed, open, and credentials are correct.")
        sys.exit(1)

    print("✅ Connected to MT5 Demo Account successfully!\n")

    # 2. Initialize Engines
    pattern_engine = PatternTrendEngine()
    trade_reasoner = TradeReasoner(min_confluence_threshold=0.60, target_risk_reward=2.0)
    agent = PPOAgent()
    ingestion = DataIngestion()
    trade_journal = TradeJournal()

    equity, free_margin, used_margin = adapter.fetch_account_balance()
    print(f"💰 Account Equity: ${equity:.2f} | Free Margin: ${free_margin:.2f} | Used Margin: ${used_margin:.2f}\n")

    step = 0
    try:
        while True:
            step += 1
            # Fetch latest 100 candles from MT5
            df = adapter.fetch_latest_candles(symbol=symbol, timeframe=timeframe, limit=100)
            if df.empty or len(df) < 20:
                print(f"[{step}] Waiting for candle data for {symbol}...")
                time.sleep(loop_interval_sec)
                continue

            # Compute features & pattern analysis
            df_feat = ingestion.prepare_data(df)
            market_struct = pattern_engine.analyze_chart(df)

            curr_price = float(df["close"].iloc[-1])
            row = df_feat.iloc[-1]
            state = ingestion.extract_market_state(row)
            curr_atr = float(row.get("atr_norm", 0.01)) * curr_price

            # Agent Action Selection
            proposed_action = agent.select_action(state, deterministic=True)

            # Evaluate Reasoning Engine (Confluence & TP/SL calculation)
            decision = trade_reasoner.evaluate_trade(
                proposed_action=proposed_action,
                current_price=curr_price,
                atr=curr_atr,
                market_structure=market_struct,
                df=df,
            )

            action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}[decision.approved_action]
            print(f"[{step}] Price: {curr_price:.4f} | Session: {decision.market_session_name} | Trend: {market_struct.trend.value} | Signal: {action_name} (Conf: {decision.confluence_score:.2f})")

            # If action approved (BUY/SELL), execute order on MT5 Demo!
            if decision.approved_action != 0:
                print(f"⚡ EXECUTING DEMO ORDER: {action_name} {lot_size} lots on {symbol}")
                print(f"   SL Target: ${decision.stop_loss_price:.4f} | TP Target: ${decision.take_profit_price:.4f} (R:R 1:{decision.risk_reward_ratio:.1f})")

                order_res = adapter.execute_order(
                    symbol=symbol,
                    action=decision.approved_action,
                    amount=0.0,
                    current_price=curr_price,
                    stop_loss=decision.stop_loss_price,
                    take_profit=decision.take_profit_price,
                    lot_size=lot_size,
                )
                print(f"   Order Result: {order_res}\n")

                # Log trade in persistent trade journal
                trade_journal.log_trade(
                    symbol=symbol,
                    action="BUY" if decision.approved_action == 1 else "SELL",
                    entry_price=curr_price,
                    stop_loss=decision.stop_loss_price,
                    take_profit=decision.take_profit_price,
                    units=lot_size,
                    confluence_score=decision.confluence_score,
                    pattern_name=market_struct.candlestick_pattern.value,
                    trend_name=market_struct.trend.value,
                    session_name=decision.market_session_name,
                    notes=decision.reasoning_summary,
                )

            time.sleep(loop_interval_sec)

    except KeyboardInterrupt:
        print("\n🛑 Stopping MT5 Demo Trader...")
    finally:
        adapter.disconnect()
        print("👋 MT5 Disconnected. Live loop terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SelfTrade-RL MetaTrader 5 Demo Live Trader")
    parser.add_argument("--account", type=int, help="MT5 Demo Account Number")
    parser.add_argument("--password", type=str, help="MT5 Demo Account Password")
    parser.add_argument("--server", type=str, default="MetaQuotes-Demo", help="MT5 Server Name")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading Symbol (EURUSD, XAUUSD, BTCUSD)")
    parser.add_argument("--lot-size", type=float, default=0.01, help="Demo Order Lot Size")
    parser.add_argument("--interval", type=float, default=5.0, help="Loop Refresh Interval (seconds)")

    args = parser.parse_args()

    # If args missing, prompt user interactively
    account = args.account
    password = args.password
    server = args.server

    if not account or not password:
        print("--- MetaTrader 5 Credentials ---")
        if not account:
            acc_str = input("Enter MT5 Demo Account Number: ").strip()
            account = int(acc_str) if acc_str else None
        if not password:
            password = input("Enter MT5 Demo Password: ").strip()
        if not server:
            srv_str = input("Enter MT5 Server Name (default MetaQuotes-Demo): ").strip()
            server = srv_str if srv_str else "MetaQuotes-Demo"

    run_mt5_live_trading(
        account=account,
        password=password,
        server=server,
        symbol=args.symbol,
        lot_size=args.lot_size,
        loop_interval_sec=args.interval,
    )
