"""
Multi-Symbol Simulation Runner for SelfTrade-RL.
Executes self-improving RL trading agent across XAUUSD (Gold), XAGUSD (Silver), GBPUSD, EURUSD, and BTC/USDT.
"""

import sys
import logging
from typing import List, Dict, Any

from selftrade.data.generator import MarketDataGenerator
from selftrade.system import SelfTradeSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiSymbolRunner")


def run_symbol_simulation(symbol: str, initial_capital: float = 10000.0) -> Dict[str, Any]:
    logger.info(f"\n{'='*80}\n RUNNING SELFTRADE-RL FOR SYMBOL: {symbol}\n{'='*80}")

    data_gen = MarketDataGenerator(seed=42)
    hist_df = data_gen.generate_symbol_data(symbol=symbol, num_steps=800, regime_shift_freq=200)
    live_df = data_gen.generate_symbol_data(symbol=symbol, num_steps=1000, regime_shift_freq=250)

    system = SelfTradeSystem(
        df=live_df,
        initial_capital=initial_capital,
        max_stop_loss_pct=0.02,
        max_daily_drawdown_pct=0.05,
        sharpe_threshold=0.5,
        periodic_interval=200,
        eval_test_window=80,
    )

    system.seed_historical_buffer(hist_df)
    metrics = system.run_simulation(max_steps=800)

    final_val = metrics[-1]["portfolio_value"] if metrics else initial_capital
    ret_pct = ((final_val - initial_capital) / initial_capital) * 100.0
    risk_overrides = sum(1 for m in metrics if m["risk_overridden"])
    circuit_breakers = sum(1 for m in metrics if m["circuit_breaker_active"])
    buys = sum(1 for m in metrics if m["safe_action"] == 1)
    sells = sum(1 for m in metrics if m["safe_action"] == 2)
    holds = sum(1 for m in metrics if m["safe_action"] == 0)

    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_value": final_val,
        "return_pct": ret_pct,
        "buys": buys,
        "sells": sells,
        "holds": holds,
        "risk_overrides": risk_overrides,
        "circuit_breakers": circuit_breakers,
        "final_model": metrics[-1]["active_model_version"] if metrics else "v1.0_active",
    }


def main():
    symbols = ["XAUUSD", "XAGUSD", "GBPUSD", "EURUSD", "BTCUSD"]
    results = []

    print("=" * 85)
    print("        SELFTRADE-RL: MULTI-SYMBOL (GOLD, SILVER, FOREX, CRYPTO) BENCHMARK      ")
    print("=" * 85)

    for sym in symbols:
        res = run_symbol_simulation(sym)
        results.append(res)

    print("\n" + "=" * 85)
    print("                         MULTI-SYMBOL PORTFOLIO PERFORMANCE                     ")
    print("=" * 85)
    print(f"{'SYMBOL':<10} | {'INIT CAP':<10} | {'FINAL VAL':<12} | {'RETURN (%)':<11} | {'BUYS/SELLS':<12} | {'RISK OVERRIDES':<14}")
    print("-" * 85)

    for r in results:
        trades_str = f"{r['buys']}/{r['sells']}"
        print(
            f"{r['symbol']:<10} | ${r['initial_capital']:<9,.0f} | ${r['final_value']:<11,.2f} | "
            f"{r['return_pct']:>+9.2f}% | {trades_str:<12} | {r['risk_overrides']:<14}"
        )

    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
