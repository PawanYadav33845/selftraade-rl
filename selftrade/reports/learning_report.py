"""
Session Learning Report Generator Module for SelfTrade-RL.
Generates structured Markdown & Text performance reports upon trading session closure,
summarizing trade statistics, Win Rate, Profit Factor, PnL breakdown by symbol, and PPO self-learning model evolution.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("SelfTrade.LearningReport")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")


class SessionLearningReportGenerator:
    """
    Generates and persists session learning reports when SelfTrade-RL closes or stops trading.
    """

    def __init__(self, reports_dir: str = REPORTS_DIR) -> None:
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report(
        self,
        session_start_time: datetime,
        session_end_time: datetime,
        execution_mode: str,
        active_symbols: List[str],
        initial_capital: float,
        final_capital: float,
        trades_history: List[Dict[str, Any]],
        model_versions_trained: List[str],
        active_model_version: str,
    ) -> Dict[str, Any]:
        """
        Calculates session metrics and formats structured report metadata.
        """
        duration_sec = (session_end_time - session_start_time).total_seconds()
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)
        seconds = int(duration_sec % 60)
        duration_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

        net_pnl_usd = final_capital - initial_capital
        net_pnl_pct = (net_pnl_usd / (initial_capital + 1e-8)) * 100.0

        total_trades = len(trades_history)
        winning_trades = [t for t in trades_history if t.get("pnl_usd", 0.0) > 0.0]
        losing_trades = [t for t in trades_history if t.get("pnl_usd", 0.0) < 0.0]
        even_trades = [t for t in trades_history if t.get("pnl_usd", 0.0) == 0.0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = (num_wins / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t.get("pnl_usd", 0.0) for t in winning_trades)
        gross_loss = abs(sum(t.get("pnl_usd", 0.0) for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

        # Calculate Max Drawdown from equity curve if available
        equity_curve = [initial_capital]
        current_eq = initial_capital
        max_drawdown_usd = 0.0
        max_drawdown_pct = 0.0
        peak_eq = initial_capital

        for t in trades_history:
            current_eq += t.get("pnl_usd", 0.0)
            equity_curve.append(current_eq)
            if current_eq > peak_eq:
                peak_eq = current_eq
            dd = peak_eq - current_eq
            dd_pct = (dd / peak_eq) * 100.0 if peak_eq > 0 else 0.0
            if dd > max_drawdown_usd:
                max_drawdown_usd = dd
                max_drawdown_pct = dd_pct

        # Per-Symbol Performance Breakdown
        symbol_breakdown = {}
        for sym in active_symbols:
            sym_trades = [t for t in trades_history if t.get("symbol") == sym]
            sym_wins = len([t for t in sym_trades if t.get("pnl_usd", 0.0) > 0.0])
            sym_pnl = sum(t.get("pnl_usd", 0.0) for t in sym_trades)
            sym_wr = (sym_wins / len(sym_trades) * 100.0) if sym_trades else 0.0
            symbol_breakdown[sym] = {
                "trades": len(sym_trades),
                "wins": sym_wins,
                "losses": len(sym_trades) - sym_wins,
                "win_rate": sym_wr,
                "pnl_usd": sym_pnl,
            }

        report_data = {
            "session_id": f"SESS-{session_start_time.strftime('%Y%m%d-%H%M%S')}",
            "start_time": session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": session_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": duration_str,
            "execution_mode": execution_mode,
            "active_symbols": active_symbols,
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "net_pnl_usd": net_pnl_usd,
            "net_pnl_pct": net_pnl_pct,
            "total_trades": total_trades,
            "num_wins": num_wins,
            "num_losses": num_losses,
            "num_even": len(even_trades),
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "max_drawdown_usd": max_drawdown_usd,
            "max_drawdown_pct": max_drawdown_pct,
            "symbol_breakdown": symbol_breakdown,
            "active_model_version": active_model_version,
            "model_versions_trained": model_versions_trained,
        }

        # Save Markdown and Plain Text Report Files
        timestamp_slug = session_end_time.strftime("%Y%m%d_%H%M%S")
        md_filepath = os.path.join(self.reports_dir, f"session_learning_report_{timestamp_slug}.md")
        latest_md_filepath = os.path.join(self.reports_dir, "latest_learning_report.md")

        txt_filepath = os.path.join(self.reports_dir, f"session_learning_report_{timestamp_slug}.txt")
        latest_txt_filepath = os.path.join(self.reports_dir, "latest_learning_report.txt")

        md_content = self.format_markdown_report(report_data)

        try:
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(md_content)
            with open(latest_md_filepath, "w", encoding="utf-8") as f:
                f.write(md_content)

            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.write(md_content)
            with open(latest_txt_filepath, "w", encoding="utf-8") as f:
                f.write(md_content)

            logger.info(f"Generated Session Learning Report saved to {md_filepath} and {txt_filepath}")
        except Exception as e:
            logger.error(f"Failed to write learning report files: {e}")

        report_data["report_filepath"] = md_filepath
        report_data["report_txt_filepath"] = txt_filepath
        report_data["markdown_content"] = md_content
        return report_data

    def format_markdown_report(self, r: Dict[str, Any]) -> str:
        """
        Formats report metadata into clean Markdown text.
        """
        pnl_symbol = "🟢" if r["net_pnl_usd"] >= 0 else "🔴"
        lines = [
            "# 🧠 SelfTrade-RL | Session Learning & Performance Report",
            "",
            f"**Session ID:** `{r['session_id']}`  ",
            f"**Execution Mode:** `{r['execution_mode']}`  ",
            f"**Start Time:** `{r['start_time']}` | **End Time:** `{r['end_time']}`  ",
            f"**Total Session Duration:** `{r['duration']}`  ",
            f"**Active Model Version:** `{r['active_model_version']}`  ",
            "",
            "---",
            "",
            "## 📊 Financial Summary",
            "",
            f"- **Starting Capital:** `${r['initial_capital']:,.2f}`",
            f"- **Final Capital:** `${r['final_capital']:,.2f}`",
            f"- **Net Session PnL:** {pnl_symbol} **`${r['net_pnl_usd']:+,.2f}` ({r['net_pnl_pct']:+.2f}%)**",
            f"- **Profit Factor:** **`{r['profit_factor']:.2f}`** (Gross Wins: `${r['gross_profit']:,.2f}` / Gross Losses: `${r['gross_loss']:,.2f}`)",
            f"- **Max Drawdown:** `${r['max_drawdown_usd']:,.2f}` (`{r['max_drawdown_pct']:.2f}%`)",
            "",
            "---",
            "",
            "## 🎯 Trade Performance Metrics",
            "",
            f"- **Total Executed Trades:** `{r['total_trades']}`",
            f"- **Winning Trades:** `{r['num_wins']}` 🟢 | **Losing Trades:** `{r['num_losses']}` 🔴 | **Breakeven Trades:** `{r['num_even']}` ⚪",
            f"- **Session Win Rate:** **`{r['win_rate']:.1f}%`**",
            "",
            "### 📈 Per-Asset Performance Breakdown",
            "",
            "| Asset Symbol | Total Trades | Wins | Losses | Win Rate % | Net PnL ($) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for sym, data in r["symbol_breakdown"].items():
            pnl_str = f"${data['pnl_usd']:+,.2f}"
            lines.append(f"| **{sym}** | {data['trades']} | {data['wins']} | {data['losses']} | {data['win_rate']:.1f}% | {pnl_str} |")

        lines.extend([
            "",
            "---",
            "",
            "## 🤖 Continuous PPO Self-Learning Evolution",
            "",
            f"- **Online Fine-Tuned Candidate Models Saved:** `{len(r['model_versions_trained'])}`",
        ])

        if r["model_versions_trained"]:
            for ver in r["model_versions_trained"]:
                lines.append(f"  - 🧠 Model Checkpoint Fine-Tuned: `{ver}`")
        else:
            lines.append("  - *No new candidate models fine-tuned during this session.*")

        lines.extend([
            "",
            "---",
            "*Report auto-generated by SelfTrade-RL Continuous Learning Engine upon session termination.*",
        ])

        return "\n".join(lines)
