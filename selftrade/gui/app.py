"""
Overhauled Modern Desktop GUI Control Center for SelfTrade-RL.
Features Multi-Symbol Concurrent Live Trading, 2x3 Matplotlib Subplot Grid View,
Active Positions Table, Trade Journal Diary, Infographics Panel, and Emergency Force Stop.
"""

from typing import Optional, Dict, Any, List
import threading
import time
import os
import numpy as np
import pandas as pd
from tkinter import ttk

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from selftrade.live.multi_symbol_manager import MultiSymbolLiveManager
from selftrade.live.mt5_adapter import load_saved_credentials, save_credentials
from selftrade.journal.trade_journal import TradeJournal
from selftrade.sessions.market_sessions import MarketSessionsEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SelfTradeModernGUI(ctk.CTk):
    """
    Overhauled Control Center Desktop App with Real-Time 2x3 Multi-Asset Charts & Infographics.
    """

    ALL_SYMBOLS = ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "BTCUSD"]

    def __init__(self) -> None:
        super().__init__()

        self.title("SelfTrade-RL | Multi-Asset Autonomous Trading Control Center & Infographics")
        self.geometry("1440x920")
        self.minsize(1024, 720)

        # Simulation & Engine State
        self.is_running = False
        self.is_force_stopped = False
        self.execution_mode = "Simulated Paper"
        self.target_rr_ratio = 2.0

        # Multi-Symbol Manager
        self.multi_manager: Optional[MultiSymbolLiveManager] = None

        # Saved MT5 Credentials
        saved_cred = load_saved_credentials()
        acc_val = str(saved_cred.get("account", "")) if saved_cred.get("account") else ""
        pwd_val = str(saved_cred.get("password", "")) if saved_cred.get("password") else ""
        srv_val = str(saved_cred.get("server", "MetaQuotes-Demo"))

        self.mt5_account_var = ctk.StringVar(value=acc_val)
        self.mt5_password_var = ctk.StringVar(value=pwd_val)
        self.mt5_server_var = ctk.StringVar(value=srv_val)

        # Active Symbol Checkbox Vars
        self.symbol_vars: Dict[str, ctk.BooleanVar] = {
            sym: ctk.BooleanVar(value=True) for sym in self.ALL_SYMBOLS
        }

        # Persistent Trade Journal
        self.journal = TradeJournal(journal_filepath="journal/trade_journal.json")

        # Telemetry & Equity History
        self.portfolio_history: List[float] = []
        self.refresh_timer_id: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        # 1. Header Frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.pack(fill="x", padx=15, pady=(15, 10))

        title_label = ctk.CTkLabel(
            self.header_frame,
            text="⚡ SELFTRADE-RL (MULTI-ASSET)",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        title_label.pack(side="left", padx=(15, 10), pady=12)

        # Symbol Multi-Select Checkboxes
        chk_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        chk_frame.pack(side="left", padx=5)

        for sym in self.ALL_SYMBOLS:
            chk = ctk.CTkCheckBox(
                chk_frame,
                text=sym,
                variable=self.symbol_vars[sym],
                width=75,
                font=ctk.CTkFont(size=11, weight="bold"),
            )
            chk.pack(side="left", padx=3)

        # Mode Selector (Simulated vs MT5 Demo)
        ctk.CTkLabel(self.header_frame, text="Mode:", font=ctk.CTkFont(weight="bold", size=11)).pack(side="left", padx=(8, 2))
        self.mode_dropdown = ctk.CTkOptionMenu(
            self.header_frame,
            values=["Simulated Paper", "MT5 Demo Live"],
            command=self._on_mode_changed,
            width=125,
        )
        self.mode_dropdown.set(self.execution_mode)
        self.mode_dropdown.pack(side="left", padx=3)

        self.btn_mt5_login = ctk.CTkButton(
            self.header_frame,
            text="⚙️ MT5 Login",
            width=85,
            fg_color="#30363D",
            hover_color="#484F58",
            font=ctk.CTkFont(weight="bold", size=11),
            command=self._show_mt5_login_dialog,
        )
        self.btn_mt5_login.pack(side="left", padx=3)

        # Risk:Reward Ratio Selector
        ctk.CTkLabel(self.header_frame, text="R:R:", font=ctk.CTkFont(weight="bold", size=11)).pack(side="left", padx=(6, 2))
        self.rr_dropdown = ctk.CTkOptionMenu(
            self.header_frame,
            values=["1:1.5", "1:2.0", "1:2.5", "1:3.0"],
            command=self._on_rr_changed,
            width=80,
        )
        self.rr_dropdown.set("1:2.0")
        self.rr_dropdown.pack(side="left", padx=3)

        # Control Buttons
        self.btn_start = ctk.CTkButton(
            self.header_frame,
            text="▶ START BOT",
            width=95,
            fg_color="#2EA043",
            hover_color="#238636",
            font=ctk.CTkFont(weight="bold"),
            command=self.start_bot,
        )
        self.btn_start.pack(side="left", padx=6)

        self.btn_pause = ctk.CTkButton(
            self.header_frame,
            text="⏸ PAUSE",
            width=80,
            fg_color="#D29922",
            hover_color="#9E6A03",
            font=ctk.CTkFont(weight="bold"),
            command=self.pause_bot,
        )
        self.btn_pause.pack(side="left", padx=4)

        # EMERGENCY FORCE STOP BUTTON
        self.btn_force_stop = ctk.CTkButton(
            self.header_frame,
            text="🚨 FORCE STOP",
            width=110,
            fg_color="#DA3633",
            hover_color="#B62324",
            font=ctk.CTkFont(weight="bold", size=12),
            command=self.force_stop_bot,
        )
        self.btn_force_stop.pack(side="right", padx=15)

        # 2. Main Content Split
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=15, pady=5)

        # Left Column: Status Cards & Log Terminal
        self.left_column = ctk.CTkFrame(self.main_container, width=420)
        self.left_column.pack(side="left", fill="both", padx=(0, 10), pady=0)

        self._build_dashboard_cards()
        self._build_log_terminal()

        # Right Column: Tabview (Charts & Infographics | Active Positions | Trade Journal)
        self.right_column = ctk.CTkFrame(self.main_container)
        self.right_column.pack(side="right", fill="both", expand=True, padx=0, pady=0)

        self.tabview = ctk.CTkTabview(self.right_column)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_charts = self.tabview.add("📊 Multi-Asset Charts & Infographics")
        self.tab_positions = self.tabview.add("💼 Active Open Positions")
        self.tab_journal = self.tabview.add("📓 Trade Journal & Diary")

        self._build_charts_tab()
        self._build_positions_tab()
        self._build_journal_tab()

    def _build_dashboard_cards(self) -> None:
        cards_frame = ctk.CTkFrame(self.left_column, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=10)

        # Card 1: Portfolio Net Worth
        self.card_val = ctk.CTkFrame(cards_frame, corner_radius=8)
        self.card_val.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(self.card_val, text="PORTFOLIO NET WORTH", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=8, pady=(6, 0))
        self.lbl_net_worth = ctk.CTkLabel(self.card_val, text="$10,000.00 (+0.00%)", font=ctk.CTkFont(size=15, weight="bold"), text_color="#3FB950")
        self.lbl_net_worth.pack(anchor="w", padx=8, pady=(0, 6))

        # Card 2: Action & Active Model
        self.card_action = ctk.CTkFrame(cards_frame, corner_radius=8)
        self.card_action.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(self.card_action, text="ACTION & MODEL", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=8, pady=(6, 0))
        self.lbl_action_model = ctk.CTkLabel(self.card_action, text="IDLE | v1.0_multi_active", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_action_model.pack(anchor="w", padx=8, pady=(0, 6))

        # Card 3: Active Assets Count
        self.card_reasoning = ctk.CTkFrame(cards_frame, corner_radius=8)
        self.card_reasoning.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(self.card_reasoning, text="ACTIVE ASSETS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=8, pady=(6, 0))
        self.lbl_confluence = ctk.CTkLabel(self.card_reasoning, text="5 Assets Parallel", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_confluence.pack(anchor="w", padx=8, pady=(0, 6))

        # Card 4: Market Opening Session
        self.card_session = ctk.CTkFrame(cards_frame, corner_radius=8)
        self.card_session.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(self.card_session, text="MARKET SESSION", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=8, pady=(6, 0))
        sess_info = MarketSessionsEngine.get_session_info()
        self.lbl_session = ctk.CTkLabel(self.card_session, text=f"{sess_info.session_name[:16]} ({sess_info.volatility_multiplier:.1f}x)", font=ctk.CTkFont(size=13, weight="bold"), text_color="#D29922")
        self.lbl_session.pack(anchor="w", padx=8, pady=(0, 6))

        # Card 5: Dynamic Target Geometry
        self.card_tpsl = ctk.CTkFrame(self.left_column, corner_radius=8)
        self.card_tpsl.pack(fill="x", padx=15, pady=4)
        ctk.CTkLabel(self.card_tpsl, text="DYNAMIC DUAL TARGET GEOMETRY", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=10, pady=(6, 0))
        self.lbl_tpsl = ctk.CTkLabel(self.card_tpsl, text="Multi-Asset TP/SL Active (R:R 1:2.0)", font=ctk.CTkFont(size=13, weight="bold"), text_color="#58A6FF")
        self.lbl_tpsl.pack(anchor="w", padx=10, pady=(0, 6))

        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)

    def _build_log_terminal(self) -> None:
        log_frame = ctk.CTkFrame(self.left_column, fg_color="transparent")
        log_frame.pack(fill="both", expand=True, padx=10, pady=8)

        ctk.CTkLabel(log_frame, text="📋 LIVE BOT LOG & THOUGHT TERMINAL", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(0, 4))

        self.log_textbox = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#0D1117",
            text_color="#C9D1D9",
            wrap="word",
        )
        self.log_textbox.pack(fill="both", expand=True)

    def _build_charts_tab(self) -> None:
        plt.style.use("dark_background")

        self.fig = plt.Figure(figsize=(10, 6.5), facecolor="#161B22")
        self.axes_grid = self.fig.subplots(2, 3)

        # Color mapping for assets
        self.symbol_colors = {
            "XAUUSD": "#FFD700",  # Gold
            "XAGUSD": "#C0C0C0",  # Silver
            "EURUSD": "#58A6FF",  # Blue
            "GBPUSD": "#BC8CFF",  # Purple
            "BTCUSD": "#F7931A",  # Orange
        }

        # Map symbol to 2x3 grid positions
        self.symbol_ax_map = {
            "XAUUSD": self.axes_grid[0, 0],
            "XAGUSD": self.axes_grid[0, 1],
            "EURUSD": self.axes_grid[0, 2],
            "GBPUSD": self.axes_grid[1, 0],
            "BTCUSD": self.axes_grid[1, 1],
        }
        self.ax_portfolio = self.axes_grid[1, 2]

        for ax in self.axes_grid.flat:
            ax.set_facecolor("#0D1117")

        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_charts)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_positions_tab(self) -> None:
        ctk.CTkLabel(self.tab_positions, text="💼 ACTIVE OPEN TRADING POSITIONS (MULTI-ASSET)", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#0D1117", foreground="#C9D1D9", fieldbackground="#0D1117", rowheight=25)
        style.configure("Treeview.Heading", background="#161B22", foreground="#FFFFFF", font=("Arial", 10, "bold"))
        style.map("Treeview", background=[("selected", "#1F6FEB")])

        cols = ("Symbol", "Side", "Entry Price", "Current Price", "Unrealized PnL", "TP Level", "SL Level", "R:R")
        self.tree_positions = ttk.Treeview(self.tab_positions, columns=cols, show="headings", height=15)
        for col in cols:
            self.tree_positions.heading(col, text=col)
            self.tree_positions.column(col, anchor="center", width=110)

        self.tree_positions.pack(fill="both", expand=True, padx=15, pady=10)

    def _build_journal_tab(self) -> None:
        ctk.CTkLabel(self.tab_journal, text="📓 HISTORICAL TRADE DIARY & CONDITION LOG", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)

        cols = ("Timestamp", "Symbol", "Side", "Entry", "Exit", "PnL ($)", "Confluence", "Pattern", "Risk Verdict")
        self.tree_journal = ttk.Treeview(self.tab_journal, columns=cols, show="headings", height=15)
        for col in cols:
            self.tree_journal.heading(col, text=col)
            self.tree_journal.column(col, anchor="center", width=105)

        self.tree_journal.pack(fill="both", expand=True, padx=15, pady=10)
        self.refresh_journal_table()

    def refresh_journal_table(self) -> None:
        for item in self.tree_journal.get_children():
            self.tree_journal.delete(item)

        # Reload entries from file
        self.journal.load()

        for e in reversed(self.journal.entries[-50:]):
            pnl_str = f"${e.pnl_usd:+.2f} ({e.pnl_pct:+.2f}%)"
            self.tree_journal.insert(
                "",
                "end",
                values=(
                    e.timestamp[:16],
                    e.symbol,
                    e.side,
                    f"${e.entry_price:,.2f}",
                    f"${e.exit_price:,.2f}",
                    pnl_str,
                    f"{e.confluence_score:.2f}",
                    e.candlestick_pattern,
                    e.risk_verdict[:15],
                ),
            )

    def append_log(self, text: str) -> None:
        self.log_textbox.insert("end", text + "\n")
        self.log_textbox.see("end")

    def _on_rr_changed(self, choice: str) -> None:
        val = float(choice.replace("1:", ""))
        self.target_rr_ratio = val
        if self.multi_manager:
            self.multi_manager.target_rr_ratio = val
        self.append_log(f"--> Updated Risk-to-Reward Ratio setting to 1:{val:.1f}")

    def _on_mode_changed(self, choice: str) -> None:
        self.execution_mode = choice
        self.append_log(f"--> Switched execution mode to: {choice}")
        if choice == "MT5 Demo Live":
            acc = self.mt5_account_var.get()
            if acc:
                self.append_log(f"--> Found saved MT5 credentials for Account: {acc} (Server: {self.mt5_server_var.get()})")
            else:
                self._show_mt5_login_dialog()

    def _show_mt5_login_dialog(self) -> None:
        dialog = ctk.CTkInputDialog(text="Enter MT5 Demo Account Number:", title="MetaTrader 5 Credentials")
        acc = dialog.get_input()
        if acc:
            self.mt5_account_var.set(acc)
            pwd_dialog = ctk.CTkInputDialog(text="Enter MT5 Demo Password:", title="MetaTrader 5 Credentials")
            pwd = pwd_dialog.get_input()
            if pwd:
                self.mt5_password_var.set(pwd)
                srv_dialog = ctk.CTkInputDialog(text="Enter MT5 Server (default MetaQuotes-Demo):", title="MetaTrader 5 Credentials")
                srv = srv_dialog.get_input()
                if srv:
                    self.mt5_server_var.set(srv)
                save_credentials(int(acc), pwd, self.mt5_server_var.get())
                self.append_log(f"--> Saved MT5 Demo Credentials for Account: {acc}, Server: {self.mt5_server_var.get()}")

    def start_bot(self) -> None:
        if self.is_running:
            return

        selected_symbols = [sym for sym, var in self.symbol_vars.items() if var.get()]
        if not selected_symbols:
            selected_symbols = ["XAUUSD"]
            self.symbol_vars["XAUUSD"].set(True)

        self.is_running = True
        self.is_force_stopped = False
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal")

        acc_val = self.mt5_account_var.get()
        mt5_creds = {
            "account": int(acc_val) if acc_val.isdigit() and int(acc_val) != 123456 else None,
            "password": self.mt5_password_var.get() or None,
            "server": self.mt5_server_var.get() or "MetaQuotes-Demo",
        }

        self.multi_manager = MultiSymbolLiveManager(
            target_rr_ratio=self.target_rr_ratio,
            journal_path="journal/trade_journal.json",
            mt5_credentials=mt5_creds,
        )

        success = self.multi_manager.start_trading(symbols=selected_symbols, mode=self.execution_mode)
        if not success:
            self.append_log("❌ Failed to launch MultiSymbolLiveManager. Check MT5 connection.")
            self.is_running = False
            self.btn_start.configure(state="normal")
            self.btn_pause.configure(state="disabled")
            return

        self.lbl_action_model.configure(text=f"RUNNING | {self.multi_manager.agent.version_tag}")
        self.lbl_confluence.configure(text=f"{len(selected_symbols)} Assets Parallel")
        self.append_log(f"🚀 MULTI-ASSET BOT LAUNCHED for assets: {selected_symbols} (Mode: {self.execution_mode}, R:R 1:{self.target_rr_ratio:.1f})...")

        self.portfolio_history.clear()
        self._schedule_gui_refresh()

    def pause_bot(self) -> None:
        self.is_running = False
        if self.multi_manager:
            self.multi_manager.stop_trading()
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.lbl_action_model.configure(text="PAUSED | v1.0_multi_active")
        self.append_log("⏸ MULTI-ASSET BOT PAUSED by user.")

    def force_stop_bot(self) -> None:
        """
        EMERGENCY FORCE STOP: Immediately terminates execution, halts threads, overrides signals.
        """
        self.is_running = False
        self.is_force_stopped = True
        if self.multi_manager:
            self.multi_manager.force_stop_all()

        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.lbl_action_model.configure(text="HALTED | EMERGENCY_STOP")
        self.append_log("🚨 EMERGENCY FORCE STOP TRIGGERED! All multi-asset threads halted instantly.")

    def _schedule_gui_refresh(self) -> None:
        if self.is_running and not self.is_force_stopped:
            self._update_gui_tick()
            self.after(500, self._schedule_gui_refresh)

    def _update_gui_tick(self) -> None:
        if not self.multi_manager:
            return

        overall_equity = self.multi_manager.overall_portfolio_value
        ret_pct = ((overall_equity - 10000.0) / 10000.0) * 100.0
        val_color = "#3FB950" if ret_pct >= 0 else "#F85149"
        self.lbl_net_worth.configure(text=f"${overall_equity:,.2f} ({ret_pct:+.2f}%)", text_color=val_color)

        self.portfolio_history.append(overall_equity)

        # Update Active Positions Treeview
        for item in self.tree_positions.get_children():
            self.tree_positions.delete(item)

        with self.multi_manager.lock:
            for sym, telem in self.multi_manager.telemetry.items():
                if telem.open_position > 0:
                    entry_p = telem.position_entry_price
                    curr_p = telem.last_price
                    unrealized_pnl = (curr_p - entry_p) * telem.open_position
                    unrealized_str = f"${unrealized_pnl:+.2f}"
                    dec = telem.last_decision
                    tp_val = dec.take_profit_price if dec else 0.0
                    sl_val = dec.stop_loss_price if dec else 0.0

                    self.tree_positions.insert(
                        "",
                        "end",
                        values=(
                            sym,
                            "BUY",
                            f"${entry_p:,.2f}",
                            f"${curr_p:,.2f}",
                            unrealized_str,
                            f"${tp_val:,.2f}",
                            f"${sl_val:,.2f}",
                            f"1:{self.target_rr_ratio:.1f}",
                        ),
                    )

        # Refresh Trade Journal Table
        self.refresh_journal_table()

        # Update 2x3 Matplotlib Charts Grid
        try:
            with self.multi_manager.lock:
                for sym, ax in self.symbol_ax_map.items():
                    ax.clear()
                    ax.set_facecolor("#0D1117")

                    telem = self.multi_manager.telemetry.get(sym)
                    if telem and len(telem.price_history) > 1:
                        prices = telem.price_history[-50:]
                        steps = list(range(len(prices)))
                        color = self.symbol_colors.get(sym, "#58A6FF")

                        ax.plot(steps, prices, color=color, linewidth=1.5, label=f"{sym}")
                        if telem.support_level > 0:
                            ax.axhline(telem.support_level, color="#3FB950", linestyle="--", alpha=0.5)
                            ax.axhline(telem.resistance_level, color="#F85149", linestyle="--", alpha=0.5)
                        ax.set_title(f"{sym}: ${telem.last_price:,.2f}", color="white", fontsize=9, fontweight="bold")
                    else:
                        ax.set_title(f"{sym} (Standby)", color="#8B949E", fontsize=9)

            # Subplot 6: Aggregated Portfolio Equity & Win/Loss Infographic
            self.ax_portfolio.clear()
            self.ax_portfolio.set_facecolor("#0D1117")

            if len(self.portfolio_history) > 1:
                eq_steps = list(range(len(self.portfolio_history[-50:])))
                self.ax_portfolio.plot(eq_steps, self.portfolio_history[-50:], color="#3FB950" if ret_pct >= 0 else "#F85149", linewidth=1.8)
                self.ax_portfolio.set_title(f"Aggregated Equity (${overall_equity:,.2f})", color="white", fontsize=9, fontweight="bold")
            else:
                self.ax_portfolio.set_title("Aggregated Equity", color="#8B949E", fontsize=9)

            self.fig.canvas.draw_idle()
        except Exception:
            pass


def main() -> None:
    app = SelfTradeModernGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
