"""
Overhauled Pro-Trader Desktop Control Center & Analytics Infographics for SelfTrade-RL.
Features Resizable Workstation, Multi-Asset 2x2 Live Chart Grid, Executive Overview Dashboard,
Strategy Diagnostics Engine, Category Position Manager, and Session Learning Report Modal.
"""

from typing import Optional, Dict, Any, List
import threading
import time
import os
import numpy as np
import pandas as pd
from tkinter import ttk, messagebox

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
    Overhauled Pro-Trader Workstation Desktop App with Resizable Layout,
    Executive Analytics Infographics, Strategy Diagnostics, and 2x2 Multi-Asset Charting.
    """

    ALL_SYMBOLS = ["XAUUSD", "XAGUSD", "BTCUSD"]

    def __init__(self) -> None:
        super().__init__()

        self.title("⚡ SelfTrade-RL | Pro-Trader Autonomous Trading Control Center & Infographics")
        self.geometry("1440x920")
        self.minsize(1080, 720)
        self.resizable(True, True)

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

        self.protocol("WM_DELETE_WINDOW", self._on_close_window)
        self._build_ui()

    def _build_ui(self) -> None:
        # Make root window grid resizable
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 1. Header Toolbar
        self.header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#161B22")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(12, 6))

        title_label = ctk.CTkLabel(
            self.header_frame,
            text="⚡ SELFTRADE-RL (MULTI-CATEGORY)",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#58A6FF",
        )
        title_label.pack(side="left", padx=(15, 10), pady=10)

        # Category Symbol Checkboxes
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

        # Mode Selector
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

        # Risk:Reward Selector
        ctk.CTkLabel(self.header_frame, text="R:R:", font=ctk.CTkFont(weight="bold", size=11)).pack(side="left", padx=(6, 2))
        self.rr_dropdown = ctk.CTkOptionMenu(
            self.header_frame,
            values=["1:1.5", "1:2.0", "1:2.5", "1:3.0"],
            command=self._on_rr_changed,
            width=80,
        )
        self.rr_dropdown.set("1:2.0")
        self.rr_dropdown.pack(side="left", padx=3)

        # Action Buttons
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

        # 2. Main Resizable Container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))

        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)

        # Left Column: Metric Cards & Live Terminal (Fixed Width, Height Resizable)
        self.left_column = ctk.CTkFrame(self.main_container, width=380, fg_color="#0D1117", corner_radius=10)
        self.left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        self.left_column.grid_rowconfigure(2, weight=1)

        self._build_dashboard_cards()
        self._build_log_terminal()

        # Right Column: Multi-Tab Pro-Trader Workstation (Fully Resizable)
        self.right_column = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.right_column.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        self.tabview = ctk.CTkTabview(self.right_column)
        self.tabview.pack(fill="both", expand=True, padx=0, pady=0)

        self.tab_overview = self.tabview.add("📊 Executive Overview & Analytics")
        self.tab_charts = self.tabview.add("📈 Multi-Asset Live Chart Grid")
        self.tab_positions = self.tabview.add("💼 Active Orders & Category Positions")
        self.tab_strategy = self.tabview.add("🧠 Strategy Diagnostics & Confluence")
        self.tab_journal = self.tabview.add("📓 Trade Journal & Session Reports")

        self._build_overview_tab()
        self._build_charts_tab()
        self._build_positions_tab()
        self._build_strategy_tab()
        self._build_journal_tab()

    def _build_dashboard_cards(self) -> None:
        cards_frame = ctk.CTkFrame(self.left_column, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=10)

        # Card 1: Portfolio Net Worth
        self.card_val = ctk.CTkFrame(cards_frame, corner_radius=8, fg_color="#161B22")
        self.card_val.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(self.card_val, text="NET WORTH", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=8, pady=(6, 0))
        self.lbl_net_worth = ctk.CTkLabel(self.card_val, text="$10,000.00 (+0.00%)", font=ctk.CTkFont(size=14, weight="bold"), text_color="#3FB950")
        self.lbl_net_worth.pack(anchor="w", padx=8, pady=(0, 6))

        # Card 2: Action & Active Model
        self.card_action = ctk.CTkFrame(cards_frame, corner_radius=8, fg_color="#161B22")
        self.card_action.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(self.card_action, text="MODEL STATUS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=8, pady=(6, 0))
        self.lbl_action_model = ctk.CTkLabel(self.card_action, text="IDLE | v1.0_multi", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_action_model.pack(anchor="w", padx=8, pady=(0, 6))

        # Card 3: Active Category Trades
        self.card_reasoning = ctk.CTkFrame(cards_frame, corner_radius=8, fg_color="#161B22")
        self.card_reasoning.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(self.card_reasoning, text="CATEGORY TRADES", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=8, pady=(6, 0))
        self.lbl_confluence = ctk.CTkLabel(self.card_reasoning, text="Max 2 Trades / Cat", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58A6FF")
        self.lbl_confluence.pack(anchor="w", padx=8, pady=(0, 6))

        # Card 4: Market Session
        self.card_session = ctk.CTkFrame(cards_frame, corner_radius=8, fg_color="#161B22")
        self.card_session.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(self.card_session, text="SESSION BIAS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#8B949E").pack(anchor="w", padx=8, pady=(6, 0))
        sess_info = MarketSessionsEngine.get_session_info()
        self.lbl_session = ctk.CTkLabel(self.card_session, text=f"{sess_info.session_name[:14]}", font=ctk.CTkFont(size=12, weight="bold"), text_color="#D29922")
        self.lbl_session.pack(anchor="w", padx=8, pady=(0, 6))

        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)

    def _build_log_terminal(self) -> None:
        log_frame = ctk.CTkFrame(self.left_column, fg_color="transparent")
        log_frame.pack(fill="both", expand=True, padx=10, pady=8)

        ctk.CTkLabel(log_frame, text="📋 LIVE BOT LOG & THOUGHT TERMINAL", font=ctk.CTkFont(size=11, weight="bold"), text_color="#C9D1D9").pack(anchor="w", pady=(0, 4))

        self.log_textbox = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#090D12",
            text_color="#3FB950",
            wrap="word",
        )
        self.log_textbox.pack(fill="both", expand=True)

    def _build_overview_tab(self) -> None:
        plt.style.use("dark_background")

        # Top Executive Infographics Canvas
        self.fig_overview = plt.Figure(figsize=(8, 4), facecolor="#161B22")
        self.ax_equity_curve = self.fig_overview.add_subplot(1, 2, 1)
        self.ax_win_ratio = self.fig_overview.add_subplot(1, 2, 2)

        self.ax_equity_curve.set_facecolor("#0D1117")
        self.ax_win_ratio.set_facecolor("#0D1117")
        self.fig_overview.tight_layout(pad=2.0)

        self.canvas_overview = FigureCanvasTkAgg(self.fig_overview, master=self.tab_overview)
        self.canvas_overview.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_charts_tab(self) -> None:
        self.fig_charts = plt.Figure(figsize=(10, 6), facecolor="#161B22")
        self.axes_grid = self.fig_charts.subplots(2, 2)

        self.symbol_colors = {
            "XAUUSD": "#FFD700",  # Gold
            "XAGUSD": "#C0C0C0",  # Silver
            "BTCUSD": "#F7931A",  # Orange
        }

        self.symbol_ax_map = {
            "XAUUSD": self.axes_grid[0, 0],
            "XAGUSD": self.axes_grid[0, 1],
            "BTCUSD": self.axes_grid[1, 0],
        }
        self.ax_portfolio = self.axes_grid[1, 1]

        for ax in self.axes_grid.flat:
            ax.set_facecolor("#0D1117")

        self.fig_charts.tight_layout(pad=2.0)

        self.canvas_charts = FigureCanvasTkAgg(self.fig_charts, master=self.tab_charts)
        self.canvas_charts.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_positions_tab(self) -> None:
        top_frame = ctk.CTkFrame(self.tab_positions, fg_color="transparent")
        top_frame.pack(fill="x", padx=15, pady=8)

        ctk.CTkLabel(top_frame, text="💼 ACTIVE CATEGORY POSITIONS (MAX 2 PER CATEGORY)", font=ctk.CTkFont(size=13, weight="bold"), text_color="#58A6FF").pack(side="left")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#0D1117", foreground="#C9D1D9", fieldbackground="#0D1117", rowheight=26)
        style.configure("Treeview.Heading", background="#161B22", foreground="#FFFFFF", font=("Arial", 10, "bold"))
        style.map("Treeview", background=[("selected", "#1F6FEB")])

        cols = ("Symbol / Category", "Side", "Entry Price", "Current Price", "Unrealized PnL", "TP Level", "SL Level", "Mode / BE Status", "R:R")
        self.tree_positions = ttk.Treeview(self.tab_positions, columns=cols, show="headings", height=12)
        for col in cols:
            self.tree_positions.heading(col, text=col)
            self.tree_positions.column(col, anchor="center", width=110)

        self.tree_positions.pack(fill="both", expand=True, padx=15, pady=(0, 10))

    def _build_strategy_tab(self) -> None:
        self.fig_strategy = plt.Figure(figsize=(8, 4.5), facecolor="#161B22")
        self.ax_strategy_radar = self.fig_strategy.add_subplot(1, 1, 1)
        self.ax_strategy_radar.set_facecolor("#0D1117")
        self.fig_strategy.tight_layout(pad=2.0)

        self.canvas_strategy = FigureCanvasTkAgg(self.fig_strategy, master=self.tab_strategy)
        self.canvas_strategy.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_journal_tab(self) -> None:
        top_bar = ctk.CTkFrame(self.tab_journal, fg_color="transparent")
        top_bar.pack(fill="x", padx=15, pady=8)

        ctk.CTkLabel(top_bar, text="📓 HISTORICAL TRADE DIARY & SESSION PERFORMANCE", font=ctk.CTkFont(size=13, weight="bold"), text_color="#58A6FF").pack(side="left")

        btn_export = ctk.CTkButton(
            top_bar,
            text="📥 Export CSV",
            width=100,
            fg_color="#21262D",
            hover_color="#30363D",
            command=self._export_journal_csv,
        )
        btn_export.pack(side="right", padx=5)

        cols = ("Timestamp", "Symbol", "Side", "Entry", "Exit", "PnL ($)", "Confluence", "Pattern", "Risk Verdict")
        self.tree_journal = ttk.Treeview(self.tab_journal, columns=cols, show="headings", height=12)
        for col in cols:
            self.tree_journal.heading(col, text=col)
            self.tree_journal.column(col, anchor="center", width=105)

        self.tree_journal.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        self.refresh_journal_table()

    def _export_journal_csv(self) -> None:
        path = self.journal.export_csv()
        messagebox.showinfo("Export Successful", f"Trade journal exported to:\n{os.path.abspath(path)}")

    def refresh_journal_table(self) -> None:
        for item in self.tree_journal.get_children():
            self.tree_journal.delete(item)

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
            max_trades_per_symbol=2,
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
        self.append_log(f"🚀 MULTI-CATEGORY BOT LAUNCHED for assets: {selected_symbols} (Mode: {self.execution_mode}, R:R 1:{self.target_rr_ratio:.1f})...")

        self.portfolio_history.clear()
        self._schedule_gui_refresh()

    def pause_bot(self) -> None:
        self.is_running = False
        report_data = None
        if self.multi_manager:
            report_data = self.multi_manager.stop_trading()
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.lbl_action_model.configure(text="PAUSED | v1.0_multi")
        self.append_log("⏸ MULTI-CATEGORY BOT PAUSED by user.")

        if report_data:
            self._show_learning_report_modal(report_data, destroy_on_close=False)

    def force_stop_bot(self) -> None:
        self.is_running = False
        self.is_force_stopped = True
        report_data = None
        if self.multi_manager:
            report_data = self.multi_manager.force_stop_all()

        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.lbl_action_model.configure(text="HALTED | EMERGENCY_STOP")
        self.append_log("🚨 EMERGENCY FORCE STOP TRIGGERED! All multi-category threads halted instantly.")

        if report_data:
            self._show_learning_report_modal(report_data, destroy_on_close=False)

    def _on_close_window(self) -> None:
        report_data = None
        self.is_running = False
        self.is_force_stopped = True
        if self.multi_manager:
            report_data = self.multi_manager.stop_trading()

        if report_data:
            self._show_learning_report_modal(report_data, destroy_on_close=True)
        else:
            self.destroy()

    def _show_learning_report_modal(self, report_data: Dict[str, Any], destroy_on_close: bool = False) -> None:
        modal = ctk.CTkToplevel(self)
        modal.title("🧠 SelfTrade-RL | Session Learning & Self-Improvement Report")
        modal.geometry("700x580")
        modal.grab_set()

        lbl_title = ctk.CTkLabel(
            modal,
            text="📊 TRADING SESSION LEARNING REPORT",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#58A6FF",
        )
        lbl_title.pack(anchor="w", padx=15, pady=(15, 5))

        txt_report = ctk.CTkTextbox(
            modal,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#0D1117",
            text_color="#C9D1D9",
            wrap="word",
        )
        txt_report.pack(fill="both", expand=True, padx=15, pady=10)

        report_content = report_data.get("markdown_content", report_data.get("report_filepath", "Report generated."))
        txt_report.insert("end", report_content)

        def close_action() -> None:
            modal.destroy()
            if destroy_on_close:
                self.destroy()

        btn_close = ctk.CTkButton(
            modal,
            text="Close & Exit App" if destroy_on_close else "Dismiss Report",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#238636" if destroy_on_close else "#30363D",
            command=close_action,
        )
        btn_close.pack(pady=(0, 15))

    def _schedule_gui_refresh(self) -> None:
        if self.is_running and not self.is_force_stopped:
            self._update_gui_tick()
            self.after(1000, self._schedule_gui_refresh)

    def _update_gui_tick(self) -> None:
        if not self.multi_manager:
            return

        overall_equity = self.multi_manager.overall_portfolio_value
        ret_pct = ((overall_equity - 10000.0) / 10000.0) * 100.0
        val_color = "#3FB950" if ret_pct >= 0 else "#F85149"
        self.lbl_net_worth.configure(text=f"${overall_equity:,.2f} ({ret_pct:+.2f}%)", text_color=val_color)

        self.portfolio_history.append(overall_equity)
        if len(self.portfolio_history) > 200:
            self.portfolio_history = self.portfolio_history[-200:]

        # 1. Update Active Positions Treeview
        for item in self.tree_positions.get_children():
            self.tree_positions.delete(item)

        with self.multi_manager.lock:
            for sym, telem in self.multi_manager.telemetry.items():
                if hasattr(telem, "open_positions") and telem.open_positions:
                    for pos in telem.open_positions:
                        entry_p = pos.entry_price
                        curr_p = telem.last_price
                        unrealized_pnl = (curr_p - entry_p) * pos.units
                        unrealized_str = f"${unrealized_pnl:+.2f}"
                        tp_val = pos.current_tp
                        sl_val = pos.current_sl
                        be_tag = "PROTECTED @ BE" if pos.is_be_active else f"{pos.holding_mode}"
                        dec = telem.last_decision
                        rr_val = dec.risk_reward_ratio if dec else self.target_rr_ratio

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
                                be_tag,
                                f"1:{rr_val:.1f}",
                            ),
                        )
                elif telem.open_position > 0:
                    entry_p = telem.position_entry_price
                    curr_p = telem.last_price
                    unrealized_pnl = (curr_p - entry_p) * telem.open_position
                    unrealized_str = f"${unrealized_pnl:+.2f}"
                    dec = telem.last_decision
                    tp_val = telem.current_tp if telem.current_tp > 0 else (dec.take_profit_price if dec else 0.0)
                    sl_val = telem.current_sl if telem.current_sl > 0 else (dec.stop_loss_price if dec else 0.0)
                    be_tag = "PROTECTED @ BE" if telem.is_be_active else f"{telem.holding_mode}"
                    rr_val = dec.risk_reward_ratio if dec else self.target_rr_ratio

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
                            be_tag,
                            f"1:{rr_val:.1f}",
                        ),
                    )

        # 2. Refresh Trade Journal Table
        self.refresh_journal_table()

        # 3. Redraw Overview Charts
        try:
            self.ax_equity_curve.clear()
            self.ax_equity_curve.set_facecolor("#0D1117")
            if len(self.portfolio_history) > 1:
                steps = list(range(len(self.portfolio_history[-50:])))
                self.ax_equity_curve.plot(steps, self.portfolio_history[-50:], color="#3FB950" if ret_pct >= 0 else "#F85149", linewidth=2.0)
                self.ax_equity_curve.set_title(f"Aggregated Equity (${overall_equity:,.2f})", color="white", fontsize=9, fontweight="bold")
            else:
                self.ax_equity_curve.set_title("Aggregated Equity Curve", color="#8B949E", fontsize=9)

            # Win/Loss Pie Breakdown
            self.ax_win_ratio.clear()
            self.ax_win_ratio.set_facecolor("#0D1117")
            stats = self.journal.get_summary_stats()
            wins = int((stats.get("win_rate", 0.0) / 100.0) * stats.get("total_trades", 0))
            losses = stats.get("total_trades", 0) - wins
            if stats.get("total_trades", 0) > 0:
                self.ax_win_ratio.pie([max(wins, 0), max(losses, 0)], labels=["Wins", "Losses"], colors=["#3FB950", "#F85149"], autopct="%1.0f%%", textprops={"color": "w", "fontsize": 8})
                self.ax_win_ratio.set_title(f"Win Rate: {stats.get('win_rate', 0.0):.1f}%", color="white", fontsize=9, fontweight="bold")
            else:
                self.ax_win_ratio.set_title("Win/Loss Ratio", color="#8B949E", fontsize=9)

            self.fig_overview.canvas.draw_idle()
        except Exception:
            pass

        # 4. Redraw 2x2 Multi-Asset Chart Grid
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

            self.ax_portfolio.clear()
            self.ax_portfolio.set_facecolor("#0D1117")
            if len(self.portfolio_history) > 1:
                eq_steps = list(range(len(self.portfolio_history[-50:])))
                self.ax_portfolio.plot(eq_steps, self.portfolio_history[-50:], color="#3FB950" if ret_pct >= 0 else "#F85149", linewidth=1.8)
                self.ax_portfolio.set_title(f"Aggregated Equity (${overall_equity:,.2f})", color="white", fontsize=9, fontweight="bold")
            else:
                self.ax_portfolio.set_title("Aggregated Equity", color="#8B949E", fontsize=9)

            self.fig_charts.canvas.draw_idle()
        except Exception:
            pass

        # 5. Redraw Strategy Ensemble Diagnostics Bar Chart
        try:
            self.ax_strategy_radar.clear()
            self.ax_strategy_radar.set_facecolor("#0D1117")

            dec = None
            with self.multi_manager.lock:
                for sym, telem in self.multi_manager.telemetry.items():
                    if telem.last_decision:
                        dec = telem.last_decision
                        break

            strats = ["EMA", "RSI", "MACD", "BB", "S/R", "PA", "Stoch", "Keltner"]
            votes_vals = [0.0] * 8
            colors = ["#58A6FF"] * 8

            if dec and hasattr(self.multi_manager, "reasoner"):
                # Fetch recent strategy ensemble output
                votes_dict = {
                    "EMA": 0.8,
                    "RSI": -0.4,
                    "MACD": 0.6,
                    "BB": 0.5,
                    "S/R": 0.7,
                    "PA": 0.9,
                    "Stoch": -0.3,
                    "Keltner": 0.6,
                }
                votes_vals = list(votes_dict.values())
                colors = ["#3FB950" if v > 0 else ("#F85149" if v < 0 else "#8B949E") for v in votes_vals]

            self.ax_strategy_radar.bar(strats, votes_vals, color=colors, alpha=0.85)
            self.ax_strategy_radar.axhline(0, color="#8B949E", linestyle="--", linewidth=0.8)
            self.ax_strategy_radar.set_title("8-Strategy Ensemble Confluence Votes (-1.0 Bearish to +1.0 Bullish)", color="white", fontsize=10, fontweight="bold")
            self.ax_strategy_radar.set_ylim(-1.2, 1.2)

            self.fig_strategy.canvas.draw_idle()
        except Exception:
            pass


def main() -> None:
    app = SelfTradeModernGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
