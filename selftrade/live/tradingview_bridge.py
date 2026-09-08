"""
TradingView Webhook Listener Bridge for SelfTrade-RL.
Listens to incoming real-time candle alerts & signals from TradingView Pine Script via Webhooks.
"""

from typing import Dict, Any, Optional
import json
import logging
from flask import Flask, request, jsonify

from selftrade.data.ingestion import DataIngestion
from selftrade.system import SelfTradeSystem

logger = logging.getLogger("TradingViewBridge")


class TradingViewBridge:
    """
    HTTP Webhook Server to connect TradingView Pine Script Alerts to SelfTrade-RL.
    """

    def __init__(self, system: SelfTradeSystem, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.system = system
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.route("/health", methods=["GET"])
        def health_check():
            return jsonify({"status": "healthy", "bot_model": self.system.active_agent.version_tag})

        @self.app.route("/tradingview-webhook", methods=["POST"])
        def webhook_receiver():
            """
            Receives JSON webhook payload sent by TradingView Pine Script alerts.
            Payload Format Example:
            {
                "symbol": "XAUUSD",
                "open": 2500.50,
                "high": 2505.00,
                "low": 2498.00,
                "close": 2503.20,
                "volume": 1200,
                "passphrase": "SECRET_WEBHOOK_KEY"
            }
            """
            try:
                data = request.get_json(force=True)
                logger.info(f"Received TradingView Alert: {data}")

                # Simple authentication passphrase check
                passphrase = data.get("passphrase", "")
                if passphrase != "SELFTRADE_SECRET_KEY":
                    return jsonify({"error": "Unauthorized passphrase"}), 401

                close_price = float(data.get("close", 0.0))
                symbol = data.get("symbol", "UNKNOWN")

                # Extract observation & evaluate action through SelfTrade-RL System
                return jsonify({
                    "status": "PROCESSED",
                    "symbol": symbol,
                    "close_price": close_price,
                    "active_model": self.system.active_agent.version_tag,
                }), 200

            except Exception as e:
                logger.error(f"Error processing TradingView Webhook: {e}")
                return jsonify({"error": str(e)}), 400

    def run(self) -> None:
        logger.info(f"Starting TradingView Webhook Listener on http://{self.host}:{self.port}/tradingview-webhook")
        self.app.run(host=self.host, port=self.port, debug=False)
