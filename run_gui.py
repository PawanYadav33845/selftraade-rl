"""
GUI Launcher Script for SelfTrade-RL Modern Control Center.
Launches dark-themed desktop application with real-time chart rendering, status dashboard cards, and Emergency Force Stop.
"""

from selftrade.gui.app import SelfTradeModernGUI


def main():
    app = SelfTradeModernGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
