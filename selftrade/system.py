"""
SelfTradeSystem Orchestration Module.
Combines Data, Trading Environment, Market Sessions, Strategy Ensemble, PPO Agent, Risk Engine, Continuous Learning, and Shadow Validation into a unified pipeline.
"""

from typing import Dict, Any, Optional, Tuple, List
import logging
import numpy as np
import pandas as pd

from selftrade.data.ingestion import DataIngestion, MarketFeatureEngine
from selftrade.env.trading_env import TradingEnv
from selftrade.agent.ppo_agent import PPOAgent
from selftrade.agent.buffer import LiveExperienceBuffer, HistoricalBuffer, Transition
from selftrade.analysis.pattern_engine import PatternTrendEngine
from selftrade.sessions.market_sessions import MarketSessionsEngine, MarketSessionInfo
from selftrade.strategies.ensemble import StrategyEnsemble
from selftrade.reasoning.trade_reasoner import TradeReasoner, ReasonedTradeDecision
from selftrade.risk.risk_engine import DeterministicRiskEngine, RiskEvaluationResult
from selftrade.continuous_learning.monitor import PerformanceMonitor
from selftrade.continuous_learning.self_improver import SelfImprovingEngine
from selftrade.shadow.shadow_engine import ShadowValidationEngine, ShadowValidationResult

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SelfTradeSystem")


class SelfTradeSystem:
    """
    Main Orchestrator for SelfTrade-RL Autonomous Trading System.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        max_stop_loss_pct: float = 0.02,
        max_daily_drawdown_pct: float = 0.05,
        sharpe_threshold: float = 0.5,
        target_risk_reward: float = 2.0,
        volatility_shift_threshold: float = 0.25,
        periodic_interval: int = 2048,
        eval_test_window: int = 500,
        device: str = "cpu",
    ) -> None:
        self.df = df
        self.device = device
        self.target_risk_reward = target_risk_reward

        # 1. Initialize Environment
        self.env = TradingEnv(
            df=df,
            initial_capital=initial_capital,
            max_drawdown_limit=0.50,
        )

        # 2. Initialize Agent
        self.active_agent = PPOAgent(
            state_dim=7,
            action_dim=3,
            version_tag="v1.0_active",
            device=device,
        )

        # 3. Initialize Memory Buffers
        self.live_buffer = LiveExperienceBuffer(capacity=5000)
        self.historical_buffer = HistoricalBuffer()

        # 4. Initialize Engines
        self.pattern_engine = PatternTrendEngine()
        self.trade_reasoner = TradeReasoner(target_risk_reward=target_risk_reward)
        self.risk_engine = DeterministicRiskEngine(
            max_stop_loss_pct=max_stop_loss_pct,
            max_daily_drawdown_pct=max_daily_drawdown_pct,
        )

        # 5. Initialize Continuous Learning & Shadow Validation
        self.monitor = PerformanceMonitor(
            sharpe_threshold=sharpe_threshold,
            volatility_shift_threshold=volatility_shift_threshold,
            periodic_interval=periodic_interval,
        )

        self.self_improver = SelfImprovingEngine(
            agent=self.active_agent,
            live_buffer=self.live_buffer,
            historical_buffer=self.historical_buffer,
            monitor=self.monitor,
            fine_tune_epochs=5,
            batch_size=64,
        )

        self.shadow_engine = ShadowValidationEngine(
            promotion_threshold_multiplier=1.10,
            eval_test_window=eval_test_window,
        )

        self.system_metrics: List[Dict[str, Any]] = []

    def seed_historical_buffer(self, historical_df: pd.DataFrame) -> int:
        """
        Pre-populates the historical buffer with state-action transitions from historical market data.
        """
        hist_env = TradingEnv(df=historical_df, initial_capital=10000.0)
        obs, _ = hist_env.reset()

        count = 0
        done = False
        while not done:
            action = int(np.random.choice([0, 1, 2]))
            next_obs, reward, terminated, truncated, _ = hist_env.step(action)
            done = terminated or truncated

            transition = Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
            )
            self.historical_buffer.add(transition)
            obs = next_obs
            count += 1

        logger.info(f"Seeded Historical Buffer with {count} historical transitions.")
        return count

    def run_simulation(self, max_steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes trading simulation loop combining Market Sessions, Strategy Ensemble, Risk Layer, Agent, and Shadow Engine.
        """
        obs, info = self.env.reset()
        step = 0
        total_steps = max_steps or (self.env.num_steps - 1)

        logger.info(f"Starting SelfTrade-RL System Loop for {total_steps} steps...")

        while step < total_steps:
            current_state = obs.copy()
            current_df = self.env.df.iloc[: self.env.current_step + 1]

            # 1. Market Session Info
            session_info: MarketSessionInfo = MarketSessionsEngine.get_session_info()

            # 2. Real-Time Pattern Analysis
            market_struct = self.pattern_engine.analyze_chart(current_df)

            # 3. Agent predicts proposed action
            action_net, log_prob, val = self.active_agent.select_action(current_state, deterministic=False)

            # 4. Cognitive Reasoner Evaluation ("Think before trading")
            close_price = info["close_price"]
            atr = float(obs[4]) * close_price
            reasoned_decision: ReasonedTradeDecision = self.trade_reasoner.evaluate_trade(
                proposed_action=action_net,
                current_price=close_price,
                atr=atr,
                market_structure=market_struct,
                df=current_df,
            )

            # 5. Risk Engine evaluates action against hard safety rules
            risk_verdict: RiskEvaluationResult = self.risk_engine.evaluate_action(
                proposed_action=reasoned_decision.approved_action,
                env_info=info,
            )
            safe_action = risk_verdict.safe_action

            # 6. Environment executes safe action
            next_obs, reward, terminated, truncated, info = self.env.step(safe_action)
            done = terminated or truncated

            # 7. Log transition into Live Replay Buffer
            self.self_improver.record_transition(
                state=current_state,
                action=safe_action,
                reward=reward,
                next_state=next_obs,
                done=done,
                log_prob=log_prob,
                value=val,
            )

            # 8. Continuous Learning Triggers Check
            market_atr = float(current_state[4])
            triggered, candidate_agent, trigger_desc = self.self_improver.check_and_fine_tune(
                reward=reward,
                market_atr_norm=market_atr,
            )

            if triggered and candidate_agent is not None:
                logger.info(f"Step {step}: {trigger_desc}")
                self.shadow_engine.start_shadow_evaluation(
                    candidate_agent=candidate_agent,
                    candidate_tag=candidate_agent.version_tag,
                )

            # 9. Shadow Execution & Promotion Check
            shadow_result: Optional[ShadowValidationResult] = self.shadow_engine.record_shadow_step(
                state=current_state,
                active_reward=reward,
                env=self.env,
            )

            if shadow_result is not None:
                logger.info(f"Shadow Evaluation Finished: {shadow_result.reason}")
                if shadow_result.promoted and self.shadow_engine.candidate_agent is not None:
                    logger.info(
                        f"PROMOTION EVENT: Replacing Active Weights ({self.active_agent.version_tag}) "
                        f"with Candidate Weights ({shadow_result.candidate_tag})!"
                    )
                    self.active_agent = self.shadow_engine.candidate_agent
                    self.self_improver.agent = self.active_agent

            # Record telemetry step metrics
            step_record = {
                "step": step,
                "net_action": action_net,
                "safe_action": safe_action,
                "risk_overridden": risk_verdict.is_overridden,
                "circuit_breaker_active": risk_verdict.circuit_breaker_active,
                "reward": reward,
                "portfolio_value": info["portfolio_value"],
                "close_price": info["close_price"],
                "market_session": session_info.session_name,
                "confluence_score": reasoned_decision.confluence_score,
                "active_model_version": self.active_agent.version_tag,
            }
            self.system_metrics.append(step_record)

            obs = next_obs
            step += 1

            if done:
                logger.info(f"Episode Finished at step {step}. Portfolio Value: ${info['portfolio_value']:.2f}")
                break

        return self.system_metrics
