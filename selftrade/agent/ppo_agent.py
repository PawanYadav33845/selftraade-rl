"""
Proximal Policy Optimization (PPO) PyTorch Agent Module.
Implements Actor-Critic architecture, policy loss optimization, state serialization, and checkpointing.
"""

from typing import Dict, Any, Tuple, Optional
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Neural Network for Discrete Action Space (Hold, Buy, Sell).
    Input: State vector S_t of size 7.
    Outputs: Action Categorical Distribution + Scalar Value V(s).
    """

    def __init__(self, state_dim: int = 7, action_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()

        # Shared Feature Extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor (Policy) Head
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic (Value) Head
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        logits, _ = self.forward(state)
        return Categorical(logits=logits)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(state)
        return value.squeeze(-1)


class PPOAgent:
    """
    PPO Agent wrapper managing policy network, optimization steps, and checkpoint saving/loading.
    """

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        version_tag: str = "v1.0",
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.version_tag = version_tag
        if device in ("auto", "cpu", None):
            is_cuda = False
            try:
                is_cuda = torch.cuda.is_available()
            except Exception:
                is_cuda = False
            self.device = torch.device("cuda" if is_cuda else "cpu")
        else:
            self.device = torch.device(device)

        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Selects action for a given single state observation.

        Returns:
            (action_int, log_prob_float, value_float)
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.network(state_t)
            dist = Categorical(logits=logits)

            if deterministic:
                action_t = torch.argmax(logits, dim=-1)
            else:
                action_t = dist.sample()

            log_prob_t = dist.log_prob(action_t)

        action = int(action_t.item())
        log_prob = float(log_prob_t.item())
        val = float(value.squeeze(0).item())

        return action, log_prob, val

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        ppo_epochs: int = 4,
    ) -> Dict[str, float]:
        """
        Executes PPO update step over a sampled batch.
        """
        states = batch["states"].to(self.device).detach()
        actions = batch["actions"].to(self.device).detach().view(-1)
        rewards = batch["rewards"].to(self.device).detach().view(-1)
        old_log_probs = batch["log_probs"].to(self.device).detach().view(-1)
        old_values = batch["values"].to(self.device).detach().view(-1)

        # Target returns calculation (Monte-Carlo / 1-step target for batch simplicity)
        with torch.no_grad():
            returns = (rewards + self.gamma * old_values).detach().view(-1)
            advantages = (returns - old_values).detach().view(-1)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach().view(-1)

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(ppo_epochs):
            logits, values = self.network(states)
            values = values.view(-1)
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions).view(-1)
            entropy = dist.entropy().mean()

            # PPO Clipped Objective
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            surr_loss = torch.where(surr1 < surr2, surr1, surr2)
            policy_loss = -surr_loss.mean()

            # Value Loss
            value_loss = nn.functional.mse_loss(values, returns)

            # Total Loss
            total_loss = policy_loss + (self.value_coef * value_loss) - (self.entropy_coef * entropy)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
        }

    def clone(self, new_version_tag: Optional[str] = None) -> "PPOAgent":
        """
        Creates a deep copy clone of the agent.
        """
        tag = new_version_tag or f"{self.version_tag}_clone"
        cloned_agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=self.lr,
            gamma=self.gamma,
            clip_eps=self.clip_eps,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            version_tag=tag,
            device=str(self.device),
        )
        cloned_agent.network.load_state_dict(copy.deepcopy(self.network.state_dict()))
        return cloned_agent

    def save(self, filepath: str) -> None:
        """
        Saves agent weights and metadata checkpoint.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "state_dict": self.network.state_dict(),
            "version_tag": self.version_tag,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> None:
        """
        Loads agent weights from checkpoint.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.version_tag = checkpoint.get("version_tag", self.version_tag)
