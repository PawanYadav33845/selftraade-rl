"""
Dual Memory Buffer Module: Live Experience Replay Buffer + Historical Market Dataset Buffer.
Implements 70/30 mixed batch sampling to prevent catastrophic forgetting.
"""

from typing import Dict, Any, List, Tuple
from collections import deque
import numpy as np
import torch


class Transition:
    """Represents a single state transition tuple."""
    __slots__ = ("state", "action", "reward", "next_state", "done", "log_prob", "value")

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
    ) -> None:
        self.state = np.array(state, dtype=np.float32)
        self.action = int(action)
        self.reward = float(reward)
        self.next_state = np.array(next_state, dtype=np.float32)
        self.done = bool(done)
        self.log_prob = float(log_prob)
        self.value = float(value)


class LiveExperienceBuffer:
    """
    Circular FIFO Replay Buffer for live and paper trading transitions.
    """

    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class HistoricalBuffer:
    """
    Static/Expanding buffer containing historical market regime dataset transitions.
    """

    def __init__(self) -> None:
        self.transitions: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def add_bulk(self, transitions: List[Transition]) -> None:
        self.transitions.extend(transitions)

    def sample(self, batch_size: int) -> List[Transition]:
        if not self.transitions:
            return []
        indices = np.random.choice(len(self.transitions), size=min(batch_size, len(self.transitions)), replace=False)
        return [self.transitions[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.transitions)


class MixedBatchSampler:
    """
    Samples training batches combining 70% Recent Live Replay Buffer + 30% Historical Dataset.
    """

    def __init__(
        self,
        live_buffer: LiveExperienceBuffer,
        historical_buffer: HistoricalBuffer,
        live_ratio: float = 0.70,
    ) -> None:
        self.live_buffer = live_buffer
        self.historical_buffer = historical_buffer
        self.live_ratio = live_ratio

    def sample_batch(self, batch_size: int = 64) -> Dict[str, torch.Tensor]:
        """
        Samples a mixed batch formatted as PyTorch Tensors.
        """
        live_len = len(self.live_buffer)
        hist_len = len(self.historical_buffer)

        if live_len == 0 and hist_len == 0:
            raise ValueError("Cannot sample from empty buffers.")

        if hist_len == 0:
            n_live = batch_size
            n_hist = 0
        elif live_len == 0:
            n_live = 0
            n_hist = batch_size
        else:
            n_live = max(1, int(round(batch_size * self.live_ratio)))
            n_hist = batch_size - n_live

        live_samples = self.live_buffer.sample(n_live) if n_live > 0 else []
        hist_samples = self.historical_buffer.sample(n_hist) if n_hist > 0 else []
        combined = live_samples + hist_samples

        states = np.array([t.state for t in combined], dtype=np.float32)
        actions = np.array([t.action for t in combined], dtype=np.int64)
        rewards = np.array([t.reward for t in combined], dtype=np.float32)
        next_states = np.array([t.next_state for t in combined], dtype=np.float32)
        dones = np.array([t.done for t in combined], dtype=np.float32)
        log_probs = np.array([t.log_prob for t in combined], dtype=np.float32)
        values = np.array([t.value for t in combined], dtype=np.float32)

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "next_states": torch.tensor(next_states, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
            "log_probs": torch.tensor(log_probs, dtype=torch.float32),
            "values": torch.tensor(values, dtype=torch.float32),
        }
