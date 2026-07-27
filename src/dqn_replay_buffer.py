
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class ReplayTransition:
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    action_mask: List[float]
    next_action_mask: List[float]

    def to_json_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict) -> "ReplayTransition":
        return cls(
            state=[float(v) for v in payload["state"]],
            action=int(payload["action"]),
            reward=float(payload["reward"]),
            next_state=[float(v) for v in payload["next_state"]],
            done=bool(payload["done"]),
            action_mask=[float(v) for v in payload.get("action_mask", [])],
            next_action_mask=[float(v) for v in payload.get("next_action_mask", [])],
        )


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = max(1, int(capacity))
        self._storage: List[ReplayTransition] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: ReplayTransition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
            return
        self._storage[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self.capacity

    def extend(self, transitions: Sequence[ReplayTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int, *, rng: random.Random | None = None) -> List[ReplayTransition]:
        if not self._storage:
            raise ValueError("Cannot sample from an empty replay buffer.")
        rng = rng or random
        size = min(len(self._storage), max(1, int(batch_size)))
        return rng.sample(self._storage, size)

    def to_json_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": len(self._storage),
            "transitions": [item.to_json_dict() for item in self._storage],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict()), encoding="utf-8")

    @classmethod
    def from_json_dict(cls, payload: dict) -> "ReplayBuffer":
        buffer = cls(capacity=int(payload.get("capacity", 100_000)))
        transitions = [ReplayTransition.from_json_dict(item) for item in payload.get("transitions", [])]
        buffer.extend(transitions)
        return buffer

    @classmethod
    def load(cls, path: Path) -> "ReplayBuffer":
        return cls.from_json_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_numpy(self) -> Tuple[np.ndarray, ...]:
        if not self._storage:
            raise ValueError("Replay buffer is empty.")
        states = np.asarray([item.state for item in self._storage], dtype=float)
        actions = np.asarray([item.action for item in self._storage], dtype=int)
        rewards = np.asarray([item.reward for item in self._storage], dtype=float)
        next_states = np.asarray([item.next_state for item in self._storage], dtype=float)
        dones = np.asarray([item.done for item in self._storage], dtype=bool)
        masks = np.asarray([item.action_mask for item in self._storage], dtype=float)
        next_masks = np.asarray([item.next_action_mask for item in self._storage], dtype=float)
        return states, actions, rewards, next_states, dones, masks, next_masks
