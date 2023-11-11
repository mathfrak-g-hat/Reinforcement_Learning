'''
Source:

https://github.com/Lightning-Universe/lightning-bolts/blob/0.5.0/pl_bolts/models/rl/common/agents.py
'''

from abc import ABC
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Agent(ABC):
    """Basic agent that always returns 0."""

    def __init__(self, net: nn.Module):
        self.net = net

    def __call__(self, state: Tensor, device: str, *args, **kwargs) -> List[int]:
        """Using the given network, decide what action to carry.
        Args:
            state: current state of the environment
            device: device used for current batch
        Returns:
            action
        """
        return [0]

class SoftActorCriticAgent(Agent):
    """Actor-Critic based agent that returns a continuous action based on the policy."""

    def __call__(self, states: Tensor, device: str) -> List[float]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        if not isinstance(states, list):
            states = [states]

        if not isinstance(states, Tensor):
            states = torch.tensor(states, device=device)

        dist = self.net(states)
        actions = [a for a in dist.sample().cpu().numpy()]

        return actions

    def get_action(self, states: Tensor, device: str) -> List[float]:
        """Get the action greedily (without sampling)
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        if not isinstance(states, list):
            states = [states]

        if not isinstance(states, Tensor):
            states = torch.tensor(states, device=device)

        actions = [self.net.get_action(states).cpu().numpy()]

        return actions