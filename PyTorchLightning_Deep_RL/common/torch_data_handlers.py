###########################################
##### PYTORCH DATASET AND DATALOADERS #####
###########################################
## AJ Zerouali, 2023/07/02
'''
Sources:
https://lightning.ai/docs/pytorch/1.9.5/notebooks/lightning_examples/reinforce-learning-DQN.html#Memory

https://github.com/Lightning-Universe/lightning-bolts/blob/0.5.0/pl_bolts/datamodules/experience_source.py

https://github.com/Lightning-Universe/lightning-bolts/blob/0.5.0/pl_bolts/models/rl/common/memory.py
'''

# Named tuple for storing experience steps gathered in training
from abc import ABC
import collections
from collections import deque, namedtuple
from typing import List, Tuple, Union, Callable, Iterator

import numpy as np

import torch
from torch.utils.data import IterableDataset


from common.replay_buffers import ReplayBuffer



Transition = namedtuple("Transition", field_names=["state", "action", "reward", "state_next", "done"])

DelayPrdData = namedtuple("DelayPrdData", field_names=["state", "ampl", "omega", "phi", "prd_return"])
    
class RLDataset(IterableDataset):
    """
        Calls the batch generator function to produce an iterator

    """

    def __init__(self, generate_batch: Callable,) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator[Tuple]:
        '''
            Get iterator from the generate_batch() function
        '''
        iterator = self.generate_batch()
        return iterator