##########################
##### REPLAY BUFFERS #####
##########################
## AJ Zerouali
## Updated 23/06/30

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from torch import nn

import random
from collections import deque, namedtuple
from typing import List, Tuple, Union, Callable, Iterator
from itertools import islice
#from common.torch_data_handlers import Transition, DelayPrdData

Transition = namedtuple("Transition", field_names=["state", "action", "reward", "state_next", "done"])

DelayPrdData = namedtuple("DelayPrdData", field_names=["state", "ampl", "omega", "phi", "prd_return"])


'''
    REPLAY BUFFER
    23/06/30 Version.
    Source:
    https://lightning.ai/docs/pytorch/1.9.5/notebooks/lightning_examples/reinforce-learning-DQN.html
'''
class ReplayBuffer:
    '''
        General replay buffer
    '''
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size 
        self.buffer = deque(maxlen = self.buffer_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

    def store(self, transition: Transition)->None:
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(a = len(self.buffer),
                                   size = batch_size,
                                   replace = False)
        
        states, actions, rewards, states_next, dones\
            = zip(*(self.buffer[idx] for idx in indices))
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        states_next = np.array(states_next)
        dones = np.array(dones)
        
        return states, actions, rewards, states_next, dones
        
    def get_last_N(self, N: int):
        batch = list(islice(self.buffer,
                            len(self.buffer)-N,
                            len(self.buffer),)
                    )
        
        states, actions, rewards, states_next, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, states_next, dones

'''
    DELAY BUFFER
    23/06/30 Version
'''
class DelayBuffer:
    '''
        End of period replay buffer for delay learning
        t, state, ampl, omega, phi, prd_return = map(np.stack, zip(*batch))
    '''
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size 
        self.buffer = deque(maxlen = self.buffer_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

    def store(self, prd_data: DelayPrdData)->None:
        self.buffer.append(prd_data)

    def sample(self, batch_size):
        indices = np.random.choice(a = len(self.buffer),
                                   size = batch_size,
                                   replace = False)
        
        states, ampls, omegas, phis, prd_returns\
            = zip(*(self.buffer[idx] for idx in indices))
        
        states = np.array(states)
        ampls = np.array(ampls)
        omegas = np.array(omegas)
        phis = np.array(phis)
        prd_returns = np.array(prd_returns)
        
        return states, ampls, omegas, phis, prd_returns

