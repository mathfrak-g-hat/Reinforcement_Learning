#####################################
##### PROBABILITY DISTRIBUTIONS #####
#####################################
## AJ Zerouali
## Updated: 23/07/04
'''
    Comments:
    - This submodule contains the implementation
      of the classes of objects returned by
      stochastic policies, including generalized
      state-dependent exploration.
    - We draw inspiration from the Distribution class implemented
      by the stable_baselines3 team:
      https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py
    
      
'''

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Distribution, Normal, MultivariateNormal


'''
    TANH GAUSSIAN DISTRIBUTION
'''
class TanhNormal(Normal):
    '''
        Probability distribution class for a random
        variable Y = Tanh(X), X ~ Normal(mu, std).
        
        :param loc: Mean of the underlying Gaussian.
        :param scale: Standard deviation of the underlying Gaussian.
        :param epsilon: A cutoff value to avoid NaN.
    '''
    
    def __init__(self, loc: th.Tensor, scale: th.Tensor, epsilon: float = 1e-6, validate_args=None):
        super().__init__(loc, scale)
        self.epsilon = epsilon
        
    
    def rsample(self, sample_shape = th.Size()) -> th.Tensor:
        gaussian_rsample = super().rsample()
        return th.tanh(gaussian_rsample)
    
    def sample(self, sample_shape = th.Size()) -> th.Tensor:
        gaussian_sample = super().rsample()
        return th.tanh(gaussian_sample)
    
    def log_prob(self, value: th.Tensor) -> th.Tensor:
        '''
            Compute log likelihood at "value".
            NOTE: In practice, the output of this method should be summed along
            the second dimension.
            
            :param value: th.Tensor value at which to evaluate the log
                    of the probability distribution. Should 
                    be in the interval (-1.0, 1.0).
                    
            :return log_prob: th.Tensor of likelihood values.
        '''
        # To avoid issues with numerical imprecisions
        value_eps = th.finfo(value.dtype).eps
        value_ = th.clamp(value, -1.0 + value_eps, 1.0 - value_eps)
        
        # Compute likelihood at values
        gaussian_val = 0.5*th.log((1+value_)/(1-value_))
        gaussian_log_prob = super().log_prob(gaussian_val)
        log_prob = gaussian_log_prob - th.log(1-th.pow(value_,2)+self.epsilon)
        #log_prob = th.sum(log_prob, dim = 1)
        
        return log_prob
    
    def entropy(self):
        '''
            The Shannon entropy has no closed form solution for 
            this type of random variable
        '''
        return None