#########################
##### NOISE CLASSES #####
#########################
## AJ Zerouali, 2023/07/03
## Inspired by stable baselines3
## https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/noise.py
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional
import numpy as np


'''
    ACTION NOISE ABSTRACT CLASS
'''
class ActionNoise:
    '''
        Abstract class for callable action noise.
        Any subclass should implement the reset()
        method to reset the noise
    '''
    
    def __init__(self) -> None:
        pass
    
    def set_numpy_seed(self, seed: int = 123) -> None:
        np.random.seed(seed)
    
    @abstractmethod
    def __call__(self) -> None:
        raise NotImplementedError()
        
    def reset(self) -> None:
        raise NotImplementedError()


'''
    GAUSSIAN ACTION NOISE
'''
class GaussianActionNoise(ActionNoise):
    '''
        Gaussian action noise generator with
        fixed standard deviation (covariance is
        sigma times identity)
        
        :param mu: np.ndarray. Mean array. Should typicalloy have the shape of the action
                    vectors.
        :param sigma: float, standard deviation of the noise terms. sigma = 0.1 by default
        
        METHODS:
        __call__: Generates np.ndarray of noise terms whose shape is that of mu
        
        
    '''
    def __init__(self, mu: np.ndarray, sigma: float = 0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        x = np.random.normal(loc = self.mu, scale = self.sigma, size=self.mu.shape)
        return x

    def reset(self):
        pass


'''
    ORNSTEIN-UHLENBECK ACTION NOISE
    ## After Phil Tabor:
    https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/DDPG/noise.py
'''
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    '''
        Callable object for generating random numbers following
        an Ornstein-Uhlenbeck process specified by user.
        
        :param mu: np.ndarray. Mean array. Should typicalloy have the shape of the action
                    vectors.
        :param sigma: float, standard deviation of the noise terms. sigma = 0.15 by default
        :param theta: float, equals 0.2 by default. Controls rate of reversion to the mean.
        :param dt: float, eqals 0.01 by default. Time increment for the process' evolution.
                Should typically be the time increment of the environment.
    '''
    def __init__(self, mu: np.ndarray, sigma: float = 0.15, 
                 theta: float = 0.2, dt: float = 1e-2, 
                 x0: np.ndarray = None) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self) -> np.ndarray:
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self) -> None:
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)