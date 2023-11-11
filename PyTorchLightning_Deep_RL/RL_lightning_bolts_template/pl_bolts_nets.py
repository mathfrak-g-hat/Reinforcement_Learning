#############################################
#### PyTorch Lightning Bolts Neural nets ####
#############################################
## Updated: 23/06/25

import math
from typing import Tuple

import numpy as np
import torch
from torch import FloatTensor, Tensor, nn
from torch.distributions import Categorical, Normal, MultivariateNormal
from torch.nn import functional as F

'''
Source:

https://github.com/Lightning-Universe/lightning-bolts/blob/0.5.0/pl_bolts/models/rl/common/distributions.py
'''
class TanhMultivariateNormal(torch.distributions.MultivariateNormal):
    """The distribution of X is an affine of tanh applied on a normal distribution.
    X = action_scale * tanh(Z) + action_bias
    Z ~ Normal(mean, variance)
    
    AJ Zerouali, 23/06/21: They forgot about the devices
    AJ Zerouali, 23/07/04: You have to let lightning assign the devices. It's more
                           more involved than it looks.
    """

    def __init__(self, action_bias, action_scale, **kwargs):
        super().__init__(**kwargs)

        self.action_bias = action_bias
        self.action_scale = action_scale

    def rsample_with_z(self, sample_shape=torch.Size()):
        """Samples X using reparametrization trick with the intermediate variable Z.
        Returns:
            Sampled X and Z
        """
        z = super().rsample()
        '''
        # DEBUG
        print(f"z.device = {z.device}")
        print(f"type(z) = {type(z)}")
        print(f"self.action_scale.device = {self.action_scale.device}")
        print(f"self.action_bias.device = {self.action_bias.device}")
        #print(f"next(self.parameters()).is_cuda = {next(self.parameters()).is_cuda}")
        '''
        
        action_scale = torch.Tensor(self.action_scale).to(z.device)
        action_bias = torch.Tensor(self.action_bias).to(z.device)
        
        output = (action_scale * torch.tanh(z) + action_bias, z)
        
        return output

    def log_prob_with_z(self, value, z):
        """Computes the log probability of a sampled X.
        Refer to the original paper of SAC for more details in equation (20), (21)
        Args:
            value: the value of X
            z: the value of Z
        Returns:
            Log probability of the sample
        """
        action_scale = torch.Tensor(self.action_scale).to(z.device)
        action_bias = torch.Tensor(self.action_bias).to(z.device)
        
        value = (value - action_bias) / action_scale
        z_logprob = super().log_prob(z)
        correction = torch.log(action_scale * (1 - value ** 2) + 1e-7).sum(1)
        return z_logprob - correction

    def rsample_and_log_prob(self, sample_shape=torch.Size()):
        """Samples X and computes the log probability of the sample.
        Returns:
            Sampled X and log probability
        """
        
        z = super().rsample()
        z_logprob = super().log_prob(z)
        value = torch.tanh(z)
        
        action_scale = torch.Tensor(self.action_scale).to(z.device)
        action_bias = torch.Tensor(self.action_bias).to(z.device)        
        
        correction = torch.log(action_scale * (1 - value ** 2) + 1e-7).sum(1)
        return action_scale * value + action_bias, z_logprob - correction

    def rsample(self, sample_shape=torch.Size()):
        fz, z = self.rsample_with_z(sample_shape)
        return fz

    def log_prob(self, value):
        
        action_scale = torch.Tensor(self.action_scale).to(value.device)
        action_bias = torch.Tensor(self.action_bias).to(value.device)
        
        value = (value - action_bias) / action_scale
        z = torch.log(1 + value) / 2 - torch.log(1 - value) / 2
        return self.log_prob_with_z(value, z)
    
'''
Source:
 
https://github.com/Lightning-Universe/lightning-bolts/blob/0.5.0/pl_bolts/models/rl/common/networks.py
'''
class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self, input_shape: Tuple[int], n_actions: int, hidden_size: int = 128):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        '''
        self.net = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        '''
        # AJZ, 23/06/25
        self.net = nn.Sequential(
            nn.Linear(input_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, input_x):
        """Forward pass through network.
        Args:
            x: input to network
        Returns:
            output of network
        """
        return self.net(input_x.float())

class ContinuousMLP(nn.Module):
    """MLP network that outputs continuous value via Gaussian distribution."""

    def __init__(
        self,
        input_shape: Tuple[int],
        n_actions: int,
        hidden_size: int = 128,
        action_bias: int = 0,
        action_scale: int = 1,
    ):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: dimension of actions in the environment
            hidden_size: size of hidden layers
            action_bias: the center of the action space
            action_scale: the scale of the action space
        """
        super().__init__()
        self.action_bias = action_bias
        self.action_scale = action_scale
        '''
        self.shared_net = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_size, n_actions)
        self.logstd_layer = nn.Linear(hidden_size, n_actions)
        '''
        self.shared_net = nn.Sequential(
            nn.Linear(input_shape[0], 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, n_actions)
        self.logstd_layer = nn.Linear(256, n_actions)

    def forward(self, x: FloatTensor) -> TanhMultivariateNormal:
        """Forward pass through network. Calculates the action distribution.
        Args:
            x: input to network
        Returns:
            action distribution
        """
        # DEBUG
        #print(f"x.device = {x.device}")
        #print(f"next(self.parameters()).is_cuda = {next(self.parameters()).is_cuda}")
        
        x = self.shared_net(x.float())
        batch_mean = self.mean_layer(x)
        logstd = torch.clamp(self.logstd_layer(x), -20, 2) # Shouldn't this be (-2,2)?
        batch_scale_tril = torch.diag_embed(torch.exp(logstd))
        output = TanhMultivariateNormal(action_bias=self.action_bias, 
                                        action_scale=self.action_scale, 
                                        loc=batch_mean, 
                                        scale_tril=batch_scale_tril,)
        return output

    def get_action(self, x: FloatTensor) -> Tensor:
        """Get the action greedily (without sampling)
        Args:
            x: input to network
        Returns:
            mean action
        """
        x = self.shared_net(x.float())
        batch_mean = self.mean_layer(x)
        return self.action_scale * torch.tanh(batch_mean) + self.action_bias