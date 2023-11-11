###################################################
##### DQN - nn.LightningModule implementation #####
###################################################
### AJ Zerouali, 2023/06/20

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import collections

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl
# from pytorch_lightning import cli_lightning_logo, LightningModule, seed_everything, Trainer

import argparse
from collections import deque, namedtuple, OrderedDict
from typing import Iterator, List, Tuple

class DQN(pl.LightningModule):
    
    # Constructor (CRUCIAL)
    def __init__(self,
                 train_env: gym.Env,
                 gamma: float = 0.99,
                 dqn_lr: float = 1e-4,
                 batch_size: int = 128,
                 buffer_size: int = 1000000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.02,
                 eps_last_frame: int = 150000,
                 sync_rate: int = 1,
                 target_alpha: float = 5e-3,
                 n_warmup_steps: int = 10000,
                 avg_reward_len: int = 100,
                 #min_episode_reward: int = -21, # ?
                 seed: int = 101,
                 batches_per_epoch: int = 10000,
                 n_steps: int = 1,
                 **kwargs,
                ):
        '''
            Explain constructor params...
        '''
        
        # Mandatory torch call
        super(DQN, self).__init__()
        
        # Assign constructor attr
        self.train_env = train_env
        self.test_env = None
        self.observation_space_shape = self.train_env.observation_space.shape
        ### Discrete action space here
        self.n_actions = self.train_env.action_space.n #### INCORRECT
        
        # Model attributes
        ### Don't assign the dataloader as a class attribute
        self.replay_buffer = None
        self.dataset = None
        self.net = None
        self.target_net = None
        ### Build model 
        self._build_model()
        
        # Save mdoel hparams (lightning mandatory call)
        self.save_hyperparameters()
        
        
        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        for _ in range(avg_reward_len):
            self.total_rewards.append(torch.tensor(min_episode_reward, device=self.device))

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

        
        # Transition book-keeping
        self.state = None
        self.replay_buffer = ReplayBuffer(buffer_size = self.hparams.buffer_size)
        
        # See docs (looks important)
        self.automatic_optimization = False
        
        
    # Build the model
    def _build_model(self) -> None:
        '''
            Initializes the DQN, and target net
        '''
             
        # Instantiate DQN
        ### Will use this class for box2d or classic control envs,
        ### in which case: observation_space_dim = observation_space_shape[0]
        self.net = DQN(observation_space_dim = self.hparams.observation_space_shape[0], 
                       n_actions = self.hparams.n_actions, 
                       n_hidden = 128)
        self.target_net = DQN(observation_space_dim = self.hparams.observation_space_shape[0],
                              n_actions = self.hparams.n_actions, 
                              n_hidden = 128)
        ### Copy params from DQN
        self.target_net.load_state_dict(self.net.state_dict())
        
    
    # Get actions from DQN (necessary for RL)
    def get_action(self, state: th.Tensor, device: th.device):
        """
            Computes action from DQN output
        """
        if not isinstance(state, th.Tensor):
            state = th.tensor(state, device=device)

        
        q_val = self.net(state)
        q_val_max, action_star = th.max(q_val, dim=1)
        return action_star.detach().cpu().numpy()
    
    
    # Run n_episodes (for testing)
    def run_n_episodes(self, 
                       test_env: gym.Env,
                       n_episodes: int = 1,
                       max_episode_length: int = 10000,
                      ):
        """
            Runs a number of episodes in a test environment without exploration.
            Actions are obtained from current DQN using self.get_actions().
            Called by the test_step() method.
            
            :param env: environment to use, either train environment or test environment
            :param n_episodes: number of episodes to run
            :param max_episode_length: Maximal num. of steps per episode
            
            :return total_rewards_hist:
        """
        # Init rwrds list
        total_rewards_hist = []
        
        # Main loop
        for i in range(n_episodes):
            
            # Initializations
            state = test_env.reset()
            done = False
            episode_tot_rwrd = 0
            step = 0
            
            # Episodic loop
            while not done and (step<max_episode_length):
                action = self.get_action(state, self.device)
                ## NOTE: gym >= 0.26.2 and gymnasium
                state_next, reward, done, truncated, _ \
                    = test_env.step(action[0])
                episode_tot_rwrd += reward
                state = state_next
                step+=1
            
            # Append episode total reward to hist
            total_rewards_hist.append(episode_tot_rwrd)
        
        # Output
        return total_rewards_hist
    
    # Necessary?
    def populate(self, n_warmup_steps: int) -> None:
        """
            Populates the replay buffer with the specified number
            of warmup transitions in the training environment.
            Resets env if done to continue warmup, and uses epsilon
            greedy policy.
            
            :param n_warmup_steps: Num. of warmup steps to make
        """
        if n_warmup_steps>0:
            self.state = self.train_env.reset()
            
            for i in range(n_warmup_steps):
                
                # Get action following eps-greedy policy
                ### Q: Where do we initialize self.epsilon?
                if np.random.random() < self.epsilon:
                    ### WARNING:Review action shape
                    action = np.array([self.test_env.action_space.sample()])
                else:
                    action = self.get_action(self.state, self.device)
                
                state_next, reward, done, truncated, _ \
                    = self.train_env.step(action[0])
                self.replay_buffer.append(self.state, action, reward, done, state_next)
                self.state = state_next
                
                if done:
                    self.state = self.train_env.reset()
            
            
    
    # Training dataloader assignment (CRUCIAL)
    def train_dataloader(self) -> DataLoader:
        """
            Initialize training dataloader
        """
        self.dataset = RLDataset(self.replay_buffer)
        return DataLoader(dataset = self.dataset, batch_size = self.hparams.batch_size)
    
    # Forward (CRUCIAL)
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
            Passes in a state x through the network and gets the q_values of each action as an output.
            # NOTE: For an actor-critic algorithm with a stochastic policy, this function shou
        """
        output = self.net(x)
        return output
    
    # Optimizers' initialization f'n (CRUCIAL)
    def configure_optimizers(self) -> List[Optimizer]:
        """
            Initialize optimizers for all class networks.
            # NOTE: Should return a list or a tuple of Optimizer objects.
        """
        optimizer = optim.Adam(self.net.parameters(), lr = self.hparams.dqn_lr)
        return [optimizer]
    
    # Compute loss (CRUCIAL)
    def loss(self, batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor])-> th.Tensor:
        """
            Method to calculate the loss value
            # NOTE: Review this part. I hardly recognize what this does.
        """
        state_b, action_b, reward_b, done_b, state_next_b = batch
        
        state_action_values = self.net(state_b).gather(1, 
                                                       action_b.unsqueeze(-1)).squeeze(-1)
        
        # Evaluate target 
        with th.no_grad():
            state_next_q_vals = self.target_net(state_next_b).max(1)[0]
            state_next_q_vals[done_b] = 0.0
            state_next_q_vals = state_next_q_vals.detach()
        
        # Bellman backup
        expected_state_action_vals = self.hparams.gamma*state_next_q_vals+reward_b
        
        # Output
        return nn.MSELoss()(state_action_values, expected_state_action_vals)
    
    # Training step (CRUCIAL)
    def training_step(self, batch):
        """
            Need to add comments here. Seems to be the the crux of the implementation.
            Where is this executed.
        """
        
        # Get optimizer
        dqn_optim = self.optimizers()
                
        # Get training batch
        
        # Compute loss
        loss = self.loss(batch)
        ### Clarify this condition
        #if self.trainer.use_dp or self.trainer.use_ddp2:
        
        # Gradient step
        ## NOTE: You use a manual backward here
        ## Important for actor-critic algos
        dqn_optim.zero_grad()
        self.manual_backward(loss)
        dqn_optim.step()
        
        # Update target net
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        # Log dict
        self.log_dict(
            {
                "total_reward": self.total_rewards_hist[-1],
                "avg_reward": self.avg_rewards,
                "train_loss": loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )
        
        # Output
        return OrderedDict({"loss": loss, "avg_reward": self.avg_rewards})    
        
    
    