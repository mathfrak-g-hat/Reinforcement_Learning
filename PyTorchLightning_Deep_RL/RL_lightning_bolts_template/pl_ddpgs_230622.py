###########################
##### Simplified DDPG #####
###########################
### AJ Zerouali, 23/06/22


import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import gym
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from RL_lightning_bolts_template.pl_bolts_replay_buffers import Experience, ExperienceSourceDataset, MultiStepBuffer
from RL_lightning_bolts_template.pl_bolts_nets import MLP, ContinuousMLP


class DDPGS(LightningModule):
    def __init__(
        self,
        env: gym.Env, # Initially a str
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1,
        gamma: float = 0.99,
        policy_learning_rate: float = 3e-4,
        q_learning_rate: float = 3e-4,
        target_alpha: float = 5e-3,
        batch_size: int = 128,
        replay_size: int = 1000000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        seed: int = 123,
        batches_per_epoch: int = 10000,
        n_steps: int = 1,
        **kwargs,
    ):
        super().__init__()

        ### NOTE: I dislike this
        # Training environment
        self.env = env
        #self.env = gym.make(env)
        #self.test_env = gym.make(env) # NOTE: Could be important. Get back to this

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.shape[0]

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.policy = None
        self.q_net = None
        self.target_q_net = None
        self.build_networks()

        # Hyperparameters
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
        
        self.automatic_optimization = False

    '''
        AJZerouali
    '''
    # This method is originally from SoftActorCriticAgent(Agent)
    # SoftActorCriticAgent.get_action()
    # This method calls ContinuousMLP.get_action(), returns only
    # the mean of the stochastic policy.
    def policy_eval_mean(self, states: Tensor, device: str) -> List[float]:
        """Deterministic evaluation of the stochastic policy,
           returns mean (mu) only.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action given by policy's mean
        """
        # Debug
        print(f"ENTERING policy_eval_mean()")
        
        if not isinstance(states, list):
            states = [states]

        if not isinstance(states, Tensor):
            states = torch.tensor(states, device=device)
        
        actions_ = self.policy.get_action(states)
        actions = [actions_.cpu().numpy()]
        
        return actions
            
        
    
    # This is originally SoftActorCriticAgent.__call__().
    # This method first gets a distrib. by calling ContinuousMLP.forward(),
    # and then returns an action sampled from this distribution.
    def policy_sample(self, states: Tensor, device: str) -> List[float]:
        """Sample an action from the stochastic policy.
            
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action sampled from stochastic policy
        """
        
        # DEBUG
        #print(f"ENTERING policy_sample()")
        
        if not isinstance(states, list):
            states = [states]

        if not isinstance(states, Tensor):
            states = torch.tensor(states, device=device)

        pi = self.policy(states)
        actions_ = pi.sample()
        actions = [a for a in actions_.cpu().numpy()]

        return actions
    
    '''
        PL-BOLTS
    '''
    def run_n_episodes(self, env, n_episodes: int = 1) -> List[int]:
        """Carries out N episodes of the environment with the current agent without exploration.

        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
        """
        total_rewards = []

        for _ in range(n_episodes):
            episode_state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Get stochastic policy mean (deterministic eval)
                action = self.policy_eval_mean(self.state, self.device)
                
                # gym v0.26.2+: step() returns (observation, reward, terminated, truncated, info)
                next_state, reward, done, _, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience."""
        if warm_start > 0:
            self.state, _ = self.env.reset()

            for i in range(warm_start):
                
                # Sample an action from the policy distribution
                #print(f"CALLING policy_sample() FROM populate()")
                #print(f"ITER i={i} of warm_start={warm_start}")
                # 2306221430 Bug isn't from here
                action = self.policy_sample(self.state, self.device)
                
                next_state, reward, done, _, _ = self.env.step(action[0])
                
                # NOTE: Change this shit
                exp = Experience(state=self.state, 
                                 action=action[0], 
                                 reward=reward, 
                                 done=done, 
                                 new_state=next_state)
                
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state, _ = self.env.reset()

    def build_networks(self) -> None:
        """Initializes the DDPG policy and q network with target"""
        
        # Policy net (stochastic)
        action_bias = torch.from_numpy((self.env.action_space.high + self.env.action_space.low) / 2)
        action_scale = torch.from_numpy((self.env.action_space.high - self.env.action_space.low) / 2)
        self.policy = ContinuousMLP(self.obs_shape, 
                                    self.n_actions, 
                                    action_bias=action_bias, 
                                    action_scale=action_scale)
        
        # Critic net and target
        concat_shape = [self.obs_shape[0] + self.n_actions]
        self.q_net = MLP(concat_shape, 1)
        self.target_q_net = MLP(concat_shape, 1)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def soft_update_target(self):
        """Update the weights in target network using a weighted sum.

        w_target := (1-a) * w_target + a * w_q

        """
        ### NOTE: Clarify what happens in this loop
        for q_param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(
                (1.0 - self.hparams.target_alpha) * target_param.data + self.hparams.target_alpha * q_param
            )

    def forward(self, state: Tensor) -> Tensor:
        """Passes in a state through the network and gets the q_values of each action as an output.

        Args:
            state: environment state

        Returns:
            Sampled action
        """
        pi = self.policy(state)
        output = pi.sample()
        return output

    def train_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            
            action = self.policy_sample(self.state, self.device)

            next_state, r, is_done, _, _ = self.env.step(action[0])

            episode_reward += r
            episode_steps += 1

            exp = Experience(state=self.state, 
                             action=action[0], 
                             reward=r, 
                             done=is_done, 
                             new_state=next_state)

            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
                self.state, _ = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.hparams.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.hparams.batches_per_epoch == 0:
                break

    def loss(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the loss for SAC which contains a total of 3 losses.
            AJ Zerouali, 23/06/22: It's this method that causes me confusion. Normally, this
                                   is part of the training_step() (i.e. learn() method).
                                   Lightning separated this process.
                                   The SAC implementation of bolts made it even more confusing
                                   with the "new" prefix.

        Args:
            batch: a batch of states, actions, rewards, dones, and next states
        """
        # DEBUG
        #print(f"ENTERING loss()")
        states, actions, rewards, dones, next_states = batch
        rewards = rewards.unsqueeze(-1)
        dones = dones.float().unsqueeze(-1)
        
        # Evauate current Q-values, w.r.t. current (batch) state-actions
        states_actions = torch.cat((states, actions), 1)
        q_values = self.q_net(states_actions)
        
        # Evaluate "new" actions and Q-values, w.r.t. current (batch) states
        pi = self.policy(states)
        new_actions, new_logprobs = pi.rsample_and_log_prob()
        new_logprobs = new_logprobs.unsqueeze(-1)
        new_states_actions = torch.cat((states, new_actions), 1)
        new_q_values = self.q_net(new_states_actions)
        
        # Target network evaluations
        with torch.no_grad():
            
            # Evaluate "new" next actions, w.r.t. (batch) next states
            next_pi = self.policy(next_states)
            new_next_actions, new_next_logprobs = next_pi.rsample_and_log_prob()
            new_next_logprobs = new_next_logprobs.unsqueeze(-1)
            
            # Evaluate next Q-values, w.r.t. (batch) next states
            new_next_states_actions = torch.cat((next_states, new_next_actions), 1)
            next_q_values = self.target_q_net(new_next_states_actions)
            
            # Get target q_values
            target_q_values = rewards + (1.0 - dones) * self.hparams.gamma * next_q_values

        # Compute critic loss
        critic_loss= F.mse_loss(q_values, target_q_values)
        
        # Compute actor loss
        policy_loss = -new_q_values.mean()
        
        return policy_loss, critic_loss


    def training_step(self, batch: Tuple[Tensor, Tensor], _):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        """
        # DEBUG
        #print(f"ENTERING training_step()")
        
        # Get optimizers and losses
        policy_optim, critic_optim = self.optimizers()
        policy_loss, critic_loss = self.loss(batch)
        
        # DEBUG
        #print(f"training_step(): Beginning gradient steps")

        '''
            IMPORTANT REMARK: IF YOU CALL THE CRITIC STEP
            BEFORE THE POLICY STEP, THEN PYTORCH CRASHES
            WITH THE FOLLOWING ERROR MESSAGE:
            
            RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
            [torch.cuda.FloatTensor [128, 1]], which is output 0 of AsStridedBackward0, is at version 2; 
            expected version 1 instead. 
            Hint: enable anomaly detection to find the operation that failed to compute its gradient, 
            with torch.autograd.set_detect_anomaly(True).
            
        '''
        
        # Policy gradient step
        policy_optim.zero_grad()
        self.manual_backward(policy_loss)
        policy_optim.step()
        #print(f"training_step(): Did actor grad step"
        
        # Critic gradient step
        critic_optim.zero_grad()
        self.manual_backward(critic_loss)
        critic_optim.step()
        #print(f"training_step(): Did critic grad step")
 
        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.soft_update_target()

        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "policy_loss": policy_loss,
                "critic_loss": critic_loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

    ## Question: What is this?
    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 10 episodes."""
        test_reward = self.run_n_episodes(self.test_env, 1)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}

    '''
        NOTE: The replay buffer changes
    '''
    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.hparams.replay_size, self.hparams.n_steps)
        self.populate(self.hparams.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        # DEBUG
        #print(f"ENTERING train_dataloader()")
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self._dataloader()

    def configure_optimizers(self) -> Tuple[Optimizer]:
        """Initialize Adam optimizer."""
        # DEBUG
        #print(f"ENTERING configure_optimizers()")
        policy_optim = optim.Adam(self.policy.parameters(), self.hparams.policy_learning_rate)
        critic_optim = optim.Adam(self.q_net.parameters(), self.hparams.q_learning_rate)
        return policy_optim, critic_optim

    # Maybe I should remove this
    @staticmethod
    def add_model_specific_args(
        arg_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        raise NotImplementedError()