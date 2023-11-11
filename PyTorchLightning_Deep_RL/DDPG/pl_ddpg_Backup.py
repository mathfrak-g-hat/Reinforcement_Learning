################
##### DDPG #####
################
### AJ Zerouali, 23/07/02
## gym v0.24.0 version

from collections import OrderedDict, deque, namedtuple
from typing import Dict, List, Tuple
import numpy as np
import torch
import gym

# Torch imports 
import torch as th
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

# Torch lightning imports
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.loggers import CSVLogger

# Common submodule imports
from common.replay_buffers import ReplayBuffer
from common.networks import MLP
from common.torch_data_handlers import Transition, RLDataset
from common.noise import OrnsteinUhlenbeckActionNoise


DEFAULT_POLICY_ARCH = {"net_arch":[256, 256],
                       "dropout_probs":[],
                       "layer_activation_fn":nn.ReLU,
                       "output_activation_fn":nn.Tanh,
                       "weight_init_mthd": "Xavier_uniform",
                       "weight_init_seed":123,}

DEFAULT_CRITIC_ARCH = {"net_arch":[256, 256],
                       "dropout_probs":[],
                       "layer_activation_fn":nn.ReLU,
                       "output_activation_fn":nn.Identity,
                       "weight_init_mthd": "Xavier_uniform",
                       "weight_init_seed":123,}


class DDPG(LightningModule):
    ###################
    ### Constructor ###
    ###################
    def __init__(self,
                 env: gym.Env,
                 gamma: float = 0.99,
                 policy_arch_dict: dict = DEFAULT_POLICY_ARCH,
                 q_net_arch_dict: dict = DEFAULT_CRITIC_ARCH,
                 policy_optimizer_class: Optimizer = Adam,
                 q_net_optimizer_class: Optimizer = Adam,
                 policy_lr: float = 3e-4, 
                 q_net_lr: float = 3e-4,
                 target_tau: float = 5e-3,
                 n_tgt_sync_steps: int = 10,
                 buffer_size: int = 1000000,
                 batch_size: int = 128,
                 n_warm_start_steps: int = 0,
                 min_episode_score: float = 0.0,
                 max_episode_steps: int = 1000,
                 batches_per_epoch: int = 10000,
                 avg_score_len: int = 100,
                 OU_AN_sigma: float = 0.15,
                 OU_AN_theta: float = 0.2,
                 OU_AN_dt: float = 1e-2,
                 **kwargs,
                ) -> None:
                
                
        super().__init__()
        # Hyperparameters
        self.save_hyperparameters()
        '''
            CONSTRUCTOR TO DO:
            1) Add actor and critic architecture params.
            2) Be careful with the device handling
        '''
        
        # Training environment attributes
        self.env = env
        self.state_space_dim = self.env.observation_space.shape[0]
        self.action_space_dim = self.env.action_space.shape[0]
        self.state = None

        # Model Attributes
        ## Replay buffer and dataset
        self.replay_buffer = ReplayBuffer(buffer_size = self.hparams.buffer_size)
        self.dataset = None
        
        ## Actor and critic networks
        self.policy = None
        self.action_noise_gen = None
        self.q_net = None
        self.target_q_net = None
        self.build_networks()
        
        
        # Logging attributes and metrics
        self.min_episode_score = min_episode_score
        self.total_episode_steps = [0]
        self.avg_score_len = avg_score_len
        self.avg_score = 0.0
        self.episode_score = 0.0
        self.total_scores = [0]
        self.done_episodes = 0
        self.total_steps = 0
        
        # CRUCIAL
        self.automatic_optimization = False
        
        # Hyperparameters
        self.save_hyperparameters()
        
        # From lightning bolts.
        ## What does this do?
        for _ in range(self.avg_score_len):
            self.total_scores.append(torch.tensor(self.min_episode_score, device=self.device))

        self.avg_score = float(np.mean(self.total_scores[-self.avg_score_len :]))
        
        # Warm start steps
        if self.hparams.n_warm_start_steps>0:
            self.action_noise_gen.reset()
            self.populate(n_steps = self.hparams.n_warm_start_steps)

    
    ###############################
    ### Neural nets contruction ###
    ###############################
    def build_networks(self)->None:
        '''
            Build networks by calling MLP class contstructor
        '''
        
        # Build Q-network 
        self.q_net = MLP(mlp_type = "q_net",
                         state_space_dim = self.state_space_dim,
                         action_space_dim = self.action_space_dim,
                         net_arch = self.hparams.q_net_arch_dict["net_arch"],
                         dropout_probs = self.hparams.q_net_arch_dict["dropout_probs"],
                         layer_activation_fn = self.hparams.q_net_arch_dict["layer_activation_fn"],
                         output_activation_fn = self.hparams.q_net_arch_dict["output_activation_fn"],
                         weight_init_mthd = self.hparams.q_net_arch_dict["weight_init_mthd"],
                         weight_init_seed = self.hparams.q_net_arch_dict["weight_init_seed"])
        
        # Build target Q-network
        self.target_q_net = MLP(mlp_type = "q_net",
                         state_space_dim = self.state_space_dim,
                         action_space_dim = self.action_space_dim,
                         net_arch = self.hparams.q_net_arch_dict["net_arch"],
                         dropout_probs = self.hparams.q_net_arch_dict["dropout_probs"],
                         layer_activation_fn = self.hparams.q_net_arch_dict["layer_activation_fn"],
                         output_activation_fn = self.hparams.q_net_arch_dict["output_activation_fn"],
                         weight_init_mthd = self.hparams.q_net_arch_dict["weight_init_mthd"],
                         weight_init_seed = self.hparams.q_net_arch_dict["weight_init_seed"])
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Build policy network
        self.policy = MLP(mlp_type = "policy",
                         state_space_dim = self.state_space_dim,
                         action_space_dim = self.action_space_dim,
                         net_arch = self.hparams.policy_arch_dict["net_arch"],
                         dropout_probs = self.hparams.policy_arch_dict["dropout_probs"],
                         layer_activation_fn = self.hparams.policy_arch_dict["layer_activation_fn"],
                         output_activation_fn = self.hparams.policy_arch_dict["output_activation_fn"],
                         weight_init_mthd = self.hparams.policy_arch_dict["weight_init_mthd"],
                         weight_init_seed = self.hparams.policy_arch_dict["weight_init_seed"])
        
        # Init. action noise generate generate
        self.action_noise_gen = OrnsteinUhlenbeckActionNoise(mu = np.zeros(shape = (self.action_space_dim,)),
                                                             sigma = self.hparams.OU_AN_sigma,
                                                             theta = self.hparams.OU_AN_theta,
                                                             dt = self.hparams.OU_AN_dt,
                                                            )
    
    ##########################
    ### Network evaluation ###
    ##########################
    def forward(self, state: th.Tensor) -> th.Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            actions
        """
        action = self.policy(state)
        return action
    
    def get_action(self, state: th.Tensor, device) -> th.Tensor:
        '''
            Get action from policy and add exploration noise
        '''
        state_ = th.Tensor(state).to(device)
        with th.no_grad():
            action_0_ = self.policy(state_)
            action_ = action_0_ + th.Tensor(self.action_noise_gen()).to(device)
        action = action_.cpu().numpy()
        return action
        
    
    #####################
    ### Data handling ###
    #####################
    def populate(self, n_steps: int = 1000) -> None:
        """
            Fills replay buffer with initial experience.
            Called by constructor if n_warm_start_steps>0.
            
            :param n_steps: Num. of transitions to add to buffer.
        """
        self.state = self.env.reset()
        #self.action_noise_gen.reset()
        for i in range(n_steps):
            action = self.get_action(self.state, self.device)
            state_next, reward, done, _ = self.env.step(action)
            # Store with float dtype
            transition = Transition(self.state.astype(dtype = np.float32), 
                                    action.astype(dtype = np.float32), 
                                    reward.astype(dtype = np.float32), 
                                    state_next.astype(dtype = np.float32), 
                                    done)
            self.replay_buffer.store(transition)
            self.state = state_next
            if done:
                self.state = self.env.reset()
                #self.action_noise_gen.reset()
            

    def train_dataloader(self) -> DataLoader:
        """
            Get train loader.
        """
        # DEBUG
        #print("==> ENTERING train_dataloader()")
        
        self.action_noise_gen.reset()
        dataset = RLDataset(generate_batch = self.train_batch)
        dataloader = DataLoader(dataset = dataset, batch_size=self.hparams.batch_size,)
        
        # DEBUG
        #print("==> EXITING train_dataloader()")
        return dataloader

    ###############################
    ### Learning param. methods ###
    ###############################
    def configure_optimizers(self) -> List[Optimizer]:
        """
            Initialize Adam optimizer.
        """
        policy_optim = self.hparams.policy_optimizer_class(self.policy.parameters(), 
                                                           self.hparams.policy_lr)
        critic_optim = self.hparams.q_net_optimizer_class(self.q_net.parameters(), 
                                                          self.hparams.q_net_lr)
        return policy_optim, critic_optim



    def get_device(self, batch) -> str:
        """
            Retrieve device currently being used by minibatch.
            QUESTION (23/07/01):
            1) I thought the batch was made of numpy arrays
            2) What batch are we talking about
            3)
        """
        return batch[0].device.index if self.on_gpu else "cpu"
    
    '''
    ############################################
    ### Training step - DDPG impelementation ###
    ############################################
    '''
    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """
            DDPG
        """
        
        '''
            NOTE: all the batch outputs are torch tensors.
        '''
        #device = self.get_device(batch)
        policy_optim, critic_optim = self.optimizers()
        
        states_, actions_, rewards_, states_next_, dones_ = batch
        
        
        '''
        # DEBUG
        #print(f"self.get_device(batch) = {self.get_device(batch)}")
        print("==> FROM training_step():")
        print(f"self.on_gpu = {self.on_gpu}")
        print(f"states_.shape = {states_.shape}, states_.dtype = {states_.dtype}")
        print(f"actions_.shape = {actions_.shape}, actions_.dtype = {actions_.dtype}")
        print(f"rewards_.shape = {rewards_.shape}, rewards_.dtype = {rewards_.dtype}")
        print(f"states_next_.shape = {states_next_.shape}, states_next_.dtype = {states_next_.dtype}")
        print(f"dones_.shape = {dones_.shape}, dones_.dtype = {dones_.dtype}")
        '''
        
                
        self.policy.eval()
        self.target_q_net.eval()
        self.q_net.train()
        
        # Compute current Q-values
        q_values = self.q_net.forward(states_, actions_)
        '''
        # DEBUG
        print(f"q_values.shape = {q_values.shape}, q_values.dtype = {q_values.dtype}")
        '''
        
        # Compute next target Q-values
        actions_next_ = self.policy.forward(states_next_)
        target_q_values_next = self.target_q_net.forward(states_next_,
                                                         actions_next_)
        # I want this to crash...
        target_q_values_next[dones_.view(-1,1)] = 0.0
        target_q_values = rewards_.view(-1,1) + self.hparams.gamma*target_q_values_next
        
        '''
        # DEBUG
        print(f"rewards_.view(-1,1).shape = {rewards_.view(-1,1).shape}, rewards_.view(-1,1).dtype = {rewards_.view(-1,1).dtype}")
        print(f"target_q_values_next.shape = {target_q_values_next.shape}, target_q_values_next.dtype = {target_q_values_next.dtype}")
        '''
        
        
        # Critic gradient step
        critic_optim.zero_grad()
        critic_loss= F.mse_loss(q_values, target_q_values)
        self.manual_backward(critic_loss)
        critic_optim.step()
        #print(f"training_step(): Did critic grad step")
        
        self.q_net.eval()
        self.policy.train()
        
        # Policy loss computation and gradient step
        policy_optim.zero_grad()
        new_actions_ = self.policy.forward(states_)
        policy_loss = self.q_net.forward(states_, new_actions_)
        policy_loss = -th.mean(policy_loss)
        self.manual_backward(policy_loss)
        policy_optim.step()

        # Update of target network
        if self.global_step % self.hparams.n_tgt_sync_steps == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Check with older implementation
        self.log_dict(
            {
                "total_score": self.total_scores[-1],
                "avg_score": self.avg_score,
                "policy_loss": policy_loss,
                "critic_loss": critic_loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

    
    '''
        ############################
        #### GET TRAINING BATCH ####
        ############################
        Batch generation for off-policy RL algorithms.
        
    '''
    def train_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        """
            Builds a generator that will pass a new batch of data to the DataLoader.

        Returns:
            yields a Transition tuple containing the state, action, reward, state_next and done.
        """
        # DEBUG
        #print("==> STARTING train_batch()")
        
        # Episode total score (sum of rewards)
        episode_score = 0.0
        # Episode steps
        episode_steps = 0
        
        if self.total_steps == 0:
            self.state = self.env.reset()
            self.action_noise_gen.reset()

        while True:
            
            # Compute action
            action = self.get_action(self.state, self.device)
            
            # Environment step
            #next_state, r, is_done, _, _ = self.env.step(action[0])
            state_next, reward, done, _= self.env.step(action)
            
            # Store transition in replay buffer
            '''
                Maybe this is where I should change the dtype
            '''
            transition = Transition(state=self.state.astype(dtype = np.float32), 
                                    action=action.astype(dtype = np.float32),
                                    reward=reward.astype(dtype = np.float32),
                                    state_next=state_next.astype(dtype = np.float32),
                                    done=done,)
            self.replay_buffer.store(transition)
            
            # Update total score, num. of steps, and agent's observed state
            episode_score += reward
            episode_steps += 1
            self.total_steps += 1
            self.state = state_next

            if done:
                # 
                self.done_episodes += 1
                self.total_scores.append(episode_score)
                self.total_episode_steps.append(episode_steps)
                self.avg_score = float(np.mean(self.total_scores[-self.avg_score_len :]))
                
                # Reset env
                self.state = self.env.reset()
                episode_score = 0.0
                episode_steps = 0

            states, actions, rewards, states_next, dones = self.replay_buffer.sample(self.hparams.batch_size)
            '''
            # DEBUG
            print(f"type(states) = {type(states)}, ")
            print(f"type(actions) = {type(actions)}, ")
            print(f"type(rewards) = {type(rewards)}, ")
            print(f"type(states_next) = {type(states_next)}, ")
            print(f"type(dones) = {type(dones)}, ")
            '''
            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], states_next[idx], dones[idx]

            # Simulates epochs
            if self.total_steps % self.hparams.batches_per_epoch == 0:
                break
