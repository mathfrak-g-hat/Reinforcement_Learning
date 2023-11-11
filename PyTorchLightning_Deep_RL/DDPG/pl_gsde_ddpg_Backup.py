#############################################################
##### DDPG WITH GENERALIZED STATE-DEPENDENT EXPLORATION #####
#############################################################
### AJ Zerouali, 23/07/09
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
from common.networks import MLP, GSDEPolicy
from common.torch_data_handlers import Transition, RLDataset


DEFAULT_POLICY_PARAMS = {"n_weight_resample_steps": 250,
                         "net_arch":[256, 256],
                         "dropout_probs":[],
                         "layer_activation_fn":nn.ReLU,
                         "log_sigma_init": 1.5,
                         "log_sigma_bound": 2.0,
                         "use_tanh_scaling": True,
                         "weight_init_mthd": "Xavier_uniform",
                         "weight_init_seed":123,}


DEFAULT_CRITIC_ARCH = {"net_arch":[256, 256],
                       "dropout_probs":[],
                       "layer_activation_fn":nn.ReLU,
                       "output_activation_fn":nn.Identity,
                       "weight_init_mthd": "Xavier_uniform",
                       "weight_init_seed":123,}



class GSDE_DDPG(LightningModule):
    ###################
    ### Constructor ###
    ###################
    def __init__(self,
                 env: gym.Env,
                 gamma: float = 0.99,
                 policy_params_dict: dict = DEFAULT_POLICY_PARAMS,
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
        self.n_weight_resample_steps = policy_params_dict["n_weight_resample_steps"]
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
        
        # From lightning bolts.
        ## What does this do?
        for _ in range(self.avg_score_len):
            self.total_scores.append(torch.tensor(self.min_episode_score, device=self.device))

        self.avg_score = float(np.mean(self.total_scores[-self.avg_score_len :]))
        
    
    ###############################
    ### Neural nets contruction ###
    ###############################
    def build_networks(self)->None:
        '''
            Builds networks by calling MLP class contstructor,
            initializes the action noise generator if provided,
            and assigns the get_action() method depending on
            whether or not action noise is added.
        '''
        
        # Build Q-network 
        self.q_net = MLP(mlp_type = "q_net",
                         state_space_dim = self.state_space_dim,
                         action_space_dim = self.action_space_dim,
                         **self.hparams.q_net_arch_dict)
        
        # Build target Q-network
        self.target_q_net = MLP(mlp_type = "q_net",
                         state_space_dim = self.state_space_dim,
                         action_space_dim = self.action_space_dim,
                         **self.hparams.q_net_arch_dict)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Build policy network
        self.policy = GSDEPolicy(state_space_dim = self.state_space_dim,
                                 action_space_dim = self.action_space_dim,
                                 **self.hparams.policy_params_dict)
        self.policy.set_random_weights_distribution()
            
    
    ##########################
    ### Network evaluation ###
    ##########################
    def forward(self, state: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
            Forward function of the LightningModule
            Evaluates both the q_net and policy for 
            a given state.
            
            :param state: th.Tensor, state to evaluate
            :return q_value: th.Tensor, where q_value = q_net(state, policy(state))
            :param action: th.Tensor, where action = policy(state)
            
        """
        action = self.policy(state)
        q_value = self.q_net(state, action)
        return q_value, action
    
    def get_action(self, state: th.Tensor, device, deterministic: bool = False):
        '''
             Get action from policy. If deterministic is False, returns only
             the mean action.
            
            :param state: th.Tensor, state to evaluate.
            :param device: "cpu" or "cuda", depending on the module's current device
                            (use device = self.device when in doubt)
                            
            :return action: np.ndarray action vector 
        '''
        state_ = th.Tensor(state).to(device)
        
        if len(state.shape)==1:
            state_ = state_.view(1,-1)
        
        with th.no_grad():
            # Deterministic case
            if deterministic:
                action_mu_ = self.policy.forward_mu(state_)
                action_ = self.policy.output_activation_fn(action_mu_).to(device)
                if len(state.shape)==1:
                    action_ = action_[0]
            # Stochastic case
            else:
                '''
                # DEBUG
                policy_param_list = list(self.policy.parameters())
                print(f"device = {device}")
                #print(f"state.device = {state.device}")
                print(f"policy_param_list[0].device = {policy_param_list[0].device}")
                print(f"policy_param_list[1].device = {policy_param_list[1].device}")
                print(f"self.policy.xi.device = {self.policy.xi.device}")
                '''
                
                action_ = self.policy(state_).to(device)
                '''
                ## By the way, not how it should be computed.
                pi = self.policy.get_action_distribution(state_)
                action_ = pi.sample()
                '''
                if len(state.shape)==1:
                    action_ = action_[0]
        
        action = action_.detach().cpu().numpy()
        
        return action

        
    
    #####################
    ### Data handling ###
    #####################
    def populate(self, n_steps: int = 1000) -> None:
        """
            Fills replay buffer with initial experience.
            Called by train_dataloader() if n_warm_start_steps > 0.
            
            :param n_steps: Num. of transitions to add to buffer.
        """
        self.state = self.env.reset()
        self.policy.sample_random_weights()
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
                self.policy.sample_random_weights()
            
            if ((i+1) % self.n_weight_resample_steps) == 0:
                self.policy.sample_random_weights()
            


    def train_dataloader(self) -> DataLoader:
        """
            Get training loader. Populates the replay buffer if it is empty
            and n_warm_start_steps>0.
        """
        # DEBUG
        #print("==> ENTERING train_dataloader()")
        
        # Populate replay buffer if empty and n_warm_start_steps>0
        if (len(self.replay_buffer) == 0) and (self.hparams.n_warm_start_steps>0):
            self.populate(n_steps = self.hparams.n_warm_start_steps) 
        
        # Build RLDataset and dataloader
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
            Initialize the policy and q_net optimizers.
        """
        policy_optim = self.hparams.policy_optimizer_class(self.policy.parameters(), 
                                                           self.hparams.policy_lr)
        critic_optim = self.hparams.q_net_optimizer_class(self.q_net.parameters(), 
                                                          self.hparams.q_net_lr)
        return policy_optim, critic_optim


    def get_device(self, batch) -> str:
        """
            Retrieve device currently being used by minibatch.
        """
        return batch[0].device.index if self.on_gpu else "cpu"
    
    '''
    ############################################
    ### Training step - DDPG impelementation ###
    ############################################
    '''
    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """
            DDPG algorithm learning step.
            Original paper: https://arxiv.org/abs/1509.02971
            
            Note: The batch collection is done by train_dataloader() before
                this function is called.
                The batch collection is different for on-policy algorithms.
            
            :param batch: 
        """
        
        '''
            NOTE: all the batch outputs are torch tensors.
            
            
            REMARKS: 
            - There's a messup with the gradient update for the policy loss.
            RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
        '''
        #device = self.get_device(batch)
        policy_optim, critic_optim = self.optimizers()
        
        states_, actions_, rewards_, states_next_, dones_ = batch
        
        # WHY?
        ## The agent.policy.log_sigma parameter is updated at
        ## each gradient step. The weights distribution should
        ## then be updated
        self.policy.set_random_weights_distribution()
        self.policy.rsample_random_weights()
        
        # Compute current Q-values
        q_values = self.q_net.forward(states_, actions_)
        
        with th.no_grad():
            # Compute next target Q-values
            
            # NOTE: Try by sampling action from the policy distribution
            #       instead of sampling the noise
            actions_next_ = self.policy.forward(states_next_)
            '''
            actions_next_ = self.policy.get_action_distribution(states_next_).sample()
            '''
            target_q_values_next = self.target_q_net.forward(states_next_,
                                                             actions_next_)
            # Note: dones_.shape = rewards_.shape = (batch_size,)
            target_q_values_next[dones_.view(-1,1)] = 0.0
            target_q_values = rewards_.view(-1,1) + self.hparams.gamma*target_q_values_next # Beware of hparams
        
        # Critic gradient step
        critic_optim.zero_grad() # MOdify for lightning
        critic_loss= F.mse_loss(q_values, target_q_values)
        self.manual_backward(critic_loss)
        critic_optim.step() # MOdify for lightning
        
        new_actions_ = self.policy.forward(states_)
        '''
        new_actions_ = self.policy.get_action_distribution(states_).rsample()
        '''
        policy_loss = self.q_net.forward(states_, new_actions_) # q_net in eval() mode
        policy_loss = -th.mean(policy_loss)
        
        # Policy loss computation and gradient step
        policy_optim.zero_grad() # MOdify for lightning
        self.manual_backward(policy_loss)
        policy_optim.step()

        # Update of target network
        if self.global_step % self.hparams.n_tgt_sync_steps == 0:
            '''
                Use tau? Polyak updates?
            '''
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
            self.policy.sample_random_weights()

        while True:
            
            # Compute action
            action = self.get_action(self.state, self.device)
            
            # Environment step
            #next_state, r, is_done, _, _ = self.env.step(action[0])
            state_next, reward, done, _= self.env.step(action)
            
            # Store transition in replay buffer
            ## Note: Keeping floats to avoid lightning from crashing
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
            
            if (episode_steps % self.n_weight_resample_steps) == 0:
                self.policy.set_random_weights_distribution()
                self.policy.sample_random_weights()

            if done:
                # 
                self.done_episodes += 1
                self.total_scores.append(episode_score)
                self.total_episode_steps.append(episode_steps)
                self.avg_score = float(np.mean(self.total_scores[-self.avg_score_len :]))
                
                # Reset env
                self.state = self.env.reset()
                self.policy.set_random_weights_distribution()
                self.policy.sample_random_weights()
                episode_score = 0.0
                episode_steps = 0

            states, actions, rewards, states_next, dones = self.replay_buffer.sample(self.hparams.batch_size)
            
            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], states_next[idx], dones[idx]

            # Simulates epochs
            if self.total_steps % self.hparams.batches_per_epoch == 0:
                break
