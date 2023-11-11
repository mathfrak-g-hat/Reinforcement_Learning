### AJ Zerouali, 2023/07/09

from collections import OrderedDict, deque, namedtuple
from typing import Dict, List, Tuple
import numpy as np
import gym

# Torch imports 
import torch as th
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer


from common.networks import MLP, GSDEPolicy
from common.replay_buffers import ReplayBuffer
from common.torch_data_handlers import Transition


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

class Agent(nn.Module):
            
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
                n_warm_start_steps: int = 2000,):
                
        super().__init__()
        
        
        # device
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        # Training environment attributes
        self.env = env
        self.state_space_dim = self.env.observation_space.shape[0]
        self.action_space_dim = self.env.action_space.shape[0]
        self.state = None
        
        '''
            Hyperparameters
            These can be automatically saved in Lightning
        '''
        self.gamma = gamma
        self.policy_params_dict = policy_params_dict
        self.q_net_arch_dict = q_net_arch_dict
        self.policy_lr = policy_lr
        self.q_net_lr = q_net_lr
        self.target_tau = target_tau
        self.n_tgt_sync_steps = n_tgt_sync_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_warm_start_steps = n_warm_start_steps
        self.n_weight_resample_steps = policy_params_dict["n_weight_resample_steps"]
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Build Q-network 
        self.q_net = MLP(mlp_type = "q_net",
                         state_space_dim = self.state_space_dim,
                         action_space_dim = self.action_space_dim,
                         **self.q_net_arch_dict)
        self.q_net.to(self.device)
                         
        # Build target Q-network
        self.target_q_net = MLP(mlp_type = "q_net",
                         state_space_dim = self.state_space_dim,
                         action_space_dim = self.action_space_dim,
                         **self.q_net_arch_dict)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.to(self.device)
        
        
        # Build policy network
        self.policy = GSDEPolicy(state_space_dim = self.state_space_dim,
                                 action_space_dim = self.action_space_dim,
                                 **self.policy_params_dict)
        self.policy.set_random_weights_distribution()
        self.policy.to(self.device)
        
        # Optimizers
        
        self.policy_optim = policy_optimizer_class(self.policy.parameters(),
                                                   self.policy_lr)
        self.critic_optim = q_net_optimizer_class(self.q_net.parameters(), 
                                                  self.q_net_lr)
        
        # Logging attributes and metrics
        #self.min_episode_score = 0.0
        self.total_episode_steps = [0]
        #self.avg_score_len = avg_score_len
        self.avg_score = 0.0
        self.episode_score = 0.0
        self.total_scores = [0]
        self.done_episodes = 0
        self.global_step = 0
        self.total_steps = 0
    
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
                action_ = self.policy(state_).to(device)
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
                
                
    '''
    ############################################
    ### Training step - DDPG impelementation ###
    ############################################
    '''
    def training_step(self, batch) -> Tuple[np.ndarray, np.ndarray]:
        """
            DDPG algorithm learning step.
            Original paper: https://arxiv.org/abs/1509.02971
            
            Note: The batch collection is done by train_dataloader() before
                this function is called.
                The batch collection is different for on-policy algorithms.
            
            :param batch: 
        """
        states, actions, rewards, states_next, dones = batch
        '''
            IMPORTANT NOTE: Lightning converts the batch to th.Tensor.
            Remove the conversion below when porting to Lightning.
        '''
        states_ = th.Tensor(states).to(self.device)
        actions_ = th.Tensor(actions).to(self.device)
        rewards_ = th.Tensor(rewards).to(self.device)
        states_next_ = th.Tensor(states_next).to(self.device)
        dones_ = th.BoolTensor(dones).to(self.device)
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
            '''
            # NOTE: Try by sampling action from the policy distribution
            #       instead of sampling the noise
            actions_next_ = self.policy.forward(states_next_)
            '''
            actions_next_ = self.policy.get_action_distribution(states_next_).sample()
            
            target_q_values_next = self.target_q_net.forward(states_next_,
                                                             actions_next_)
            # Note: dones_.shape = rewards_.shape = (batch_size,)
            target_q_values_next[dones_.view(-1,1)] = 0.0
            target_q_values = rewards_.view(-1,1) + self.gamma*target_q_values_next # Beware of hparams
        
        # Critic gradient step
        self.critic_optim.zero_grad() # MOdify for lightning
        critic_loss= F.mse_loss(q_values, target_q_values)
        '''
        # IMPORTANT: In lightning:
        self.manual_backward(critic_loss)
        '''
        critic_loss.backward()
        self.critic_optim.step() # MOdify for lightning
        '''
        new_actions_ = self.policy.forward(states_)
        '''
        new_actions_ = self.policy.get_action_distribution(states_).rsample()
        
        policy_loss = self.q_net.forward(states_, new_actions_) # q_net in eval() mode
        policy_loss = -th.mean(policy_loss)
        
        # Policy loss computation and gradient step
        self.policy_optim.zero_grad() # MOdify for lightning
        '''
        # IMPORTANT: In lightning:
        self.manual_backward(policy_loss)
        '''
        policy_loss.backward()
        self.policy_optim.step()

        # Update of target network
        if self.global_step % self.n_tgt_sync_steps == 0:
            '''
                Use tau?
            '''
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        '''
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
        np_policy_loss = policy_loss.cpu().detach().numpy()
        np_critic_loss = critic_loss.cpu().detach().numpy()
        
        self.global_step += 1
        
        return np_critic_loss, np_policy_loss

    
def train(agent: Agent,
          save_checkpoints: bool = True,
          ckpt_prefix: str = "./checkpoints/gSDE_DDPG_model",
          N_training_episodes: int = 100,
          max_episode_steps: int = 10000,
          max_global_steps: int = 500000,
         ):
    # Learning params for checkpointing
    learning_param_dict = {"state_space_dim": agent.state_space_dim,
                           "action_space_dim":agent.action_space_dim,
                           "gamma": agent.gamma,
                           "policy_params_dict": agent.policy_params_dict,
                           "q_net_arch_dict": agent.q_net_arch_dict,
                           "policy_lr": agent.policy_lr,
                           "q_net_lr": agent.q_net_lr,
                           "target_tau": agent.target_tau,
                           "n_tgt_sync_steps": agent.n_tgt_sync_steps,
                           "buffer_size": agent.buffer_size,
                           "batch_size": agent.batch_size,
                           "n_warm_start_steps": agent.n_warm_start_steps,
                           "n_weight_resample_steps": agent.n_weight_resample_steps,}
    
    # Fill buffer with warmup steps if empty
    if (agent.n_warm_start_steps>0) and (len(agent.replay_buffer)==0):
        agent.populate(agent.n_warm_start_steps)
    
    # Init. avg reward hist
    avg_score_hist = []
    critic_loss_hist = []
    policy_loss_hist = []
    agent.global_step = 0
    agent.done_episodes = 0
    agent.episode_score = 0.0
    agent.avg_score = 0.0
    
    # Main loop
    for episode in range(N_training_episodes):

        # Init env
        state = agent.env.reset() # gym 0.21.0 to 0.24.0
        done = False
        # Sample agent policy noise
        agent.policy.set_random_weights_distribution()
        agent.policy.sample_random_weights()
        
        # Reward averaging variables
        n_episode_steps = 0
        episode_reward_list = []
        episode_critic_loss_list = []
        episode_policy_loss_list = []

        while not done and (n_episode_steps<max_episode_steps)\
            and (agent.global_step <max_global_steps):
            
            # Compute action and convert to np
            ## Remark: agent.get_action() uses th.no_grad()
            ##  to compute the action
            action = agent.get_action(th.Tensor(state), agent.device)
            
            # Step in environment
            state_next, reward, done, info = agent.env.step(action)
            
            # Store transition in replay buffer
            '''
                IMPORTANT: 
                1) Lightning should do the conversion and device setting automatically
                2) DO NOT convert the batch elements here, it will conflict with agent.populate()
            '''
            transition = Transition(state=state.astype(dtype = np.float32), 
                                    action=action.astype(dtype = np.float32),
                                    reward=reward.astype(dtype = np.float32),
                                    state_next=state_next.astype(dtype = np.float32),
                                    done=done,)
            agent.replay_buffer.store(transition)
            
            # Add rwrd to hist
            episode_reward_list.append(reward)
            
            # Draw batch of transitions
            batch = agent.replay_buffer.sample(batch_size = agent.batch_size)
            
            # Execute Delay learning
            critic_loss, policy_loss = agent.training_step(batch = batch)
            episode_critic_loss_list.append(critic_loss)
            episode_policy_loss_list.append(policy_loss)
            
            # Update state
            state = state_next
            
            # Update n_episode_steps
            n_episode_steps += 1
            
            # Update noise distribution and random weights
            if (n_episode_steps % agent.n_weight_resample_steps == 0):
                agent.policy.set_random_weights_distribution()
                agent.policy.sample_random_weights()
                    
            # END OF EPISODIC WHILE LOOP    
        
        # DEBUG
        print(f"agent.policy.log_sigma = {agent.policy.log_sigma}")
        
        # Avg rewards and losses
        episode_avg_score = np.mean(episode_reward_list)
        avg_score_hist.append(episode_avg_score)
        episode_avg_critic_loss = np.mean(episode_critic_loss_list)
        episode_avg_policy_loss = np.mean(episode_policy_loss_list)
        
        # Checkpointing
        '''
            TO DO (23/07/09):
            - Fix the chcekpointing and average score/reward
        '''
        ## Ref: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        if save_checkpoints:
            # First checkpoint
            if episode == 0:
                ckpt_fname = ckpt_prefix+"_episode-"+str(episode)+".ckpt"
                th.save({'episode': episode,
                         'tot_steps': agent.global_step,
                         'q_net_state_dict': agent.q_net.state_dict(),
                         'policy_state_dict': agent.policy.state_dict(),
                         'learning_params_dict': learning_param_dict,
                        }, ckpt_fname)
                print(f"Saved first checkpoint under: {ckpt_fname}")
            
            elif ((episode+1) == N_training_episodes) or ((agent.global_steps+1)==max_global_steps):
                
                ckpt_fname = ckpt_prefix+"_episode-"+str(episode)+"_EndTrain.ckpt"
                th.save({'episode': episode,
                         'tot_steps': agent.global_step,
                         'q_net_state_dict': agent.q_net.state_dict(),
                         'policy_state_dict': agent.policy.state_dict(),
                         'learning_params_dict': learning_param_dict,
                        }, ckpt_fname)
                print(f"Saved end of training checkpoint under: {ckpt_fname}")
            # Save checkpoint for highest average score
            else:
                if episode_avg_score >= np.max(avg_score_hist):
                    ckpt_fname = ckpt_prefix+"_BestAvgRwrd.ckpt"
                    th.save({'episode': episode,
                         'tot_steps': agent.global_step,
                         'q_net_state_dict': agent.q_net.state_dict(),
                         'policy_state_dict': agent.policy.state_dict(),
                         'learning_params_dict': learning_param_dict,
                        }, ckpt_fname)
                    print(f"New highest average reward: {episode_avg_score}.")
                    print(f"Previous highest value: {np.max(avg_score_hist)}")
                    print(f"Saved new checkpoint under: {ckpt_fname}")        
        
        # Update lists
        avg_score_hist.append(episode_avg_score)
        critic_loss_hist.append(episode_avg_critic_loss)
        policy_loss_hist.append(episode_avg_policy_loss)
        
        # Verbose
        print(f"End of episode {episode} ({agent.global_step} steps). \nAverage score: {episode_avg_score}")
        print(f"Average critic loss: {episode_avg_critic_loss}")
        print(f"Average policy loss: {episode_avg_policy_loss}")
        print(f"=========================================")