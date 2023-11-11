##########################################
### SIMPLE DQN AGENT (DISCRETE ACTIONS) ##
##########################################
## 2022/05/13; Ahmed J Zerouali
## Updated: 2022/06/03
## We implement a deep Q-learning agent.
## This version deals only with discrete action spaces.
## The target Gym environment for this trial is Lunar Lander.

"""
Following "Human level control through deep RL" by Mnih et al. 2015, 
the deep Q-network algorithm we implement here is described by the 
following pseudocode:

            Initialize replay buffer B
            Initialize eps in [0,1] and discount factor gamma
            Initialize evaluation net Q^ with random weights w
            Initialize target net Q^_tgt with weights w_ = w
            for episode m=1,..., M:
                Reset environment (observe s_1)
                for step t=1,...,T:
                    Choose random action a_t w/ prob. eps, and argmax_a Q^(s,a;w) w/ prob. 1-eps # Epsilon greedy policy
                    Observe (r_t,s_(t+1))
                    Store transition (s_t, a_t, r_t, s_(t+1)) in B
                    Sample minibatch {(s_j,a_j,r_j,s'_j)} of size N from B
                    for j=1,...,N:
                        if s_j is terminal:
                            Set y_j = r_j
                        else:
                            Set y_j = r_j + gamma*max_a Q^(s_j,a)
                    Perform SGD on L(w;w_,B) with respect to the evaluation parameters w
                    Every K steps update w_ = w.
                    
In this first implementation, we do not use a target network.

"""

from __future__ import print_function, division
from builtins import range


# NumPy, matplotlib, TF2, Keras, Gym
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow import keras


####################
### REPLAY BUFFER ##
####################
# 22/05/15, AJ Zerouali

class Replay_Buffer():
    """
        Replay Buffer class for DQN Agent. Instantiated as an attribute of DQN_Agent class.
        Manages the transitions arrays, has one method to store the transitions,
        one method to sample a minibatch of transitions.
        Attributes: 
            mem_size: Maximal size of memory (no. of (s,a,s',r) transitions); 
            mem_cntr: No. of transitions added.
            Self explanatory (np.arrays):
            memory_states, memory_actions, memory_next_states, memory_rewards, memory_terminal
        Methods:
            store_transitions
            sample_buffer
            
    """
    def __init__(self, max_size, input_dims, discrete_actions=True):
        """
            
        INPUT: - max_size: int for max no. of transition samples
               - input_dims: int for array shape of observations
               - discrete_actions: Bool descibing action space
        
        """
        
        # Memory size and counter
        self.mem_size = max_size
        self.mem_cntr = 0
        
        # Discrete actions boolean
        self.discrete_actions = discrete_actions
                
        # Transition arrays for (s,a,s',r)
        self.memory_states = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.memory_actions = np.zeros(self.mem_size, \
                            dtype = np.int8 if self.discrete_actions else np.float32 )
        self.memory_next_states = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.memory_rewards = np.zeros(self.mem_size, dtype = np.float32)
        # If memory_next_states[t] is TERMINAL, will set memory_terminal[t]= 1, and 0 otherwise.
        # Note: Different from Tabor's implementation.
        self.memory_terminal = np.zeros(self.mem_size, dtype = np.int8)
        
    def store_transition(self, state, action, new_state, reward, done):
        """
         Method for storing transitions (s, a, s', r) in the Replay_Buffer.
         INPUT: state, action, new_state, reward = (s, a, s', r)
                done: Boolean for s' terminal
         NOTE: Using tricks from P. Tabor's implementation
        """
        # Current transition index, over-write from first position when memory is full
        ind = self.mem_cntr % self.mem_size
        
        # Add new transition
        self.memory_states[ind] = state
        self.memory_actions[ind] = action
        self.memory_next_states[ind] = new_state
        self.memory_rewards[ind] = reward
        self.memory_terminal[ind] = int(done) # P. Tabor uses (1-int(done)) here
                                                # WARNING: Might be multiplying by this number somewhere
        
        # Update counter
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        """
        Method for random sampling of a memory minibatch
        INPUT: batch_size
        OUTPUT: sample_state, sample_action, sample_new_state, sample_reward, sample_terminal
                NumPy arrays of transitions of size specified by input
        """
        
        # Choose random indices 
        max_mem = min(self.mem_size, self.mem_cntr)
        batch_indices = np.random.choice(max_mem, batch_size, replace=False)
        
        # Get random minibatch
        sample_state = self.memory_states[batch_indices]
        sample_action = self.memory_actions[batch_indices]
        sample_new_state = self.memory_next_states[batch_indices]
        sample_reward = self.memory_rewards[batch_indices]
        sample_terminal = self.memory_terminal[batch_indices]
        
        return sample_state, sample_action, sample_new_state, sample_reward, sample_terminal


###########################
### DEEP Q-NETWORK AGENT ##
###########################
# 22/05/15, AJ Zerouali
# Notes: - Haven't written description yet

class DQN_Agent():
    """
        Deep Q-network agent class.
        Attributes:
            * RL hyperparameters: learn_rate, gamma, batch_size
                epsilon-greedy policy: epsilon, epsilon_dec, epsilon_min
            * Memory buffer: mem_size, memory_buffer
            * State/action space: input_dims, action_space, discrete_actions
            * Deep network: q_network, dqn_fname
        
        Methods:
            * Constructor
            * Memory management: store_transition
            * Environment interaction: choose_action
            * Deep network management: build_DQN, train_dqn, save_dqn, load_dqn
    """
    
    # Constructor
                # Environment dimensions and action space params
    def __init__(self, input_dims, discrete_actions, n_actions,  \
                 # RL hyperparameters
                 learn_rate, gamma, epsilon, batch_size, mem_size = 1000000, \
                 # Decrement and lower bound for eps-greedy
                 epsilon_dec = 1e-3, epsilon_min =0.01, \
                 # Filename model for saving
                 dqn_fname = 'dqn_model.h5'):                                             
        """
            INPUT: learn_rate, gamma, epsilon, batch_size, mem_size: Usual deep RL hyperparam
                   input_dims: Shape of states as arrays
                   n_actions: Size of the environment action space. See notes.
                   epsilon_dec, epsilon_min: Decrement of epsilon for policy, end value of epsilon.
            NOTES: 1) For a finite action space set n_actions = env.unwrapped.action_space.n as input.
                      (i.e. size of the finite action space).
                      For continuous action space take dimensionality and bounds of the "Box" object:
                      n_actions = (env.action_space.high, env.action_space.low, env.action_space.shape).
                   2) The agent uses an epsilon-greedy policy for training, with epsilon decreasing by
                      epsion_dec at each step and down to epsilon_min.
                      (Tabor starts with epsilon = 1.0)
            
        """
        # Build agent's action space
        self.discrete_actions = discrete_actions
        if self.discrete_actions:
            self.action_space = [i for i in range(n_actions)]
        else:
            self.action_space = n_actions
            
        # State space dimensionality
        self.input_dims = input_dims
        
        # Deep RL hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        
        # Init. DQN (build model with method below)
        self.q_network = "EMPTY"
        self.dqn_fname =  dqn_fname
        
        # Replay buffer. Signature: Replay_Buffer(max_size, input_dims, discrete_actions)
        self.mem_size = mem_size
        self.memory_buffer = Replay_Buffer(self.mem_size, self.input_dims, self.discrete_actions)
        
    # END DEF __init__()
    
    #############################
    ## Environment interaction ##
    #############################
    
    # Choose action following epsilon-greedy policy
    # NOTE: Implemented for discrete action space ONLY
    def choose_action(self, observation):
        """
            Method choosing action in input state following epsilon-greedy policy.
            INPUT: observation (current state)
            OUTPUT: action
        """
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.action_space) # MODIFY FOR CONTINUOUS ACTIONS
        else:
            state = np.array([observation])
            actions = self.q_network.predict(state)
            action = np.argmax(actions)
        
        return action
    # END DEF choose_action()
    
    # Sample transition storage
    def store_transition(self, state, action, new_state, reward, done):
        """
            Method to store a new (s, a, s', r, s'-terminal) transition.
            Calls method of same name from Replay_Buffer class.
        """
        self.memory_buffer.store_transition(state, action, new_state, reward, done)
    
    # END DEF store_transitions()
    
    ##########################
    ## Q-network management ##
    ##########################
    
    # Build Q-network
    def build_DQN(self, fc1_dims=256, fc2_dims=256):
        """
            Builds the DQN. 
            Will use a simple architecture here: 2 fully connected layers; each entry having 256 neurons.
            The loss function is the mean-squared error, the optimizer is Adam.
            Will return an error if attribute q_network is not "EMPTY".
            NOTE: - I will only write this function for the discrete action space for now.
                  - Will have to revise for continuous actions.
                  - I see why Tabor implemented it outside the class.
        """
        if self.q_network != "EMPTY":
            print("ERROR: Attribute q_network is non-empty.")
            self.q_network.get_config()
            return False
        else:
            # Build deep Q-network
            # Import Keras from tf in advance 
            dqn = keras.models.Sequential()
            dqn.add(keras.layers.Dense(units = fc1_dims, activation = "relu"))
            dqn.add(keras.layers.Dense(units = fc2_dims, activation = "relu"))
            dqn.add(keras.layers.Dense(units = len(self.action_space))) # MODIFY FOR CONT. ACTIONS
            dqn.compile(optimizer = keras.optimizers.Adam(learning_rate = self.learn_rate), \
                        loss = "mse")                    
            self.q_network = dqn
            return True
    # END DEF Build_DQN()
    
    # Train q_network with 
    def train_dqn(self, N_steps=0, ep_i=0, notify_end_train=False):
        """
            Method to train the agent's neural net, implements (deep) Q-learning with 1 target network.
            Will check if the memory buffer has at least self.batch_size sample transitions before
            performing training, will do nothing and return False otherwise.
            Can optionally notify when exiting train_on_batch() at every 200 steps of an episode.
            
            ARGUMENTS: - N_steps: Current step -1.
                       - ep_i: Current episode -1
                       - notify_end_train: Boolean for notifications
            
            NOTE: Should write detailed notes about this part.
                  Implemented for DISCRETE ACTIONS ONLY. To be modified.
            
        """
        if self.memory_buffer.mem_cntr < self.batch_size:
            #print("ERROR: Not enough samples (memory_buffer.mem_cntr < batch_size)") # DEBUG
            return False
        else:
            
            # Get minibatch of samples
            # Signature: sample_state, sample_action, sample_new_state, sample_reward, sample_terminal
            #            = Replay_Buffer.sample_buffer(batch_size)
            states, actions, next_states, rewards, dones = self.memory_buffer.sample_buffer(self.batch_size)
            
            # Init. Q(s,a), Q(s', a'), Q*(s,a)            
            q_eval = self.q_network.predict(states)
            q_next = self.q_network.predict(next_states)
            q_target = np.copy(q_eval)
            
            # Ones vector for "dones"
            # IMPORTANT REMARK: This might cause issues, related to Replay_Buffer.store_transition(),
            #                   on line: self.memory_terminal[ind] = int(done) 
            done_ones = np.ones(shape = dones.shape, dtype=np.int8)
            
            # Iterable for minibatch
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            
            # Q-learning update (CRUCIAL)
            q_target[batch_index, actions] = rewards + self.gamma*np.max(q_next, axis = 1)*(done_ones - dones)
            
            # Train network (CRUCIAL)
            self.q_network.train_on_batch(states, q_target)
            
            if ((N_steps+1)% 200)==0 and notify_end_train:
                print(f"Done training at step {N_steps+1} of episode {ep_i+1}")
            
            # Update epsilon for policy
            if self.epsilon_min < self.epsilon:
                self.epsilon -= self.epsilon_dec
            else:
                self.epsilon = self.epsilon_min
            
            return True
    
    # END DEF learn()
    
    ###################
    ## Save/Load DQN ##
    ###################
    
    # Save trained model
    def save_dqn(self):
        """
            Save q_network under dqn_fname (h5 file).
            Calls keras.models.Sequential.save()
        """
        self.q_network.save(self.dqn_fname)
    # END DEF save_dqn()
    
    # Load model
    def load_dqn(self, model_fname):
        """
            Load q_network at model_fname (h5 file).
            Calls keras.models.load_model()
        """
        self.q_network = keras.models.load_model(model_fname)
        
    # END DEF load_dqn()    
    
# END CLASS DQN_Agent