###########################
##### NEURAL NETWORKS #####
###########################
## AJ Zerouali
## Updated: 23/07/09

import math
from typing import Tuple

import numpy as np
import torch as th
from torch import FloatTensor, Tensor, nn
from torch.distributions import Distribution, Normal, MultivariateNormal
from torch.nn import functional as F

from common.utils import make_net_modules
from common.probability_distributions import TanhNormal


'''
    Basic MLP
    23/07/01
'''
class MLP(nn.Module):
    '''
        Basic MLP neural net for continuous state and action spaces,
        and deterministic actions (e.g. DDPG/TD3).
        The forward function depends on the mlp_type param,
        which is either "q_net" or "policy".
        IMPORTANT: Ensure that state_space_dim and action_space_dim
                    are the "squeezed" shapes of the env observation
                    and action spaces.
    '''
    def __init__(self,
                 mlp_type: str,
                 state_space_dim: int,
                 action_space_dim: int,
                 net_arch: list,
                 dropout_probs: list,
                 layer_activation_fn: nn.Module,
                 output_activation_fn: nn.Module,
                 weight_init_mthd: str = "",
                 weight_init_seed: int = 123,
                ) -> None:
        
        super().__init__()
        
        if mlp_type not in ["policy", "q_net"]:
            raise ValueError(f"mlp_type should be either \"policy\" or \"q_net\"")
            
        # Initial attributes
        self.mlp_type = mlp_type
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        
        # Determine network input and output dimensions
        if mlp_type == "q_net":
            input_dim = self.state_space_dim+self.action_space_dim
            output_dim = 1
        elif mlp_type == "policy" :
            input_dim = self.state_space_dim
            output_dim = self.action_space_dim
        
        # Net architecture dictionary and modules list
        self.mlp_architecture_dict = {"input_dim": input_dim,
                                      "output_dim": output_dim,
                                      "net_arch": net_arch,
                                      "dropout_probs" : dropout_probs,
                                      "layer_activation_fn": layer_activation_fn,
                                      "output_activation_fn": output_activation_fn,
                                      "weight_init_mthd": weight_init_mthd,
                                      "weight_init_seed": weight_init_seed,}
        '''
        net_modules = make_net_modules(input_dim = self.mlp_architecture_dict["input_dim"],
                                       output_dim = self.mlp_architecture_dict["output_dim"],
                                       net_arch = self.mlp_architecture_dict["net_arch"],
                                       dropout_probs = self.mlp_architecture_dict["dropout_probs"],
                                       layer_activation_fn = self.mlp_architecture_dict["layer_activation_fn"],
                                       output_activation_fn = self.mlp_architecture_dict["output_activation_fn"],
                                       weight_init_mthd = self.mlp_architecture_dict["weight_init_mthd"],
                                       weight_init_seed = self.mlp_architecture_dict["weight_init_seed"],
                                      )
        '''
        net_modules = make_net_modules(**self.mlp_architecture_dict)
        
        # Assign main neural net
        self.net = nn.Sequential(*net_modules)
        #self.net.to(dtype = th.float64)
        
        # Determine forward function
        if mlp_type == "q_net":
            self.forward = self.q_net_forward
        elif mlp_type == "policy":
            self.forward = self.policy_forward

    
    # Policy forward function
    def policy_forward(self, state: th.Tensor)->th.Tensor:
        '''
            Forward function for a policy. 
        '''
        return self.net.forward(state.float())
    
    # Q-net forward function
    def q_net_forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        '''
           Forward function for Q-network.
        '''
        #state_
        state_action = th.cat([state, action], dim = -1).float()
        '''
        # DEBUG
        print(f"state.dtype = {state.dtype}")
        print(f"action.dtype = {action.dtype}")
        print(f"state_action.dtype = {state_action.dtype}")
        print(f"self.net[0].weight.data.dtype = {self.net[0].weight.data.dtype}")
        '''
        return self.net(state_action)


'''
    GAUSSIAN POLICY
    23/07/06
'''
class GaussianPolicy(nn.Module):
    '''
        Gaussian policy with diagonal covariance matrix.
        Consists of the following parts:
        A) Feature extractor neural network, action mean layer, log of standard
        deviation as a learnable parameter.
        B) Pytorch distribution for sampling of stochastic actions.
        C) Hyperbolic tangent transformation for the scaling of actions. Important 
        for Mujoco environments.
        
        REMARK: The activation function for the output mean and log-std layers
        is the identity. We omit the addition of these activation functions.
        
        To do (23/07/04):
        1) Implement the networks.
        2) Implement the main probability distribution.
        3) Implement Tanh transformation of sampling
        4) Implement initialization of the mu layer.
        
        NOTES:
        - Should maybe add 
    '''
    def __init__(self,
                 state_space_dim: int,
                 action_space_dim: int,
                 net_arch: list,
                 dropout_probs: list,
                 layer_activation_fn: nn.Module = nn.ReLU,
                 log_sigma_bound: float = 2.0,
                 use_tanh_scaling: bool = False,
                 weight_init_mthd: str = "",
                 weight_init_seed: int = 123,
                ) -> None:
        
        super().__init__()
        
        # Tanh scaling
        self.use_tanh_scaling = use_tanh_scaling
        '''
            ## IMPORTANT: Assign methods
            To do (23/07/04): Implement the required methods
        '''
        if self.use_tanh_scaling:
            self.get_prob_distribution = self.get_tanhgauss_distribution
        else:
            self.get_prob_distribution = self.get_gaussian_distribution
        
        # Log-sigma bound
        self.log_sigma_bound = log_sigma_bound
        
        # Dimensions
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        if len(net_arch)>0:
            self.out_layer_in_dim = net_arch[-1]
        elif len(net_arch)==0:
            self.out_layer_in_dim = self.state_space_dim
        
        # Feature extractor network (MLP)
        ## Make architecture dictionary from input params
        ## IMPORTANT: Check what this does if len(net_arch)==0
        self.feat_extr_arch_dict = {"input_dim": self.state_space_dim,
                                    "output_dim": 0,
                                    "net_arch": net_arch,
                                    "dropout_probs" : dropout_probs,
                                    "layer_activation_fn": layer_activation_fn,
                                    "output_activation_fn": nn.Identity,
                                    "weight_init_mthd": weight_init_mthd,
                                    "weight_init_seed": weight_init_seed,}
        ## Make feature extractor net modules from arch. dict.
        feature_extractor_modules = make_net_modules(**self.feat_extr_arch_dict)
        ## Instantiate feature extractor network
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)
        
        # Mean and log sigma layers
        ## Instantiate mu output layer
        self.mu_out_layer = nn.Linear(self.out_layer_in_dim, self.action_space_dim)
        ## 
        self.log_sigma_out_layer = nn.Linear(self.out_layer_in_dim, self.action_space_dim)
        ## Initialize layers
        self._init_out_layers(weight_init_mthd)
        
        # Probability distribution
        ## This is instantiated by get_probability_distribution()
        self.probability_distribution = None
    
    ############################
    ### Forward pass methods ###
    ############################
    def forward(self, state: th.Tensor) -> Normal:
        '''
            Compute the stochastic policy corresponding
            to input state. Note that the 
            
            :param state: th.Tensor. Current state of the environment
            :return probability_distribution: Stochastic policy probability distrib'n.
                If use_tanh_scaling is False, this is a regular gaussian distribution,
                whose mean is given by the neural net and standard deviation by the
                log_sigma learnable param.
        '''
        # Compute the mean (mu)
        state_features = self.feature_extractor(state)
        mu = self.mu_out_layer(state_features)
        # Compute the covariance matrix (Sigma)
        '''
        ## IMPORTANT: What should we do about the batches?
        '''
        log_sigma = self.log_sigma_out_layer(state_features)
        log_sigma = th.clamp(log_sigma, -self.log_sigma_bound, self.log_sigma_bound)
        sigma = th.exp(log_sigma)
        #Sigma = th.diag_embed(sigma_diag) # If MultivariateNormal is used instead of Normal
        # Compute distribution
        '''
        ## IMPORTANT: 
            - If use_tanh_scaling = False, get_prob_distribution = get_gaussian_distribution
            - If use_tanh_scaling = True, get_prob_distribution = get_tanhgauss_distribution
        '''
        self.get_prob_distribution(mean = mu, std = sigma)
        
        # Output
        return self.probability_distribution
    
    def forward_mu(self, state: th.Tensor) -> th.Tensor:
        state_features = self.feature_extractor(state)
        return self.mu_out_layer(state_features)
    
    def forward_log_sigma(self, state: th.Tensor) -> th.Tensor:
        state_features = self.feature_extractor(state)
        return self.log_sigma_out_layer(state_features)
    
    ###################################
    ### Output layer init. function ###
    ###################################
    def _init_out_layers(self, weight_init_mthd: str):
        '''
            Initialization function for mu_out_layer and log_sigma_out_layer.
            
        '''
        
        # Weight initialization type
        ### Temporary: Supports Xavier and Kaiming only for now
        admissible_initializations = ["Xavier_normal", "Xavier_uniform", 
                                      "Kaiming_normal", "Kaiming_uniform"]
        no_wt_initialization_list = [None, "", "none", "None"]
        
        if weight_init_mthd not in no_wt_initialization_list:
            
            # Verify that init. method is supported
            if weight_init_mthd not in admissible_initializations:
                raise NotImplementedError(f"Unsupported weight initialization method.\n"\
                                          f"Supported methods: {admissible_initializations}"
                                         )
            else:
                # Note: It is recommended not to use Kaiming initialization if the activation
                # funtion is not some type of ReLU.
                
                # Define weight initialization function
                if weight_init_mthd == "Xavier_normal":
                    nn.init.xavier_normal_(tensor = self.mu_out_layer.weight, 
                                           gain = 1.0)
                    nn.init.xavier_normal_(tensor = self.log_sigma_out_layer.weight, 
                                           gain = 1.0)
                elif weight_init_mthd == "Xavier_uniform":
                    nn.init.xavier_uniform_(tensor = self.mu_out_layer.weight, 
                                           gain = 1.0)
                    nn.init.xavier_uniform_(tensor = self.log_sigma_out_layer.weight, 
                                           gain = 1.0)
                elif weight_init_mthd == "Kaiming_normal":
                    pass
                elif weight_init_mthd == "Kaiming_uniform":
                    pass
    
    '''
       #### PyTorch distribution methods: ####
        - get_gaussian_distribution()
        - get_tanhgauss_distribution()
        
        (23/07/04) Ensure the latter has the following
                methods:
        - sample()
        - rsample()
        - entropy() returning None
        - log_prob()
    '''
    def get_gaussian_distribution(self, mean: th.Tensor, std = th.Tensor)->None:
        '''
            Instantiate gaussian probability distribution
            corresponding to specified mean and standard deviation,
            and assign it to the probability_distribution attribute
            
            Question (23/07/04): Should the gradients be on?
        '''
        self.probability_distribution = Normal(loc = mean, scale = std)
   
    def get_tanhgauss_distribution(self, mean: th.Tensor, std = th.Tensor)->None:
        '''
            Instantiate probability distribution corresponding to tanh of
            a gaussian with specified mean and standard deviation,
            and assign it to the probability_distribution attribute.
            
            Question (23/07/04): Should the gradients be on?
        '''
        self.probability_distribution = TanhNormal(loc = mean, scale = std)
    
'''
    Generalized State-Dependent Exploration Policy
    23/07/10
    
    To do (23/07/06):
    1) Constructor
        1.a - Particular importance needs to be given to the
            shape of the log_sigma parameter
        1.b - Be careful with use_tanh_scaling
    2) Forward:
        2.a - Forward pass to compute state_features
        2.b - Get action mean by applying mu_out_layer to state_features
        2.c - Compute the additive noise: th.bmm(xi_weights, state_features)
    3) Tanh actions and non-tanh actions - How do I deal with these
    4) A method that ASSIGNS the stochastic weight distribution.
        You don't return this distribution.  ()
    5) A method that calls the rsample() of the weight distribution.
        This method will be called every n_weight_resample_steps steps
    6) A method that updates the random weights. It will call (4)-(5) to:
        6.a - Update the weight distribution.
        6.b - Sample the weights.
        The idea is to call this method at the beginning of the episode and 
        once (n_steps % n_weight_resample_steps)==0 during rollout collection.
    7) Technically, this class also models a Gaussian policy, but instead 
        of sampling the action at EVERY step you sample only the noise at 
        fixed steps (and at the beginning of the episode).
    8) The log likelihood of the actions produced by this policy might be 
        needed for SAC and similar algos.
    9) If xi denotes the noise matrix (xi.shape = (n_features, action_dim)), and if 
        state_batch_features = feature_extractor(state_batch) (state_batch.shape = (batch_size, n_features))
        then th.mm does exactly what it is supposed to; i.e. if we take:
        mm_prod = th.mm(state_batch_features, xi)
        then: mm_prod.shape = (batch_size, action_dim), and most importantly:
        mm_prod[i,:].view(1,-1) = th.mm(state_batch_features[i,:].view(1,-1), xi)
    10) To compute the action variance, use the fact that the entries of the xi matrix
        are all independent. Let f(s) = f(s,theta) = [f_1(s), .., f_m(s)]
        and let [sigma_ij] denote the std matrix of xi. Since the i-th component
        of an action is given by:
        a_i = mu_i(s,theta) + sum_j f_j(s).xi_ji, we have: E[a_i] = mu_i(s),
        and Var(a_i) = sum_j f_j(s)^2.(sigma_ji)^2 since xi_ji ~ N(0, (sigma_ji)^2)
        
'''
class GSDEPolicy(nn.Module):
    '''
        Generalized state-dependent exploration policy (gSDE), as published in
        A. Raffin et al. 2020: https://arxiv.org/abs/2005.05719.
        The present implementation is a simplification of the one provided in
        stable_baselines3 (https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3).
        This policy consists of:
        - A feature extractor network (feature_extractor).
        - An action mean layer (mu_out_layer).
        - Log standard deviation parameter for the generalized SDE noise.
        - An output activation function to scale actions (output_activation_fn).
        
        IMPORTANT CAVEATS: 
        1) This policy is very sensitive to the following quantities:
        - The initial value of the log_sigma learnable parameter.
        - The log_sigma_bound parameter, used to clip the standard deviation
          of the exploration noise distribution.
        - The action_variance_min and action_variance_max parameters, without 
          which the standard deviation of the action distribution takes NaN values.
        If these parameters are not chosen carefully, the learning either 
        crashes or is extremely slow.
        2) This module class has two distinct probability distributions:
        2.a - The normal distribution for the random weights of the model,
        whose standard deviation is computed by clipping and exponentiating
        the log_sigma learnable parameter. 
        2.b - The action distribution, which is either a normal distribution
        or the distribution of the Tanh of a normal random variable.
        3) Random weights distribution: Before calling the forward() method, which
        computes the action according to the Raffin et al. paper, the random weights
        distribution should be updated with the set_random_weights_distribution()
        method, and the random weights should be sampled using the 
        (r)sample_random_weights() method(s).
        4) Action distribution: Learning is faster when the gradient updates of the
        policy network parameters are computed using the action distribution. When 
        implementing an RL algorithm that uses this policy, it is recommended that
        the actions be sampled using: GSDEPolicy.get_action_distribution().rsample().
        See the docstring of get_action_distribution() for details.
    '''
    def __init__(self,
                 state_space_dim: int,
                 action_space_dim: int,
                 n_weight_resample_steps: int,
                 net_arch: list,
                 dropout_probs: list,
                 layer_activation_fn: nn.Module = nn.ReLU,
                 log_sigma_init: float = 2.0,
                 log_sigma_bound: float = 2.0,
                 action_variance_min: float = 1e-6,
                 action_variance_max: float = 1e4,
                 use_tanh_scaling: bool = False,
                 weight_init_mthd: str = "",
                 weight_init_seed: int = 123,
                ) -> None:
        
        super().__init__()
        
        # Log-sigma and action variance attributes
        self.log_sigma_init = log_sigma_init
        self.log_sigma_bound = log_sigma_bound
        self.action_variance_min = action_variance_min
        self.action_variance_max = action_variance_max
        
        # Dimensions
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        if len(net_arch)>0:
            self.out_layer_in_dim = net_arch[-1]
        elif len(net_arch)==0:
            self.out_layer_in_dim = self.state_space_dim
        self.n_weight_resample_steps = n_weight_resample_steps
        
        # Feature extractor network (MLP)
        ## Make architecture dictionary from input params
        self.feat_extr_arch_dict = {"input_dim": self.state_space_dim,
                                      "output_dim": 0,
                                      "net_arch": net_arch,
                                      "dropout_probs" : dropout_probs,
                                      "layer_activation_fn": layer_activation_fn,
                                      "output_activation_fn": nn.Identity,
                                      "weight_init_mthd": weight_init_mthd,
                                      "weight_init_seed": weight_init_seed,}
        ## Make feature extractor net modules from arch. dict.
        feature_extractor_modules = make_net_modules(**self.feat_extr_arch_dict)
        ## Instantiate feature extractor network
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)
        
        # Mean layer (mu)
        ## Instantiate mu output layer
        self.mu_out_layer = nn.Linear(self.out_layer_in_dim, self.action_space_dim)
        ## Initialize mu layer
        self._init_mu_layer(weight_init_mthd)
        
        # GSDE noise log std parameter
        self.log_sigma = nn.Parameter(
            th.ones(size = (self.out_layer_in_dim, self.action_space_dim))\
            *(self.log_sigma_init)
        )
        
        # Output activation
        self.use_tanh_scaling = use_tanh_scaling
        if self.use_tanh_scaling:
            self.output_activation_fn = nn.Tanh()
            self.get_action_distribution = self.get_tanhgauss_action_distribution
        else:
            self.output_activation_fn = nn.Identity()
            self.get_action_distribution = self.get_gaussian_action_distribution
            
        # Random weights distribution and random weights
        self.random_weights_distribution = None
        self.xi = None ## CRUCIAL: These are the random weights
    
    
    ############################
    ### Forward pass methods ###
    ############################
    def forward(self, 
                state: th.Tensor,
               ) -> th.Tensor:
        '''
            Compute the action with a (generalized) state-dependent 
            exploration noise term. 
            
            NOTE: 
            1) The noise matrix self.xi must be sampled prior to calling this method.
            2) The input state MUST have shape (batch_size, state_space_dim).
            
            
            :param state: th.Tensor. Current state of the environment
            :return action: th.Tensor. Noisy action.
        
        '''
        # Compute the mean (mu)
        state_features = self.feature_extractor(state)
        mu = self.mu_out_layer(state_features)
        noise_term = th.mm(state_features, 
                           self.xi.to(self.get_device())
                          )
        action = self.output_activation_fn(mu+noise_term)
        
        # Output
        return action
        
    
    def forward_mu(self, state: th.Tensor) -> th.Tensor:
        '''
            Evaluate the output of the mu layer
            applied to state features.
            
            COMMENTS: 
            - This method DOES NOT apply the output activation
              function.
            - This method is designed to evaluate the mean of 
              the actions used to instantiate distributions.
              See forward() for the correct computation of actions.
            
        '''
        state_features = self.feature_extractor(state)
        return self.mu_out_layer(state_features)
    
    
    ##############################
    ### Random weights methods ###
    ##############################
    def set_random_weights_distribution(self)->None:
        # TO DO (23/0/07): Ensure this is not problematic during training
        #self.log_sigma = self.log_sigma.clamp(-self.log_sigma_bound,self.log_sigma_bound) # Need convert to param
        sigma = th.exp(th.clamp(self.log_sigma, 
                                -self.log_sigma_bound,
                                self.log_sigma_bound)
                      )
        self.sigma = sigma
        self.random_weights_distribution = Normal(loc = th.zeros(size = sigma.shape).to(sigma.device),
                                                 scale = sigma)
    
    def rsample_random_weights(self)->None:
        self.xi = self.random_weights_distribution.rsample()
    
    def sample_random_weights(self)->None:
        self.xi = self.random_weights_distribution.sample()
    

        
    ###################################
    ### Action distribution methods ###
    ###################################
    def compute_action_gaussian_std(self, state: th.Tensor
                                        )->th.Tensor:
        '''
            Method to compute the standard deviation of the Gaussian
            random variable underlying the noisy actions.
            
            COMMENTS: Let f(s) = f(s,theta) = [f_1(s), .., f_m(s)]
            and let [sigma_ij] denote the std matrix of the noise matrix xi = [xi_ij].
            Suppose for simplicity we do not use tanh scaling of actions.
            Since the i-th component of an action is given by:
            a_i = mu_i(s,theta) + sum_j f_j(s).xi_ji,
            then we have: E[a_i] = mu_i(s) for the mean, and:
            Var(a_i) = sum_j f_j(s)^2.(sigma_ji)^2 
            since xi_ji ~ N(0, (sigma_ji)^2) and are independent.
            
            NOTE: Be careful with the shape of the state param.
        '''
        state_features = self.feature_extractor(state)
        log_sigma = th.clamp(self.log_sigma, 
                                -self.log_sigma_bound,
                                self.log_sigma_bound)
        action_std = th.sqrt(th.mm(state_features**2,
                                   th.exp(log_sigma)**2)
                            )
        return action_std
    
    def get_gaussian_action_distribution(self, state: th.Tensor)->Normal:
        '''
            Get Tanh of Gaussian action distribution.
            Used for learning.
            
            NOTE: If the bounds for log_sigma and action_variance
            are not chosen carefully this method can crash.
        '''
        # Feature extractor forward
        state_features = self.feature_extractor(state)
        
        # Get action standard deviation
        log_sigma = th.clamp(self.log_sigma,
                             -self.log_sigma_bound,
                             self.log_sigma_bound)
        action_variance = th.mm(state_features**2, th.exp(log_sigma)**2)
        action_variance = th.clamp(action_variance, 
                                   min=self.action_variance_min, 
                                   max=self.action_variance_max)
        action_sigma = th.sqrt(action_variance).to(state.device)
        
        # DEBUG
        self.action_sigma = action_sigma
        
        # Get action mean
        action_mu = self.mu_out_layer(state_features).to(state.device)
        
        return Normal(loc = action_mu, scale = action_sigma)
        
        
    def get_tanhgauss_action_distribution(self, state: th.Tensor)->TanhNormal:
        '''
            Get Tanh of Gaussian action distribution.
            Used for learning.
            
            NOTE: If the bounds for log_sigma and action_variance
            are not chosen carefully this method can crash.
        '''
        # Feature extractor forward
        state_features = self.feature_extractor(state)
        
        # Get action standard deviation
        log_sigma = th.clamp(self.log_sigma,
                             -self.log_sigma_bound,
                             self.log_sigma_bound)
        action_variance = th.mm(state_features**2, th.exp(log_sigma)**2)
        action_variance = th.clamp(action_variance, 
                                   min=self.action_variance_min, 
                                   max=self.action_variance_max)
        action_sigma = th.sqrt(action_variance).to(state.device)
        
        # DEBUG
        self.action_sigma = action_sigma
        
        # Get action mean
        action_mu = self.mu_out_layer(state_features).to(state.device)
        
        return TanhNormal(loc = action_mu, scale = action_sigma)
    
    ###################################
    ### Output layer init. function ###
    ###################################
    def _init_mu_layer(self, weight_init_mthd: str):
        '''
            Initialization function for mu_out_layer and log_sigma_out_layer.
            
        '''
        
        # Weight initialization type
        ### Temporary: Supports Xavier and Kaiming only for now
        admissible_initializations = ["Xavier_normal", "Xavier_uniform", 
                                      "Kaiming_normal", "Kaiming_uniform"]
        no_wt_initialization_list = [None, "", "none", "None"]
        
        if weight_init_mthd not in no_wt_initialization_list:
            
            # Verify that init. method is supported
            if weight_init_mthd not in admissible_initializations:
                raise NotImplementedError(f"Unsupported weight initialization method.\n"\
                                          f"Supported methods: {admissible_initializations}"
                                         )
            else:
                # Note: It is recommended not to use Kaiming initialization if the activation
                # funtion is not some type of ReLU.
                
                # Define weight initialization function
                if weight_init_mthd == "Xavier_normal":
                    nn.init.xavier_normal_(tensor = self.mu_out_layer.weight, 
                                           gain = 1.0)
                elif weight_init_mthd == "Xavier_uniform":
                    nn.init.xavier_uniform_(tensor = self.mu_out_layer.weight, 
                                           gain = 1.0)
                elif weight_init_mthd == "Kaiming_normal":
                    pass
                elif weight_init_mthd == "Kaiming_uniform":
                    pass
                
    #################################
    ### Get current device method ###
    #################################
    def get_device(self):
        '''
            Quick and dirty way of getting the 
            policy device o
        '''
        return next(self.parameters()).device