##########################################
##### Monte Carlo in Windy GridWorld #####
##########################################
## 2022/04/29, AJ Zerouali
## Monte Carlo functions adapted to the  
## Windy GridWorld environment.

from __future__ import print_function, division
from builtins import range

import numpy as np

# Import the windy GridWorld class
from Windy_GridWorld import GridWorld_Windy_small

# Import helper functions from RL_Fns_Windy_GridWorld.py
#from RL_Fns_Windy_GridWorld import 


########################
### GENERATE EPISODE ###
########################
# 2022/04/26, AJ Zerouali
# Format: Epsd = [(0, s_0, a_0), (r_1, s_1, a_1), ..., (r_T, s_T, '')]
# 

def generate_episode(s_0, a_0, Pi, env, T_max):
    '''
     Generates a random episode of max length (T_max + 1), given an initial state-action.
     ARGUMENTS: Initial state and action; policy; environment; max episode length (in this order).
     OUTPUT: episode_rewards: Rewards list
             episode_states: State list
             episode_actions: Actions list
             T: Episode length minus one.
             
     NOTE: The function below is taylored to GridWorld. The correct way
          to implement it is to use methods of the environment class.
          I'm not using the methods of the Windy_GridWorld class because
          the instructor's implementation is a little too clunky and the
          entire design should be redone from scratch.
    '''
    
    # Environment attributes
    non_term_states = env.non_term_states
    term_states = env.term_states
    adm_actions = env.adm_actions
    Rwds = env.rewards
    
    # Step t=0
    s_new = s_0
    a_new = a_0
    r_new = 0
    
    # Init. episode lists (format: step_t = (r_t, s_t, a_t)) and store Step 0
    episode_rewards = [r_new]
    episode_states = [s_new]
    episode_actions = [a_new]
    
        
    # Init. episode length
    T = 0
    
    ##### 0<t
    while (s_new not in term_states) and (T<T_max):
        
        # Init. old step
        r_old = r_new
        s_old = s_new
        a_old = a_new
        
        # WARNING: Modify for other environments
        # Compute new state
        if a_old == 'U':
            s_new = (s_old[0]-1, s_old[1])
        elif a_old == 'D':
            s_new = (s_old[0]+1, s_old[1])
        elif a_old == 'L':
            s_new = (s_old[0], s_old[1]-1)
        elif a_old == 'R':
            s_new = (s_old[0], s_old[1]+1)
        
        # Compute new action
        if s_new in non_term_states:
            a_new = list(Pi[s_new].keys())[0]
        elif s_new in term_states:
            a_new = ''
        
        # Compute new reward
        r_new = Rwds.get(s_new, 0)
        
        # Add step to episode
        episode_rewards.append(r_new)
        episode_states.append(s_new)
        episode_actions.append(a_new)
        
        # Update
        T += 1
        
    # END WHILE
    
    # Output line
    return episode_rewards, episode_states, episode_actions, T
    
# END DEF generate_episode()

###############################################
### GENERATE EPISODE FROM STOCHASTIC POLICY ###
###############################################
# 2022/05/03, AJ Zerouali
# Format: Epsd = [(0, s_0, a_0), (r_1, s_1, a_1), ..., (r_T, s_T, '')]
# 

def generate_episode_stochastic(s_0, a_0, Pi, env, T_max):
    '''
     Generates a random episode of max length (T_max + 1), given an initial state-action.
     ARGUMENTS: Initial state and action; stochastic policy; environment; max episode length (in this order).
     OUTPUT: episode_rewards: Rewards list
             episode_states: State list
             episode_actions: Actions list
             T: Episode length minus one.
             
     NOTE: The function below is taylored to GridWorld. The correct way
          to implement it is to use methods of the environment class.
    '''
    
    # Environment attributes
    non_term_states = env.non_term_states
    term_states = env.term_states
    adm_actions = env.adm_actions
    Rwds = env.rewards
    
    # Step t=0
    s_new = s_0
    a_new = a_0
    r_new = 0
    
    # Init. episode lists (format: step_t = (r_t, s_t, a_t)) and store Step 0
    episode_rewards = [r_new]
    episode_states = [s_new]
    episode_actions = [a_new]
    
        
    # Init. episode length
    T = 0
    
    ##### 0<t
    while (s_new not in term_states) and (T<T_max):
        
        # Init. old step
        r_old = r_new
        s_old = s_new
        a_old = a_new
        
        # WARNING: Modify for other environments
        # Compute new state
        if a_old == 'U':
            s_new = (s_old[0]-1, s_old[1])
        elif a_old == 'D':
            s_new = (s_old[0]+1, s_old[1])
        elif a_old == 'L':
            s_new = (s_old[0], s_old[1]-1)
        elif a_old == 'R':
            s_new = (s_old[0], s_old[1]+1)
        
        # Compute new action
        ###################################
        # CHANGE THIS FOR STOCHASTIC POLICY
        ###################################
        if s_new in non_term_states:
            # Here we are assuming that Pi[s].keys() is the same as adm_actions[s]
            actions_list_s_new = list(Pi[s_new].keys())
            probs_list_s_new = list(Pi[s_new].values())
            rand_ind = np.random.choice(len(actions_list_s_new), p = probs_list_s_new)
            # Pick action from random index
            a_new = actions_list_s_new[rand_ind]
        elif s_new in term_states:
            a_new = ''
        
        # Compute new reward
        r_new = Rwds.get(s_new, 0)
        
        # Add step to episode
        episode_rewards.append(r_new)
        episode_states.append(s_new)
        episode_actions.append(a_new)
        
        # Update
        T += 1
        
    # END WHILE
    
    # Output line
    return episode_rewards, episode_states, episode_actions, T
    
# END DEF generate_episode()

######################################
###  MONTE CARLO POLICY EVALUATION ###
######################################
## 2022/04/26, AJ Zerouali

def MC_Policy_Eval(Pi, env, gamma, N_samples, T_max, all_visits_MC):
    '''
    Function implementing Monte Carlo policy evaluation. Generates 
    a specified number of sample episodes and averages returns to 
    evaluate a given policy. 
    Can choose first visit MC or all visits MC with Boolean.
    
     ARGUMENTS: - Pi, env, gamma: Policy, environment, discount factor.
                - N_samples: No. of samples for Monte Carlo expectation.
                - T_max: Max. episode length minus 1.
                - all_visits_MC: Boolean for all visits or first visit MC.
     OUTPUT:    - V:=V_Pi, value function obtained.
     NOTE: This function calls generate_episode().
    '''
    
    # Environment attributes
    non_term_states = env.non_term_states
    term_states = env.term_states
    adm_actions = env.adm_actions
    Rwds = env.rewards
    
    # Init output and returns
    V = {}
    Returns = {}
    for s in term_states:
        V[s] = 0.0
    for s in non_term_states:
        V[s] = 0.0
        Returns[s]=[]
        # Init. counter
        N_iter = 0
    
    # Main MC loop
    while N_iter < N_samples:

        ##########################
        ##### Step t=0 setup #####
        ##########################

        # Count no. of non_term_states
        N_non_term_states = len(non_term_states)

        # Generate s_0 randomly from non_term_states, get a_0 from policy
        s_0 = list(non_term_states)[np.random.randint(N_non_term_states)]
        a_0 = list(Pi[s_0].keys())[0]

        ########################
        ##### Steps 1 to T #####
        ########################

        # Generate episode
        # Signature: episode_rewards, episode_states, episode_actions, T = generate_episode(s_0, a_0, Pi, env, T_max)
        episode_rewards, episode_states, episode_actions, T = generate_episode(s_0, a_0, Pi, env, T_max)
        # Step t of episode is (r_t, s_t, a_t) = (episode_rewards[t], episode_states[t], episode_actions[t])

        ##################################
        ### COMPUTE CUMULATIVE RETURNS ###
        ##################################

        # Init. storing variable
        G = 0.0
        
        # First visit only MC
        if not all_visits_MC:
            # Loop over episode
            for t in range(T-1, -1, -1): # Loop goes backwards from (T-1) to 0
                G = gamma*G + episode_rewards[t+1]
                s_t = episode_states[t]
                
                if s_t not in episode_states[:t]:
                    Returns[s_t].append(G)
                    V[s_t] = np.average(Returns[s_t])
                    
        # All visits MC
        elif all_visits_MC:
            for t in range(T-1, -1, -1): # Loop goes backwards from (T-1) to 0
                G = gamma*G + episode_rewards[t+1]
                s_t = episode_states[t]
                Returns[s_t].append(G)
                V[s_t] = np.average(Returns[s_t])
        
        # Update N_iter
        N_iter += 1

    # END WHILE of MC loop
    
    # Output
    return V

# END DEF MC_Policy_Eval

######################################
###  MONTE CARLO POLICY EVALUATION ###
######################################
## 2022/05/03, AJ Zerouali

def MC_Stochastic_Policy_Eval(Pi, env, gamma, N_samples, T_max, all_visits_MC):
    '''
    Function implementing Monte Carlo policy evaluation. Generates 
    a specified number of sample episodes and averages returns to 
    evaluate a given stochastic policy. 
    Can choose first visit MC or all visits MC with Boolean.
    
     ARGUMENTS: - Pi, env, gamma: Policy, environment, discount factor.
                - N_samples: No. of samples for Monte Carlo expectation.
                - T_max: Max. episode length minus 1.
                - all_visits_MC: Boolean for all visits or first visit MC.
     OUTPUT:    - V:=V_Pi, value function obtained.
     NOTE: This function calls generate_episode_stochastic().
    '''
    
    # Environment attributes
    non_term_states = env.non_term_states
    term_states = env.term_states
    adm_actions = env.adm_actions
    Rwds = env.rewards
    
    # Init output and returns
    V = {}
    Returns = {}
    for s in term_states:
        V[s] = 0.0
    for s in non_term_states:
        V[s] = 0.0
        Returns[s]=[]
        # Init. counter
        N_iter = 0
    
    # Main MC loop
    while N_iter < N_samples:

        ##########################
        ##### Step t=0 setup #####
        ##########################

        # Count no. of non_term_states
        N_non_term_states = len(non_term_states)

        # Generate s_0 randomly from non_term_states, get a_0 from policy
        s_0 = list(non_term_states)[np.random.randint(N_non_term_states)]
        a_0 = list(Pi[s_0].keys())[0]

        ########################
        ##### Steps 1 to T #####
        ########################

        # Generate episode
        # Signature: episode_rewards, episode_states, episode_actions, T = generate_episode(s_0, a_0, Pi, env, T_max)
        episode_rewards, episode_states, episode_actions, T = generate_episode_stochastic(s_0, a_0, Pi, env, T_max)
        # Step t of episode is (r_t, s_t, a_t) = (episode_rewards[t], episode_states[t], episode_actions[t])

        ##################################
        ### COMPUTE CUMULATIVE RETURNS ###
        ##################################

        # Init. storing variable
        G = 0.0
        
        # First visit only MC
        if not all_visits_MC:
            # Loop over episode
            for t in range(T-1, -1, -1): # Loop goes backwards from (T-1) to 0
                G = gamma*G + episode_rewards[t+1]
                s_t = episode_states[t]
                
                if s_t not in episode_states[:t]:
                    Returns[s_t].append(G)
                    V[s_t] = np.average(Returns[s_t])
                    
        # All visits MC
        elif all_visits_MC:
            for t in range(T-1, -1, -1): # Loop goes backwards from (T-1) to 0
                G = gamma*G + episode_rewards[t+1]
                s_t = episode_states[t]
                Returns[s_t].append(G)
                V[s_t] = np.average(Returns[s_t])
        
        # Update N_iter
        N_iter += 1

    # END WHILE of MC loop
    
    # Output
    return V

# END DEF MC_Stochastic_Policy_Eval


##############################################################
###  MONTE CARLO CONTROL WITH EXPLORING STARTS - VERSION 3 ###
##############################################################
## 2022/05/02, AJ Zerouali
# Monte Carlo with exploring starts (Lect. 64-65)
# We are not using convergence thresholds here.
# This function below generates a specified number of sample
# episodes and averages returns to find the optimal policy.
# Can choose first visit MC or all visits MC with Boolean.
# This function also calls generate_episode() of previous sec.

def MC_Ctrl_ExpStarts(Pi, env, gamma, N_samples, T_max, all_visits_MC):
    '''
     Monte Carlo control with exploring starts
     ARGUMENTS: - Pi, env, gamma: Policy, environment, discount factor.
                - N_samples: No. of samples for Monte Carlo expectation.
                - T_max: Max. episode length minus 1.
                - all_visits_MC: Boolean for all visits or first visit MC.
     OUTPUT:    - Pi_star: Optimal policy
                - V_star: Optimal value function
                - Q: State-action values from samples
                - Returns: Dict. of return samples (by init. state-action)
                - N_iter: No. of randomly generated episodes (<= N_samples)
     NOTE: This function calls generate_episode().
    '''
    
    # Environment attributes
    non_term_states = env.non_term_states
    term_states = env.term_states
    adm_actions = env.adm_actions
    Rwds = env.rewards
    
    
    # Init. Q, returns, Pi_star, and V_star dictionaries
    Pi_star = {}
    Q = {}
    Returns = {}
    V_star = {}
    for s in non_term_states:
        Pi_star[s] = {}
        Q[s]={}
        Returns[s] = {}
        V_star[s] = 0.0
        for a in adm_actions[s]:
            Q[s][a] = 0.0
            Returns[s][a]=[]
            
    for s in term_states:
        V_star[s] = 0.0
            
    
    # Init. counter
    N_iter = 0
    
    # Main MC loop
    while (N_iter<N_samples):
                
        ##########################
        ##### Step t=0 setup #####
        ##########################

        # Generate (s_0, a_0) randomly from non_term_states and adm_actions
        # np.random.choice() works only for 1-dim'l arrays
        s_0 = list(non_term_states)[np.random.randint(len(non_term_states))]
        a_0 = np.random.choice(adm_actions[s_0])

        ########################
        ##### Steps 1 to T #####
        ########################
        
        #print(f"Generating sample episode no. {N_iter+1}...\n") # Debug
        # Generate episode
        # Signature: episode_rewards, episode_states, episode_actions, T = generate_episode(s_0, a_0, Pi, env, T_max)
        episode_rewards, episode_states, episode_actions, T = generate_episode(s_0, a_0, Pi, grid, T_max)
        # Step t of episode is (r_t, s_t, a_t) = (episode_rewards[t], episode_states[t], episode_actions[t])
        #rint(f"Sample episode N_iter={N_iter} has T={T} steps after t=0.\n") # Debug
 
        #####################################################
        ### COMPUTE CUMULATIVE RETURNS AND OPTIMAL ACTION ###
        #####################################################

        # First visit only MC
        if not all_visits_MC:
            # State-action iterable
            episode_state_actions = list(zip(episode_states, episode_actions))
            
            # Init. storing variable
            G = 0.0
            
            # Loop over episode
            for t in range(T-1, -1, -1): # Loop goes backwards from (T-1) to 0
                
                # Sum the return
                G = gamma*G + episode_rewards[t+1]
                s_t = episode_states[t]
                a_t = episode_actions[t]
                
                if (s_t, a_t) not in episode_state_actions[:t]:
                    
                    # Update sample returns and Q-function
                    Returns[s_t][a_t].append(G)
                    Q[s_t][a_t] = np.average(Returns[s_t][a_t])
                    
                    
                    # Get a_star and update Pi_star
                    a_star = max(Q[s_t], key = Q[s_t].get)
                    Pi_star[s_t] = {a_star:1.0} 
                    
                    
        # END IF first visit MC
        
        # All visits MC
        elif all_visits_MC:
            
            # Init. storing variable
            G = 0.0
            
            for t in range(T-1, -1, -1): # Loop goes backwards from (T-1) to 0
                
                # Sum the returns
                G = gamma*G + episode_rewards[t+1]
                s_t = episode_states[t]
                a_t = episode_actions[t]
                
                # Update sample returns and Q-function
                Returns[s_t][a_t].append(G)
                Q[s_t][a_t] = np.average(Returns[s_t][a_t])
                
                
                # Get a_star and update Pi_star
                a_star = max(Q[s_t], key = Q[s_t].get)
                Pi_star[s_t] = {a_star:1.0} 
                
                              
        # Update N_iter
        N_iter += 1
        
        '''
        # DEBUG:
        print(f"Completed episode N_iter={N_iter} with R={G}.")
        print("--------------------------------------------------------------------")
        '''
       
    # END WHILE of MC loop
    
    # COMPUTE V
    for s in non_term_states:
        a = max(Q[s], key = Q[s].get)
        V_star[s]=Q[s][a]
    
    # Output
    return Pi_star, V_star, Q, Returns, N_iter

# END DEF MC_Ctrl_ExpStarts



#######################################################
###  MONTE CARLO CONTROL WITH EPSILON-GREEDY POLICY ###
#######################################################
## 2022/05/02, AJ Zerouali
# Monte Carlo without exploring starts (Lect. 64-65), following
# an epsilon greedy scheme.
# We are not using convergence thresholds here.
# This function below generates a specified number of sample
# episodes and averages returns to find the optimal policy.
# Can choose first visit MC or all visits MC with Boolean.
# This function also calls generate_episode() of previous sec.

def MC_Ctrl_EpsGreedy(Pi, eps, env, gamma, N_samples, T_max, all_visits_MC):
    '''
     Epsilon-greedy Monte Carlo control algorithm.
     
     ARGUMENTS: - Pi, env, gamma: Policy, environment, discount factor.
                - eps: Epsilon float for output policy.
                - N_samples: No. of samples for Monte Carlo expectation.
                - T_max: Max. episode length minus 1.
                - all_visits_MC: Boolean for all visits or first visit MC.
     OUTPUT:    - Pi_star: Optimal policy
                - V_star: Optimal value function
                - Q: State-action values from samples
                - Returns: Dict. of return samples (by init. state-action)
                - N_iter: No. of randomly generated episodes (<= N_samples)
     NOTE: This function calls generate_episode().
           Should take an eps-soft policy as input.
    '''
    
    # Environment attributes
    non_term_states = env.non_term_states
    term_states = env.term_states
    adm_actions = env.adm_actions
    Rwds = env.rewards
    
    
    # Init. Q, returns, Pi_star, and V_star dictionaries
    Pi_star = {}
    Q = {}
    Returns = {}
    V_star = {}
    for s in non_term_states:
        Pi_star[s] = {}
        Q[s]={}
        Returns[s] = {}
        V_star[s] = 0.0
        for a in adm_actions[s]:
            Q[s][a] = 0.0
            Returns[s][a]=[]
            
    for s in term_states:
        V_star[s] = 0.0
            
    
    # Init. counter
    N_iter = 0
    
    # Main MC loop
    while (N_iter<N_samples):
                
        ##########################
        ##### Step t=0 setup #####
        ##########################

        # Generate (s_0, a_0) randomly from non_term_states and adm_actions
        # np.random.choice() works only for 1-dim'l arrays
        s_0 = list(non_term_states)[np.random.randint(len(non_term_states))]
        a_0 = np.random.choice(adm_actions[s_0])

        ########################
        ##### Steps 1 to T #####
        ########################
        
        #print(f"Generating sample episode no. {N_iter+1}...\n") # Debug
        # Generate episode
        # Signature: episode_rewards, episode_states, episode_actions, T = generate_episode(s_0, a_0, Pi, env, T_max)
        # Step t of episode is (r_t, s_t, a_t) = (episode_rewards[t], episode_states[t], episode_actions[t])
        episode_rewards, episode_states, episode_actions, T = generate_episode_stochastic(s_0, a_0, Pi, grid, T_max)
        #print(f"Sample episode N_iter={N_iter} has T={T} steps after t=0.\n") # Debug
 
        #####################################################
        ### COMPUTE CUMULATIVE RETURNS AND OPTIMAL ACTION ###
        #####################################################

        # First visit only MC
        if not all_visits_MC:
            # State-action iterable
            episode_state_actions = list(zip(episode_states, episode_actions))
            
            # Init. storing variable
            G = 0.0
            
            # Loop over episode
            for t in range(T-1, -1, -1): # Loop goes backwards from (T-1) to 0
                
                # Sum the return
                G = gamma*G + episode_rewards[t+1]
                s_t = episode_states[t]
                a_t = episode_actions[t]
                
                if (s_t, a_t) not in episode_state_actions[:t]:
                    
                    # Update sample returns and Q-function
                    Returns[s_t][a_t].append(G)
                    Q[s_t][a_t] = np.average(Returns[s_t][a_t])
                    
                    # Get a_star and update Pi_star
                    a_star = max(Q[s_t], key = Q[s_t].get)
                    Pi_star[s_t][a_star] = 1-eps+eps/len(adm_actions[s_t])
                    for a in adm_actions[s_t]:
                        if a != a_star:
                            Pi_star[s_t][a] = eps/len(adm_actions[s_t])
                
                    
        # END IF first visit MC
        
        # All visits MC
        elif all_visits_MC:
            
            # Init. storing variable
            G = 0.0
            
            for t in range(T-1, -1, -1): # Loop goes backwards from (T-1) to 0
                
                # Sum the returns
                G = gamma*G + episode_rewards[t+1]
                s_t = episode_states[t]
                a_t = episode_actions[t]
                
                # Update sample returns and Q-function
                Returns[s_t][a_t].append(G)
                Q[s_t][a_t] = np.average(Returns[s_t][a_t])
                
                # Get a_star and update Pi_star
                a_star = max(Q[s_t], key = Q[s_t].get)
                Pi_star[s_t][a_star] = 1-eps+eps/len(adm_actions[s_t])
                for a in adm_actions[s_t]:
                    if a != a_star:
                        Pi_star[s_t][a] = eps/len(adm_actions[s_t])
                
                              
        # Update N_iter
        N_iter += 1
        
        '''
        # DEBUG:
        print(f"Completed episode N_iter={N_iter} with R={G}.")
        print("--------------------------------------------------------------------")
        '''
       
    # END WHILE of MC loop
    
    # COMPUTE V
    ## CHANGE IN EPSILON GREEDY??
    for s in non_term_states:
        a = max(Q[s], key = Q[s].get)
        V_star[s]=Q[s][a]
    
    # Output
    return Pi_star, V_star, Q, Returns, N_iter

# END DEF MC_Ctrl_EpsGreedy
