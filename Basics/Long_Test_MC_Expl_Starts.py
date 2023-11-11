from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from Windy_GridWorld import GridWorld_Windy_small, windy_standard_grid, test_standard_grid
from RL_Fns_Windy_GridWorld import print_values, print_policy, gen_random_policy, compare_value_fns
from Monte_Carlo_Windy_GridWorld import generate_episode, MC_Policy_Eval

#########################################################################
###  MONTE CARLO CONTROL WITH EXPLORING STARTS - VERSION 2 WITH DEBUG ###
#########################################################################
## 2022/04/28, AJ Zerouali
# Monte Carlo with exploring starts (Lect. 64-65)
# We are not using convergence thresholds here.
# This function below generates a specified number of sample
# episodes and averages returns to find the optimal policy.
# Can choose first visit MC or all visits MC with Boolean.
# This function also calls generate_episode() of previous sec.

def MC_Ctrl_ExpStarts(Pi, env, gamma, N_samples, T_max, all_visits_MC):
    # ARGUMENTS: - Pi, env, gamma: Policy, environment, discount factor.
    #            - N_samples: No. of samples for Monte Carlo expectation.
    #            - T_max: Max. episode length minus 1.
    #            - all_visits_MC: Boolean for all visits or first visit MC.
    # OUTPUT:    - Pi_star: Optimal policy
    #            - V_star: Optimal value function
    # NOTE: This function calls generate_episode().
    
    # Environment attributes
    non_term_states = env.non_term_states
    term_states = env.term_states
    adm_actions = env.adm_actions
    Rwds = env.rewards
    
    
    # Init. Q, returns, Pi_star, and V_star dictionaries
    Pi_star = {}
    Q = {}
    alt_Q = {} # DEBUG
    Qsa_samples = {}
    alt_V = {} # DEBUG
    Returns = {}
    V_star = {}
    for s in non_term_states:
        Pi_star[s] = {}
        Q[s]={}
        alt_Q[s] = {} # DEBUG
        Qsa_samples[s] = {} # DEBUG
        alt_V[s] = 0.0 # DEBUG
        Returns[s] = {}
        V_star[s] = 0.0
        for a in adm_actions[s]:
            alt_Q[s][a] = 0.0 # DEBUG
            Qsa_samples[s][a] = 0 # DEBUG
            Q[s][a] = 0.0
            Returns[s][a]=[]
            
    for s in term_states:
        V_star[s] = 0.0
        alt_V[s] = 0.0 # DEBUG
            
    
    # Init. counter and delta_V
    N_iter = 0
    delta_V = float('inf')
    deltas = []
    
    # Main MC loop
    while True:
                       
        # Store old V
        #V_old = V_star
        
        # DEBUG: Init. delta_Q
        delta_Q = 0.0
        
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
        
        print(f"Generating sample episode N_iter={N_iter}...\n")
        # Generate episode
        # Signature: episode_rewards, episode_states, episode_actions, T = generate_episode(s_0, a_0, Pi, env, T_max)
        episode_rewards, episode_states, episode_actions, T = generate_episode(s_0, a_0, Pi, grid, T_max)
        # Step t of episode is (r_t, s_t, a_t) = (episode_rewards[t], episode_states[t], episode_actions[t])
        print(f"Sample episode N_iter={N_iter} has T={T} steps after t=0.\n")

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
                
                print(f"At step t={t}, have (Rwds[s_t]==episode_rewards[t+1])={Rwds.get(s_t,0)==episode_rewards[t+1]}")
                if Rwds.get(s_t,0)!=episode_rewards[t+1]:
                    print(f"  WARNING: Rwds[s_t] = {Rwds.get(s_t,0)}")
                    print(f"           episode_rewards[t+1]={episode_rewards[t+1]}")
                else:
                    print(f"           episode_rewards[t+1]=Rwds[s_t] ={episode_rewards[t+1]}")
                
                if (s_t, a_t) not in episode_state_actions[:t]:
                    
                    old_Q = alt_Q[s_t][a_t] # DEBUG
                    Qsa_samples[s_t][a_t] += 1 # DEBUG
                    learn_rate = 1/Qsa_samples[s_t][a_t] # DEBUG
                    alt_Q[s_t][a_t] = old_Q + learn_rate*(G-old_Q) # DEBUG
                    
                    
                    # Update sample returns and Q-function
                    # WARNING: THERE SEEMS TO BE A PROBLEM HERE
                    Returns[s_t][a_t].append(G)
                    Q[s_t][a_t] = np.average(Returns[s_t][a_t])
                    
                    # Get a_star and update Pi_star
                    a_star = max(Q[s_t], key = Q[s_t].get)
                    Pi_star[s_t] = {a_star:1.0} 
                    
                    # Update V_star
                    V_star[s_t] = Q[s_t][a_star] # SHOULD THIS LINE EVEN BE HERE?
                    
                    # DEBUG: Get delta_Q
                    delta_Q = max(delta_Q, np.abs(old_Q-alt_Q[s_t][a_t])) # DEBUG
                    
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
                
                # Update V_star
                V_star[s_t] = Q[s_t][a_star]
                
          
        # DEBUG: Append delta_Q
        deltas.append(delta_Q)
        
        # Compute delta_V
        #for s in non_term_states:
        #    if (V_star.get(s,0) != 0) and (V_old.get(s,0) != 0):
        #        delta_V = max(delta_V, np.abs(V_old[s]-V_star[s]))
        # Append to deltas
        #deltas.append(delta_V)
        
        # Update N_iter
        N_iter += 1
        
        # DEBUG:
        print(f"Completed episode N_iter={N_iter} with delta_V={delta_V} and R={G}.")
        print("--------------------------------------------------------------------")
        
        if (N_iter >= N_samples):
            break
        

    # END WHILE of MC loop
    
    # DEBUG: COMPUTE alt_V
    for s in non_term_states:
        a = max(alt_Q[s], key = alt_Q[s].get)# DEBUG
        alt_V[s] = alt_Q[s][a] # DEBUG
    
    # Output
    return Pi_star, V_star, Q, Returns, N_iter, deltas, alt_V, alt_Q

# END DEF MC_Ctrl_ExpStarts


if __name__ == '__main__':

    # Create environment
    grid = test_standard_grid()

    # Policy to be evaluated
    Pi_opt = {
        (2, 0): {'U': 1.0},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'U': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'U': 1.0},
        (2, 3): {'L': 1.0},
      }

    Pi_lect = {
        (2, 0): {'U': 1.0},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'R': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'R': 1.0},
        (2, 3): {'U': 1.0},
      }

    Pi_rand = gen_random_policy(grid)

    # Discount factor and convergence threshold
    gamma = 0.9
    #epsilon = 1e-3

    # Max. episode length:
    T_max = 50
    N_samples = 30000

    # Evaluate V_Pi:
    # SIGNATURE: Pi_star, V_star, Q, Returns = MC_Ctrl_ExpStarts(Pi, env, gamma, N_samples, T_max, all_visits_MC)
    #V = MC_Policy_Eval(Pi, grid, gamma, N_samples, T_max, all_visits_MC = False)
    Pi_star, V_star, Q, Returns, N_iter, deltas, alt_V, alt_Q = MC_Ctrl_ExpStarts(Pi_rand, grid, gamma, N_samples, T_max, all_visits_MC = False)

    print(f"Printing the optimal policy obtained from MC_Ctrl_ExpStarts:")
    print_policy(Pi_star, grid)

    print(f"Printing the value fn V_star obtained from MC_Ctrl_ExpStarts:")
    print_values(V_star, grid)
    
    print(f"Printing the value fn V_star obtained from alternative MC_Ctrl_ExpStarts:")
    print_values(alt_V, grid)

    V_eval = MC_Policy_Eval(Pi_star, grid, gamma, 50, T_max, all_visits_MC = False)
    print(f"Printing the value fn obtained from MC_Policy_Eval(Pi_star,...):")
    print_values(V_eval, grid)