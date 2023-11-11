##############################################################
##### Windy GridWorld - Reinforcement learning functions #####
##############################################################
## 2022/04/29, AJ Zerouali
## Functions used with Windy GridWorld to 
## evaluate RL algorithms.

from __future__ import print_function, division
from builtins import range

import numpy as np

# Import the windy GridWorld class
from Windy_GridWorld import GridWorld_Windy_small

##############################
##### PRINTING FUNCTIONS #####
##############################
# 2022/04/06, AJ Zerouali
# Modified from Lazy Prog's GitHub

def print_values(Val_fn, env):
    '''
    Prints value function
    ARGUMENTS: Val_fn: Dictionary of values for each non 
                terminal state in env.
               env: GridWorld_Windy_small object
    '''
    print(f"## VALUE FUNCTION ##")
    for i in range(env.rows):
        print("------------------------")
        for j in range(env.cols):
            v = Val_fn.get((i,j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="") # -ve sign takes up an extra space
        print("")
    print("------------------------")
# END DEF print_values()

def print_policy(Pi_fn, env):
    '''
    Prints deterministic policy
    ARGUMENTS: Pi_fn: Deterministic policy of the form:
                pi = {
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
               env: GridWorld_Windy_small object
    Note: Keeping such policy format for stochastic case.
    '''
    # REMARK: WILL ONLY PRINT A DETERMINISTIC POLICY WITH {(i,j):{"action":1.0}}
    print(f"##  POLICY  ##")
    for i in range(env.rows):
        print("------------------------")
        for j in range(env.cols):
            if (i,j) not in [(1,1), (0,3), (1,3)]:
                # WARNING: Will only work if there's one and only one element
                a = list(Pi_fn[(i,j)].keys())[0]
                print("  %s  |" % a, end="")
            elif (i,j) == (1,1):
                print("  %s  |" % " ", end="")
        print("")
    print("------------------------")
 # END DEF print_policy()   
 
################################################# 
##### RANDOM DETERMINISTIC POLICY GENERATOR #####
#################################################
## 2022/04/08, AJ Zerouali
# Recall: rand_ind = np.random.choice(a = len(next_states), p = next_states_probs)

def gen_random_policy(env):
    '''
      Generates a random deterministic policy given an environment.
      ARGUMENTS: env, Windy_GridWorld_simple object (environment).
      OUTPUT: Pi, a (deterministic) policy dictionary.
    '''
    non_term_states = env.non_term_states
    adm_actions = env.adm_actions
    Pi = {}
    
    for s in non_term_states:
        actions_list = list(adm_actions[s])
        a_random = actions_list[np.random.randint(len(actions_list))]
        Pi[s] = {a_random:1.0}
    
    return Pi
# END DEF gen_random_policy()  

################################################
##### RANDOM EPSILON-SOFT POLICY GENERATOR #####
################################################
## 2022/05/03, AJ Zerouali
# Recall: rand_ind = np.random.choice(a = len(next_states), p = next_states_probs)

def gen_random_epslnsoft_policy(eps, env):
    '''
      Generates a random epsilon-soft policy for Windy GridWorld.
      ARGUMENTS: - eps, the epsilon float;
                 - env, Windy_GridWorld_simple object (environment).
      OUTPUT: Pi, an epsilon-soft policy dictionary.
      Note: eps should be between 5% and 10%.
    '''
    non_term_states = env.non_term_states
    adm_actions = env.adm_actions
    Pi = {}
    
    for s in non_term_states:
        Pi[s]={}
        actions_list_s = list(adm_actions[s])
        a_rand = np.random.choice(actions_list_s)
        for a in actions_list_s:
            if a==a_rand:
                Pi[s][a] = 1-eps+(eps/len(actions_list_s))
            else:
                Pi[s][a] = eps/len(actions_list_s)
    
    return Pi

##################################
##### COMPARE VALUE FUNCTION #####
##################################
## 2022/04/22, AJ Zerouali

def compare_value_fns(V_old, V_new, non_term_states):
    '''
     Compares two value functions.
     ARGUMENTS: - V_old and V_new: Dictionaries of 2 value functions to compare
                - non_term_states: Set of non-terminal states in the environment
     OUTPUT: delta_V = sup_{s in S} |V_old(s)- V_new(s)|
    '''
    delta_V = 0
    for s in non_term_states:
        delta_V = max(delta_V, abs(V_old[s]-V_new[s]))
        
    return delta_V
# END DEF compare_value_fns()  
    