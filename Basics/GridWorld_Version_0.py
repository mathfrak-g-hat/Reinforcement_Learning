from __future__ import printing_function, division
from builtins import range

import numpy as np
#from Windy_GridWorld import GridWorld_Windy_small



################################
## VALUE ITERATION ALGORITHM  ##
################################
## 2022/04/24, AJ Zerouali
# This is Bellman's famous algorithm of 1957.
# REMARK: Check the break condition is correct.

def Value_Iteration(env, epsilon, gamma):
    # ARGUMENTS: - Pi: Necessary?
    #            - env: Environment. Gives the state and action spaces.
    #            - epsilon: Convergence threshold 
    #            - gamma: Discount factor
    # OUTPUT:    - V_star: Optimal value function
    #            - Pi_star: Optimal policy
    #            - N_iter: No. of iterations 
    # NOTE: Pi_star is obtained from the actions that gave the last update of V_new = V_star
    
    # Initialize env. attributes
    term_states = env.term_states
    non_term_states = env.non_term_states
    P_trans = env.transition_probs
    Rwds = env.rewards
    adm_actions = env.adm_actions
    
    # Initialize V and Q to zero
    V_new = {}
    Q = {}
    for s in term_states:
        V_new[s] = 0.0
    for s in non_term_states:
        V_new[s] = 0.0
        Q[s] = {}
        for a in adm_actions[s]:
            Q[s][a] = 0.0
            
    # DEBUG: Initialize Pi_star
    Pi_star = {}
    
    # Init. iteration counter
    N_iter = 0
    
    ## MAIN LOOP
    while True:
        
        V_old = V_new
        
        Delta_V = 0.0
        
        for s in non_term_states:
            
            # Store V_old(s)
            Vs_old = V_old[s]
            
            # This loop computes V(s)
            for a in adm_actions[s]:
                
                # Init. Q_sa
                Q_sa = 0.0
                
                # This loop computes Q(s,a)
                # Loop only over non-zero probability transitions
                for s_ind in P_trans[(s,a)].keys():
                    # Bellman update
                    # Template: V_temp += P_trans[(s,a)].get(s_ind,0)*( Rwds.get(s_ind,0) + gamma*V_pi[s_ind] )
                    Q_sa += P_trans[(s,a)].get(s_ind,0)*( Rwds.get(s_ind,0) + gamma*V_old[s_ind] )
                
                # Update Q(s,a)
                Q[s][a] = Q_sa
                
            # END FOR a in adm_actions[s]
            
            # Get max over a's
            V_new[s] = max(Q[s].values())
            
            # Pi_star debug:
            # Store argmax
            a_star = max(Q[s], key = Q[s].get)
            Pi_star[s] = {a_star:1.0}
            
            # Update Delta_V
            Delta_V = max(Delta_V, abs(Vs_old - V_new[s]))
            
        # END FOR s in non_term_states
        
        if Delta_V < epsilon:
            break
        
        # Update iteration counter
        N_iter += 1
        
    # END WHILE
    
    # Return optimal value fn and no. of iterations
    return V_new, N_iter, Pi_star

# END DEF Value_Iteration()


################################
##      FIND OPTIMAL POLICY   ##
################################
## This function simply extracts argmaxes.
## Should necessarily be executed after value iteration.

def Get_Pi_Star(V_star, env, epsilon, gamma):
    # ARGUMENTS: - V_star: Optimal value function
    #            - env: Environment. Gives the state and action spaces.
    #            - epsilon: Convergence threshold 
    #            - gamma: Discount factor
    # OUTPUT:    - Pi := Pi_star, optimal policy
    
    # Init. env. attributes
    P_trans = env.transition_probs
    Rwds = env.rewards
    non_term_states = env.non_term_states
    term_states = env.term_states
    adm_actions = env.adm_actions

    # Init. Q and Pi (output)
    Pi = {}
    Q = {}
    for s in non_term_states:
        Pi[s] = {}
        Q[s] = {}
        for a in adm_actions[s]:
            Q[s][a] = 0.0
    
    # Here you should compute Q(s,a) then extract argmax
    # Recall argmax for dictionary given by
    # a_new = max(Vs_dict, key = Vs_dict.get)
    for s in non_term_states:
        
        for a in adm_actions[s]:
            
            Q_sa = 0.0
            
            # Loop over non-zero transitions
            for s_ind in P_trans[(s,a)].keys():
                
                # Bellman equation
                Q_sa += P_trans[(s,a)].get(s_ind,0)*(Rwds.get(s_ind, 0)+gamma*V_star[s_ind])
                
            # END FOR s_ind in admissible
            
            # Store above sum
            Q[s][a] = Q_sa
            
        # END FOR a in adm_actions[s]
        
        # Get argmax and store in Pi
        a_star = max(Q[s], key = Q[s].get)
        Pi[s] = {a_star:1.0}
        #Pi[s] = {max(Q[s], key = Q[s].get):1.0}
        
    # END FOR s in non_term_states    
    
    # Return optimal policy
    return Pi

# END DEF Value_Iteration()


##### PRINTING FUNCTIONS #####
# 2022/04/06, AJ Zerouali
# Modified from Lazy Prog's GitHub

def print_values(Val_fn, env):
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
        
def print_policy(Pi_fn, env):
    print(f"##  POLICY  ##")
    for i in range(env.rows):
        print("------------------------")
        for j in range(env.cols):
            a = Pi_fn.get((i,j), ' ')
            print("  %s  |" % a, end="")
        print("")
    print("------------------------")
    
    
    
##### ITERATIVE POLICY EVALUATION #####
## 2022/04/06, AJ Zerouali
# My first implementation. A simplified version of Lazy Programmer's lecture 51. The policy is deterministic.


def iter_policy_eval(policy, V_ini, transition_probs, rewards, epsilon, gamma):
    # ARGUMENTS:
    #  policy: Dict. Policy function to be evaluated
    #  V_ini: Dict. Initial value fn
    #  transition_probs: Dict. Transition probabilities of MDP
    #  rewards: Dict. Rewards by (state, action, state_new)
    #  epsilon: Float. Convergence threshold (for sup norm of value function)
    #  gamma: Float. Discount factor
    
    # COMMENTS:
    # 1) YOU ALSO NEED THE STATES FOR YOUR LOOPS BELOW
    # 2) USE THE ENVIRONMENT TO BUILD MDP OBJECTS?
    
    # OUTPUT:
    #  V_pi: Dict. Value function corresp. to Pi_fn
    #  k: Number of iterations for convergence of policy eval.
    
    
    # INITIALIZATIONS
    # Transitions, rewards, and policy ini (Renaming for readability. Bad idea?)
    T = transition_probs
    R = rewards
    Pi = policy
    # Ini. set of states from V_ini keys and add terminal states
    Non_Terminal_States = set(Pi.keys())
    Terminal_States = {(0,3), (1,3)}
    # V_k and V_(k+1) ini. (get switched in while loop)
    V_new = V_ini
    for s in Terminal_States:
        V_new[s] = 0
    V_old = {}
    # Iteration counter ini
    k = 0
    # Stopping Boolean ini
    V_is_stable = False
    
    
    # MAIN LOOP
    # Iterates over k
    while not V_is_stable:
        
        # Initialize V_k and V_(k+1)
        V_old = V_new
        V_new = {}
        for s in Terminal_States:
            V_new[s] = 0
        # Initialize sup|V_(k+1) - V_k|
        Delta_V = 0
        
        # EVALUATE V_(k+1)=V_new
        ##### SHOULD I ONLY LOOP OVER NON TERMINAL STATES? V_new[s_terminal] = 0 too after all #####
        # Loop over non terminal states
        for s in Non_Terminal_States: 
            
            # COMPUTE V_(k+1)(s)
            V_s_new = 0
            for s_ind in (Non_Terminal_States | Terminal_States): 
                # UPDATE V_s_new (CORE OF ITERATIVE POLICY EVALUATION)
                V_s_new += T.get((s,Pi[s],s_ind),0)*( R.get((s,Pi[s],s_ind),0) + gamma*V_old[s_ind] )
                ##### BE CAREFUL WITH TERMINAL STATES #####
            # END FOR
            # Assign V_(k+1)(s)
            V_new[s] = V_s_new
            
            # Update sup|V_(k+1) - V_k|
            Delta_V = max(Delta_V, abs(V_s_new-V_old.get(s,0)) )
            
        # END FOR     
        
        # Update stopping Boolean
        V_is_stable = (Delta_V < epsilon)
        
        # Update iteration counter
        k += 1
    # END WHILE
    
    # Return V_pi and number of iterations
    return V_new, k


'''
Below is a trivial case. Follows lectures 50 and 51 of Lazy Programmer's 
Intro to RL

##### MAIN #####
# 2022/04/05, AJ Zerouali
# Starts at 6:30 of Lazy Prog's Lecture 51
#

# Initialize transition probabilities and rewards
transition_probs = {}
rewards = {}

### Populate the transition probabilities and rewards ###
# Create environment
grid = standard_grid()
# Loop through all states:
for i in range(grid.rows):
    for j in range(grid.cols):
        s = (i,j)
        # Execute when not terminal state
        if not grid.is_terminal(s):
            # Loop over action space:
            for a in ACTION_SPACE:
                # Compute next state
                s_new = grid.get_next_state(s,a)
                transition_probs[(s, a, s_new)]=1
                # Execute when state has a reward
                # NOTE: Not strictly necessary, but useful in general
                if s_new in grid.rewards:
                    rewards[(s, a, s_new)] = grid.rewards[s_new]

### The policy dictionary ###
# It's deterministic
policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'U',
    (2, 1): 'R',
    (2, 2): 'U',
    (2, 3): 'L',
}

### Initial value function ###
# Just a dictionary of 0s
V = {}
for s in grid.all_states():
    V[s] = 0
'''

'''
When executing the commented code above, printing the policy gives:

##  POLICY  ##
------------------------
  R  |  R  |  R  |     |
------------------------
  U  |     |  U  |     |
------------------------
  U  |  R  |  U  |  L  |
------------------------

Printing the value function gives:

## VALUE FUNCTION ##
------------------------
 0.81| 0.90| 1.00| 0.00|
------------------------
 0.73| 0.00| 0.90| 0.00|
------------------------
 0.66| 0.73| 0.81| 0.73|
------------------------

'''
