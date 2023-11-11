from __future__ import print_function, division
from builtins import range

import numpy as np
from Windy_GridWorld import GridWorld_Windy_small

#######################################
##### ITERATIVE POLICY EVALUATION #####
#######################################
## 2022/04/08, AJ Zerouali

def iter_policy_eval(Pi, V_ini, P_trans, Rwds, adm_actions, non_term_states, term_states, epsilon, gamma):
    # ARGUMENTS:
    #  Pi: Dict. Policy function to be evaluated, from main() function.
    #  V_ini: Dict. Initial value fn, from main() function.
    #  P_trans: Dict. Transition probabilities of MDP, from main() function.
    #  Rwds: Dict. Rewards by (state, action, state_new), from main() function.
    #  adm_actions: Dict. Admissible actions in a given state, from grid attributes.
    #  non_term_states: Set. Non terminal states, from grid attributes.
    #  term_states: Set. Terminal states only, from grid attributes.
    #  epsilon: Float. Convergence threshold (for sup norm of value function), from main() function.
    #  gamma: Float. Discount factor, from main() function.
    
    # OUTPUT:
    #  V_pi: Dict. Value function corresp. to Pi
    #  k: Number of iterations for convergence of policy eval.
    
    
    # INITIALIZATIONS
    # V_k and V_(k+1) ini. (get switched in while loop)
    V_new = V_ini
    for s in term_states:
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
        for s in term_states:
            V_new[s] = 0
        # Initialize sup|V_(k+1) - V_k|
        Delta_V = 0
        
        # EVALUATE V_(k+1)=V_new
        # Loop over non terminal states
        for s in non_term_states:  
            
            # COMPUTE V_(k+1)(s)
            
            # Initialize
            V_s_new = 0
            
            # Loop over admissible actions in state s
            for a in adm_actions[s]:
                
                # Add sum over s_ind only if pi(a|s) is non-zero:
                if (Pi[s].get(a,0) != 0):
                
                    # This loop is only over non-trivial transitions
                    for s_ind in P_trans[(s,a)].keys(): 
                        # UPDATE V_s_new
                        V_s_new += Pi[s].get(a,0)*P_trans[(s,a)].get(s_ind,0) \
                                    *( Rwds.get(s_ind,0) + gamma*V_old[s_ind] )  
                    # END FOR OVER s_ind
                    
                # END IF
                
            # END FOR OVER a
            
            # Assign V_(k+1)(s)
            V_new[s] = V_s_new
            
            # Update sup|V_(k+1) - V_k|
            Delta_V = max(Delta_V, abs(V_s_new-V_old.get(s,0)) )
            
        # END FOR OVER s     
        
        # Update stopping Boolean
        V_is_stable = (Delta_V < epsilon)
        
        # Update iteration counter
        k += 1
    # END WHILE
    
    # Return V_pi and number of iterations
    return V_new, k
# END DEF iter_policy_eval()

################################
## POLICY ITERATION ALGORITHM ##
################################
# 2022/04/22, AJ Zerouali


def Policy_Iteration(Pi, V_ini, P_trans, Rwds, adm_actions, non_term_states, term_states, epsilon, gamma):
    
    

    # Initialize counter and looping Boolean
    N_iter = 0
    policy_is_stable = True #Necessary?
    
    # Init. V_old
    V_old = V_ini

    # Loop until policy_is_stable = True
    while True:
        #######################
        ## POLICY EVALUATION ##
        #######################

        # Execute policy eval function
        V_new, k = iter_policy_eval(Pi, V_old, P_trans, Rwds, adm_actions, non_term_states, term_states, epsilon, gamma)

        # DEBUG:
        print(f"Policy evaluation fn iter_policy_eval() converged after {k} iterations.")

        ###########################################
        ## POLICY IMPROVEMENT - improve_policy() ##
        ###########################################

        Pi, policy_is_stable = improve_policy(Pi, V_new, P_trans, Rwds, adm_actions, non_term_states, term_states, gamma)

        # Break condition (Tricky)####
        # Update policy iteration counter
        N_iter += 1

        # Compare value functions:
        delta_V = compare_value_fns(V_old, V_new, non_term_states)
        # Update value function
        V_old = V_new

        # BREAK WHILE condition
        #if policy_is_stable or N_iter>30:
        #    break
        if policy_is_stable:
            break
        elif delta_V<=epsilon:
            break

    # END WHILE not policy_is_stable

    # DEBUG/REMINDER: In function, should finish with
    return V_new, Pi, N_iter

# END DEF Policy_Iteration()

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