from __future__ import print_function, division
from builtins import range

import numpy as np
# Import environment class and helper function for non-windy test with no penalties
from Windy_GridWorld import GridWorld_Windy_small, test_standard_grid
# Get RL helper functions
from RL_Fns_Windy_GridWorld import print_values, print_policy
# Get Dp algorithms
from Dynamic_Programming_Windy_GridWorld import iter_policy_eval, Policy_Iteration, Value_Iteration, Get_Pi_Star

################################################
### Non-windy GridWorld with value iteration ###
################################################
## 2022/04/23, AJ Zerouali
# Based on Lazy Programmer's intro to RL course.

if __name__ == '__main__':
    # Create environment
    # adm_actions, rewards and transition_probs are attributes of grid

    grid = test_standard_grid()

    print(f"Test with non-windy GridWorld environment ... \n")

    # Discount factor and error threshold
    gamma = 0.9
    epsilon = 1e-3

    ##############################
    ## EXECUTE VALUE ITERATION  ##
    ##############################

    print(f"Executing value iteration algorithm ...")
    # SIGNATURE: V_star, N_iter, Pi_star = Value_Iteration(env, epsilon, gamma)
    (V_star, N_iter, Pi_star) = Value_Iteration(grid, epsilon, gamma)

    ##############################
    ##  GET OPTIMAL POLICY      ##
    ##############################

    # SIGNATURE: Pi_star = Get_Pi_Star(V_star, env, epsilon, gamma)
    Pi_computed = Get_Pi_Star(V_star, grid, epsilon, gamma)


    ###################
    ## PRINT RESULTS ##
    ###################

    # Print N_iter
    # Print optimal value function
    print(f"Value_Iteration() converged after {N_iter} iterations ...\n")

    # Print optimal value function
    print(f"Printing optimal value function ...")
    print_values(V_star, grid)

    # Print optimal (deterministic) policy
    print(f"Printing policy obtained from Value_Iteration()...")
    print_policy(Pi_star, grid)

    # Print optimal (deterministic) policy
    print(f"Printing policy obtained from Get_Pi_star()...")
    print_policy(Pi_computed, grid)

# END MAIN