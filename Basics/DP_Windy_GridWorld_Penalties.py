from __future__ import print_function, division
from builtins import range

import numpy as np
# Import environment class and helper function for non-windy test with no penalties
from Windy_GridWorld import GridWorld_Windy_small, windy_standard_grid
# Get RL helper functions
from RL_Fns_Windy_GridWorld import print_values, print_policy
# Get Dp algorithms
from Dynamic_Programming_Windy_GridWorld import iter_policy_eval, Policy_Iteration, Value_Iteration, Get_Pi_Star

######################################################################
### Non-windy GridWorld with value iteration and various penalties ###
######################################################################
## 2022/04/24, AJ Zerouali
# Based on Lazy Programmer's intro to RL course.

if __name__ == '__main__':

    ### Windy GridWorld with various penalties
    ## This time using Value Iteration Algorithm instead of Policy Iteration
    # 2022/04/24

    # Discount factor and error threshold
    gamma = 0.9
    epsilon = 1e-3

    # Penalty list
    penalties = [0.0, -0.1, -0.2, -0.4, -0.5, -2]

    # Loop over penalties
    for pen in penalties:
        
        # Create environment
        grid = windy_standard_grid(penalty=pen)
        print(f"Windy GridWorld environment with penalty = {pen} created ... \n")

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
        
        # Separator
        print("_____________________________________________\n\n")

# END MAIN