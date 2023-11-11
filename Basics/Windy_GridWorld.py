##### WINDY GRIDWORLD #####
# Updated: 22/04/29, A. J. Zerouali

# 
from __future__ import print_function, division
from builtins import range

import numpy as np


# GridWorld_simple with only 3x4 grid. This is the environment.
class GridWorld_Windy_small():
    '''
    Updated 2022/04/07, AJ Zerouali
    The Windy GridWorld environment used in Lazy Programmer's course.
    This is a 3x4 grid, with wall at (1,1), +1 reward at the terminal 
    square (0,3), and -1 reward at the terminal square (1,3).
    For the "windy" variant, the main changes occur in the move() method.
    States are (i,j) tuples, actions are characters, containers are dictionaries.
    '''

    def __init__(self, rows, cols, ini_state, non_term_states, term_states, actions):
        # Attributes rows and cols are dimensions of the grid
        self.rows = rows
        self.cols = cols
        # Coordinates of agent
        self.i = ini_state[0]
        self.j = ini_state[1]
        # State and action spaces
        self.non_term_states = non_term_states
        self.term_states = term_states
        self.actions = actions 
        # The next attributes are populated using the set() method
        self.adm_actions = {}
        self.rewards = {}
        self.transition_probs = {}
        
    # Method setting up the actions, rewards, and transition probabilities
    def set(self, rewards, adm_actions, transition_probs):
        # INPUT: adm_actions: Dictionary of (i,j):[a_i] = (row,col):[action list]
        #        rewards: Dictionary of (i,j):r = (row,col):reward
        #        transition_probs: Dictionary of (i,j):{a_i:p_ij}= ...
        #                          .. (row,col):{dictionary of probs for each action}
        # WARNING: Do not confuse self.adm_actions with self.actions. Latter is the action space,
        #          adm_actions are the accessible actions from a state (dict. {s_i:[a_ij]}).
        self.rewards = rewards
        self.adm_actions = adm_actions
        self.transition_probs = transition_probs
    
    # Method that sets current state of agent
    def set_state(self, s):
        # INPUT: s: (i,j)=(row,col), coord. of agent
        self.i = s[0]
        self.j = s[1]

    # Method to return current state of agent
    def current_state(self):
        return (self.i, self.j)
    
    # Method to check if current agent state is terminal
    # Note: Lazy Prog not explciting terminal states
    def is_terminal(self, s):
        return (s in self.term_states)
    
    # HAS TO BE MODIFIED FOR WINDY GRIDWORLD
    # Method to perform action in environment
    def move(self, action):
        # Input:  action: New action to execute
        # Output: reward
        # Comments: - Requires transition probabilities. 
        #           - Calls numpy.random.choice(), doesn't work with dictionaries.
        
        # Check if action is admissible in current state
        if action in adm_actions[self.current_state()]:
            
            # Convert transition_probs to lists compatible with np.random.choice().
            # Recall self.transition_probs[(self.current_state(), action)] is a dictionary,
            # while np.random.choice() works with ints or ndarrays.
            next_states = list(self.transition_probs[(self.current_state(), action)].keys())
            next_states_probs = list(self.transition_probs[(self.current_state(), action)].values())
            
            # Generate a random index (this Numpy function is tricky)
            rand_ind = np.random.choice(a = len(next_states), p = next_states_probs)
            # Set new state of agent
            s_new = next_states[rand_ind] # Not necessary, for debug
            self.set_state(s_new)
        # END IF
   
        # Return reward. If not in given dictionary, return 0
        return self.rewards.get((self.i, self.j), 0)

    # Method to check if agent is currently in terminal state
    def game_over(self):
        # Output true if agent is in terminal states (0,3) or (1,3)
        return ( (self.i, self.j) in self.term_states)
    
    # Method returnning all admissible states, i.e. not in the wall (1,1)
    def all_states(self):
        return (self.non_term_states | self.term_states )
# END CLASS

    
# Helper function to construct a windy environment.
# Consists mainly of initializations.
def windy_standard_grid(penalty=0):

    '''
    Helper function to construct a windy environment.
    Consists mainly of initializations.
    Input: penalty: Float. Penalty for moving to non terminal state.
    Output: env. Windy_GridWorld_small() object (the environment).
    '''
    
    # Start at bottom left (randomize later)
    ini_state = (2,0)
    # Action space 
    ACTION_SPACE = {"U", "D", "L", "R"}
    # Non terminal states
    NON_TERMINAL_STATES = {(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)}
    # Terminal states
    TERMINAL_STATES = {(0,3), (1,3)}
    
    # Instantiate:
    env = GridWorld_Windy_small(3, 4, ini_state, NON_TERMINAL_STATES, TERMINAL_STATES, ACTION_SPACE)

    
    # Dictionary of rewards
    # Not storing 0s if penalty=0
    rewards = {(0,3):1, (1,3): -1}
    # Poplate non terminal states for penalty != 0
    if penalty != 0:
        for s in NON_TERMINAL_STATES:
            rewards[s] = penalty
    
    # Dictionary of admissible actions per state
    adm_actions = {
        (0,0): ("D", "R"),
        (0,1): ("L", "R"),
        (0,2): ("L", "R", "D"),
        (1,0): ("D", "U"),
        (1,2): ("U", "D", "R"),
        (2,0): ("U", "R"),
        (2,1): ("L", "R"),
        (2,2): ("U", "R", "L"),
        (2,3): ("U", "L"),
    }
    
    # Dictionary of transition probabilities
    # NOTE: I've modified the instructor's implementation.
    #       I've removed all tautologies (agent doesn't stay in current state).
    transition_probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
    }
    
    # Assign missing environment attributes
    env.set(rewards, adm_actions, transition_probs)
    
    # Output line
    return env

# END DEF windy_standard_grid()


# Non-windy GridWorld standard test with no penalties.
#
def test_standard_grid():

    '''
    Non-windy GridWorld standard test with no penalties.
    '''

    # Start at bottom left (randomize later)
    ini_state = (2,0)
    # Action space 
    ACTION_SPACE = {"U", "D", "L", "R"}
    # Non terminal states
    NON_TERMINAL_STATES = {(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)}
    # Terminal states
    TERMINAL_STATES = {(0,3), (1,3)}
    
    # Instantiate:
    # 
    env = GridWorld_Windy_small(3, 4, ini_state, NON_TERMINAL_STATES, TERMINAL_STATES, ACTION_SPACE)

    
    # Dictionary of rewards
    # Not storing 0s
    rewards = {(0,3):1, (1,3): -1}
    
    # Dictionary of admissible actions per state
    adm_actions = {
        (0,0): ("D", "R"),
        (0,1): ("L", "R"),
        (0,2): ("L", "R", "D"),
        (1,0): ("D", "U"),
        (1,2): ("U", "D", "R"),
        (2,0): ("U", "R"),
        (2,1): ("L", "R"),
        (2,2): ("U", "R", "L"),
        (2,3): ("U", "L"),
    }
    
    # Dictionary of deterministic transitions:
    transition_probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        
        ((1, 2), 'U'): {(0, 2): 1.0},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
    }
    
    # Assign missing environment attributes
    env.set(rewards, adm_actions, transition_probs)
    
    # Output line
    return env

# END DEF test_standard_grid()