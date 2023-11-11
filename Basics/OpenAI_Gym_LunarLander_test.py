######################
## OPENAI GYM TESTS ##
######################

from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
import gym
from IPython import display
#%matplotlib
#matplotlib.rcParams['interactive']==True

if __name__ == '__main__':
    # Initialize the environment
    lunland = gym.make('LunarLander-v2')
    lunland.reset()
    
    # Loop for dynamic display
    img = plt.imshow(lunland.render('rgb_array')) # should be called once only
    
    for i in range(50):
        print(f"Beginning step {i}")
        img.set_data(lunland.render('rgb_array'))
        plt.show()
        #display.display(plt.gcf())
        #display.clear_output(wait=True)
        action = lunland.action_space.sample()
        lunland.step(action)
    
    
    # Close Pygame
    lunland.close()