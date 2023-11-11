####################################
### DQN AGENT - LUNAR LANDER TEST ##
####################################
## 2022/06/05; Ahmed J Zerouali
## Notes: - I'm not making a recording of the training here.
##        - Will try to execute this file in my Paperspace account.

from __future__ import print_function, division
from builtins import range


# NumPy, Pandas, matplotlib, TF2, Keras, Gym, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# DQN imports
from DQN_Target_Net import Replay_Buffer, DQN_Agent

if __name__ == "__main__":
    # Disable eager execution (NOTES)
    tf.compat.v1.disable_eager_execution

    # Create environment
    env = gym.make('LunarLander-v2')

    # Max no. of episodes and steps:
    #n_games = 500
    n_games = 500
    N_steps_max = 2000
    N_steps_update = 50

    # Deep RL hyperparameters:
    learn_rate = 0.001
    gamma = 0.99
    eps_ini = 1.0
    eps_dec = 5e-5
    eps_min = 0.01
    batch_size = 64
    mem_size = 100000
    model_file_name = 'dqn_tgt_net.h5'

    # Ini filenames 
    # Visualization: gif filename
    rec_gif_fname = "LunarLander_test.gif"
    # Total scores
    scores_fname = "LunarLander_Scores.csv"
    # Epsilon
    epsilon_hist_fname = "LunarLander_Epsilon.csv"


    # Create agent
    """
    class DQN_Agent():

        
        # Constructor
                    # Environment dimensions and action space params
        def __init__(self, input_dims, discrete_actions, n_actions,  \
                     # RL hyperparameters
                     learn_rate, gamma, epsilon, batch_size, mem_size = 1000000, \
                     # Decrement and lower bound for eps-greedy
                     epsion_dec = 1e-3, epsilon_min =0.01, \
                     # Filename model for saving
                     dqn_fname = 'dqn_model.h5'):    
    """
    agent = DQN_Agent(input_dims = env.observation_space.shape, discrete_actions= True, n_actions = env.action_space.n,\
                     learn_rate = learn_rate, gamma=gamma, epsilon=eps_ini, batch_size = batch_size, mem_size = mem_size, \
                     epsilon_dec =eps_dec, epsilon_min =eps_min, dqn_fname = model_file_name)
    print("DQN_Agent created...")

    # Init. score and episode (epsilon) histories
    scores_hist = []
    eps_hist = []

    # Build the DQN (Forgot this on 1st exec...)
    agent.build_DQN(256, 256)
    
    # Init. timer of main loop
    dqn_begin_time = datetime.now()

    print("Starting main loop...")
    # Loop over episodes:
    for i in range(n_games):
        
        # Init. episode
        done = False
        score = 0.0
        s = env.reset()
        # DEBUG: Fix no. of max steps
        N_steps = 0
        # Add a first frame
        #img= ax.imshow(env.render('rgb_array'), animated = True)
        #ims.append([img])
        
        # Loop over steps in episodes
        while not done and (N_steps < N_steps_max):
            
            # Choose an action
            a = agent.choose_action(s)
            # Get reward and next state
            s_, r, done, info = env.step(a)
            # Update score
            score += r
            # Store transition
            agent.store_transition(s, a, s_, r, done)
            
            
            # Learn
            agent.train_dqn(N_steps, i, notify_end_train=True)
            # Update state
            s = s_
            # DEBUG: Increment N_steps
            N_steps += 1
            # Update target net weights
            if (N_steps >= batch_size) and ((N_steps % N_steps_update) == 0):
                agent.update_trgt_wts()
            
        # END while over episode steps
        
        # Update epsilon and score histories
        eps_hist.append(agent.epsilon)
        scores_hist.append(score)

    # END OF MAIN DQN LOOP (i over episodes) 
    print("Exited main loop.")
    
    # Exec time of main loop
    dqn_exec_time = datetime.now()-dqn_begin_time
    
    env.close()
    # New
    print(f"Writing report and score histories...")
    

    # Save total scores and epsilon decrease
    ### NOTE: Modify the code below to clean-out output
    df_scores = pd.DataFrame(scores_hist, columns = ['Tot. Score'])
    df_scores.to_csv(scores_fname)
    # Compute mean, min and max of scores, added to report below.
    scores_mean = df_scores["Tot. Score"].mean()
    scores_max = df_scores["Tot. Score"].max()
    scores_min = df_scores["Tot. Score"].min()
    df_epsilon = pd.DataFrame(eps_hist, columns = ['Epsilon'])
    df_epsilon.to_csv(epsilon_hist_fname)

    # Save trained DQN model:
    agent.save_dqn()
    print(f"Saved DQN model parameters...")

    # Report file open
    dqn_begin_time = datetime.now()
    report_fname = "Report_LunLand_DQN_"\
                    +str(dqn_begin_time.year-2000)+str(dqn_begin_time.month)+str(dqn_begin_time.day)\
                    +str(dqn_begin_time.hour)+str(dqn_begin_time.minute)+".txt"
    report_file = open(report_fname, mode = "w+")
    # Title
    report_file.write(f"Deep Q-learning agent - {dqn_begin_time}:\n")
    # Write report to file
    report_file.write(f"- Execution over {n_games} episodes with {N_steps_max} max. steps completed.\n")
    report_file.write(f"- Total training time: {dqn_exec_time}.\n")
    report_file.write(f"- Average score over {n_games} episodes: {scores_mean}.\n")
    report_file.write(f"- Highest score: {scores_max}.\n")
    report_file.write(f"- Lowest score: {scores_min}.\n")
    report_file.write(f"- DRL Hypeparameters: * learn_rate = {learn_rate},\n")
    report_file.write(f"                      * gamma = {gamma},\n")
    report_file.write(f"                      * eps_ini = {eps_ini},\n")
    report_file.write(f"                      * eps_dec = {eps_dec},\n")
    report_file.write(f"                      * eps_min = {eps_min},\n")
    report_file.write(f"                      * batch_size = {batch_size},\n")
    report_file.write(f"                      * mem_size = {mem_size},\n")
    report_file.write(f"                      * Saved model filename = {model_file_name}.\n")
    # Close report file
    report_file.close()
