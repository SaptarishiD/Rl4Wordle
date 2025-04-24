#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gymnasium as gym
import random

import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
import itertools

from myQwordleEnv import WordleMetaEnv, WordleQEnv, MyAgent
from tqdm import tqdm
import time


# In[2]:


f = open('target_words.txt', 'r')
target_words = f.read().splitlines()
f.close()


# In[3]:


len(target_words)


# We can try a form of Meta-Learning to try to reduce the search space by making the agent learn to choose between few well-defined actions

# In[4]:


def Q_Learning(num_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    env = WordleMetaEnv(debug=False, word_list_path='target_words.txt')
    actions_len = len(env.action_space)
    Q = defaultdict(lambda: np.zeros(actions_len))
    
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)
    wins = np.zeros(num_episodes)
    
    for i in tqdm(range(num_episodes)):
        state = env.reset()

        # this follows the logic we did in class and extension to an action space of length actions_len by being inspired from the epsilon greedy policy implemented here but for BlackJack: https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Control%20with%20Epsilon-Greedy%20Policies%20Solution.ipynb
        
        action_probs = np.ones(actions_len, dtype = float) * (epsilon / actions_len)
        action_prob_len = len(action_probs)
        best_action = np.argmax(Q[state])
        action_probs[best_action] = (1.0 - epsilon + (epsilon / actions_len))
        episode_reward = 0
        episode_length = 0
        
        while True:
            # choose action index according to the probability distribution
            action = np.random.choice(np.arange(action_prob_len), p = action_probs / np.sum(action_probs)) # so that probs add up to one
            next, reward, done = env.step(action)
            episode_reward += reward

            opt_action = np.argmax(Q[next])

            #Q(s,a) ← Q(s,a) + α[ r + γ·max Q(s',a') - Q(s,a) ] 

            # temporal differences

            new_info = reward + gamma * Q[next][opt_action]
            oldQsa = Q[state][action]
            Q[state][action] += alpha * ( new_info - oldQsa)

            if done:
                episode_lengths[i] = episode_length + 1
                episode_rewards[i] = episode_reward
                if env.env.won_game == 'yes':
                    wins[i] = 1
                elif env.env.won_game == 'no':
                    wins[i] = 0
                    episode_lengths[i] = 7
                    
                break
                
            state = next
            episode_length += 1

            
    return Q, (episode_lengths, episode_rewards, wins)


# In[5]:


def plot_game_statistics(stats, window_size=500):
    
    episode_lengths = stats[0]
    episode_rewards = stats[1]
    win_records = stats[2]
    num_episodes = len(episode_lengths)
    
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    if num_episodes > window_size:
        ma_lengths = moving_average(episode_lengths, window_size)
        ma_rewards = moving_average(episode_rewards, window_size)
        ma_wins = moving_average(win_records, window_size)
        x_range = np.arange(window_size-1, num_episodes)
    else:
        ma_lengths = episode_lengths
        ma_rewards = episode_rewards
        ma_wins = win_records
        x_range = np.arange(num_episodes)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    ax1.plot(x_range, ma_lengths, 'steelblue', linewidth=2)
    ax1.set_title('Wordle Game Lengths Over Time', fontsize=16)
    ax1.set_ylabel('Number of Guesses', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x_range, ma_rewards, 'forestgreen', linewidth=2)
    ax2.set_title('Wordle Game Rewards Over Time', fontsize=16)
    ax2.set_ylabel('Total Reward', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(x_range, ma_wins, 'darkorange', linewidth=2)
    ax3.set_title('Wordle Win Rate Over Time', fontsize=16)
    ax3.set_ylabel('Win Rate', fontsize=14)
    ax3.set_xlabel('Episode', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    labels, counts = np.unique(episode_lengths, return_counts=True)
    plt.bar(labels, counts, align='center', color='steelblue')
    plt.xlabel('Game Length (Number of Guesses)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Distribution of Wordle Game Lengths', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(labels)
    plt.show()
    
    win_percentage = np.mean(win_records) * 100
    
    print(f"Win Rate: {win_percentage:.2f}%")
    print(f"Average Game Length: {np.mean(episode_lengths):.2f} guesses")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    
    if np.sum(win_records) > 0:
        win_lengths = episode_lengths[win_records == 1]
        plt.figure(figsize=(10, 6))
        win_labels, win_counts = np.unique(win_lengths, return_counts=True)
        plt.bar(win_labels, win_counts, align='center', color='forestgreen')
        plt.xlabel('Game Length for Wins (Number of Guesses)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Distribution of Game Lengths for Wins', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.xticks(win_labels)
        plt.show()


# In[6]:


def train_and_evaluate(num_episodes=50000, epsilon=0.1, alpha=0.05, gamma=0.5):
    return Q_Learning(num_episodes=num_episodes, epsilon=epsilon, alpha=alpha, gamma=gamma)

Q, training_details = train_and_evaluate(num_episodes=10000)
plot_game_statistics(training_details)


# In[7]:


import pickle
with open('Q_table_no_intermediate_targets.pkl', 'wb') as f:
    pickle.dump(dict(Q), f)
# load it
with open('Q_table_no_intermediate_targets.pkl', 'rb') as f:
    Q_loaded = pickle.load(f)


# In[ ]:


def run_episode_for_target(target, Q):
    env = WordleMetaEnv(debug=False, word_list_path='target_words.txt')
    env.reset()
    env.env.target_word = target  
    env.env.attempts = 0  
    env.env.guessed_words = []
    
    done = False
    state = (0, 0, 0)
    moves = 0


    while not done:
        if state in Q:
            action = np.argmax(Q[state])
        else:
            action = random.choice(env.agent.action_space)
            
        state, reward, done = env.step(action)
        moves += 1
    if env.env.won_game == 'yes':
        return moves
    else:
        env.env.won_game = 'no'
        return moves+1


moves_list = []
f = open('target_words.txt', 'r')
new_target_words = f.read().splitlines()
f.close()
for word in tqdm(new_target_words):
    m = run_episode_for_target(word, Q_loaded)
    moves_list.append(m)

plt.figure(figsize=(12, 6))
plt.hist(moves_list, bins=range(1, max(moves_list) + 2), edgecolor='black', color='green', align='left')
plt.xlabel("Number of Moves")
plt.ylabel("Number of Words")
plt.title("Histogram of Number of Moves Taken by Learned Q-Table on Each Target Word")
plt.xticks(range(1, max(moves_list) + 1))
plt.grid(True)
plt.show()


# In[9]:


print("Average game length:", np.mean(moves_list))

