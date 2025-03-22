# <myQwordle.py>
#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import gymnasium as gym
import random

import plottingepi


from collections import defaultdict
import itertools

from myQwordleEnv import WordleMetaEnv, WordleQEnv, MyAgent
from tqdm import tqdm


# In[2]:


f = open('target_words.txt', 'r')
target_words = f.read().splitlines()
f.close()


# In[3]:


len(target_words)


# In[4]:


# TODO:
"""
Make plot for which actions were chosen most of the time by the agent by the meta learning agent
"""


# We can try a form of Meta-Learning to try to reduce the search space by making the agent learn to choose between few well-defined actions instead

# In[ ]:


e = WordleMetaEnv()


# In[7]:


def eps_greedy(Q_table, state, actions_len, eps = 0.1):
	action_probs = np.ones(actions_len, dtype = float) * eps / actions_len
	best_action = np.argmax(Q_table[state])
	action_probs[best_action] += (1.0 - eps + eps/actions_len)
	return action_probs


# In[ ]:


def Q_Learning(env, num_episodes, discount_factor = 1.0, alpha = 0.5, eps = 0.1):

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    actions_len = len(env.action_space)
    Q = defaultdict(lambda: np.zeros(actions_len))
    # Keeps track of useful statistics
    stats = plottingepi.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))
    win_records = np.zeros(num_episodes)
    
    # Create an eps greedy policy function
    # appropriately for environment action space

    wordlistdict = defaultdict()
    
    # For every episode
    for ith_episode in tqdm(range(num_episodes)):

        # Reset the environment and pick the first action
        state = env.reset()
        
        for t in itertools.count():
            #print("this is q", Q)
            # get probabilities of all actions from current state
            action_probabilities = eps_greedy(Q, state, actions_len, eps)
            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                    len(action_probabilities)),
                    p = action_probabilities)
            # take action and get reward, transit to next state
            next_state, reward, done = env.step(action)
            
            #next_state = sum(next_state)
            
            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t
            if(env.env.win =='win'):
                win_records[ith_episode] = 1
            elif(env.env.win == 'lose'):
                win_records[ith_episode] = 0
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])	
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            

            # done is True if episode terminated
            if done:
                break

            state = next_state
        #wordlistdict[env.env.word] = env.tmp_wordlist
            
    return Q, stats, win_records


# In[ ]:


qouts9 = []
souts9 = []
wouts9 = []
# why loop?
for i in range(0,1):
    # print(i)
    tmpq, tmps, tmpw = Q_Learning(WordleMetaEnv(), 
                           50000, epsilon=0.1,alpha=0.05, discount_factor=0.05, prints=False)
    qouts9.append(tmpq)
    souts9.append(tmps)
    wouts9.append(tmpw)
print(np.mean([sum(x) for x in wouts9]))


# In[ ]:


print(tmpq.items())


# In[ ]:


qouts9


# In[ ]:


souts9


# In[ ]:


2


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=7,5
# plt.style.use('seaborn-white')
#plt.rcParams['axes.facecolor']='white'
#plt.rcParams["axes.edgecolor"] = "black"
#plt.rcParams["axes.linewidth"] = 2
labels, counts = np.unique(souts9[0].episode_lengths+1, return_counts=True)
plt.bar(labels, counts, align='center', color='steelblue')

#plt.hist(s1.episode_lengths+1, bins=10, align='mid')
#plt.title('10,000 trials', fontsize=24)
plt.xlabel('game length', fontsize=24)
plt.ylabel('counts', fontsize=24)
plt.gca().set_xticks(labels)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()


# In[ ]:


labels, counts


# In[ ]:


print(qouts9, souts9, wouts9)


# In[ ]:


print


# In[ ]:


import seaborn as sns
plt.rcParams['figure.figsize']=7,5
# plt.style.use('seaborn-white')
#plt.rcParams['axes.facecolor']='white'
#plt.rcParams["axes.edgecolor"] = "black"
#plt.rcParams["axes.linewidth"] = 2
labels, counts = np.unique(souts9[0].episode_lengths+1, return_counts=True)
plt.bar(labels, counts, align='center', color='steelblue')

#plt.hist(s1.episode_lengths+1, bins=10, align='mid')
#plt.title('10,000 trials', fontsize=24)
plt.xlabel('game length', fontsize=24)
plt.ylabel('counts', fontsize=24)
plt.gca().set_xticks(labels)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()


# In[ ]:


## count of times hitting on first or second guess
print('num wins on first try:', len(np.where(souts9[0].episode_lengths+1==1)[0]))
print('num wins on second try:', len(np.where(souts9[0].episode_lengths+1==2)[0]))
print('num wins on third try:', len(np.where(souts9[0].episode_lengths+1==3)[0]))

# </myQwordle.py>