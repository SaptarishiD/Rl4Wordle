import gymnasium as gym
from gymnasium import spaces
import numpy as np
from myQwordleEnv import WordleMetaEnv 
from stable_baselines3 import PPO
from tqdm import tqdm

class WordleGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, word_list_path='target_words.txt'):
        super(WordleGymEnv, self).__init__()
    
        self.env = WordleMetaEnv()
   
        self.action_space = spaces.Discrete(len(self.env.action_space))    
        self.observation_space = spaces.Box(low=0, high=5, shape=(3,), dtype=np.int32)
    
    def reset(self, **kwargs):
        state = self.env.reset() 
    
        return np.array(state, dtype=np.int32), {}
    
    def step(self, action):
    
        state, reward, done = self.env.step(action)
    
        return np.array(state, dtype=np.int32), reward, done, False, {}
    
    def render(self, mode="human"):
    
        print("Guessed words so far:", self.env.env.guessed_words)

env = WordleGymEnv()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs_meta_ppo")

model.learn(total_timesteps=500000, progress_bar=True)

model.save("meta_ppo_wordle.zip")

