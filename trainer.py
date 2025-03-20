import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
class WordleTrainer:
    """
    Class for training and evaluating RL agents on the Wordle environment.
    """
    
    def __init__(self, env, agent, valid_word_indices=None):
        """
        Initialize the trainer.
        
        Args:
            env: The Wordle environment
            agent: The RL agent
            valid_word_indices: Optional list of valid word indices for actions
        """
        self.env = env
        self.agent = agent
        self.valid_word_indices = valid_word_indices if valid_word_indices is not None else list(range(env.action_space.n))
        
        self.episode_rewards = []
        self.win_rates = []
        self.win_history = []  
        
    def train(self, num_episodes, eval_interval=100):
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train for
            eval_interval: Number of episodes between evaluations
        """
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state, _ = self.env.reset()
            episode_reward = 0
            # Reset environment
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            # Track words used in this episode to avoid repeating guesses
            used_actions = set()
            
            while not done:
                valid_actions = [a for a in self.valid_word_indices if a not in used_actions]
                
                if not valid_actions:
                    break
                
                action = self.agent.get_action(state, valid_actions)
                used_actions.add(action)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.agent.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
            
            self.episode_rewards.append(episode_reward)
            self.win_history.append(1 if self.env.won else 0)
            
            self.agent.decay_exploration()
            
            if episode % eval_interval == 0:
                win_rate = np.mean(self.win_history[-eval_interval:])
                self.win_rates.append(win_rate)
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Win Rate: {win_rate:.2f} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Exploration Rate: {self.agent.exploration_rate:.2f}")
    
    def evaluate(self, num_episodes=100, render=False):
        """
        Evaluate the agent.
        
        Args:
            num_episodes: Number of episodes to evaluate for
            render: Whether to render the environment
            
        Returns:
            float: The win rate
        """
        wins = 0
        rewards = []
        attempts = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            used_actions = set()
            
            while not done:
                valid_actions = [a for a in self.valid_word_indices if a not in used_actions]
                if not valid_actions:
                    break
                
                state_key = self.agent._get_state_key(state)
                if state_key in self.agent.q_table:
                    q_values = self.agent.q_table[state_key]
                    valid_q = np.array([q_values[a] for a in valid_actions])
                    action = valid_actions[np.argmax(valid_q)]
                else:
                    action = np.random.choice(valid_actions)
                
                used_actions.add(action)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                
                if render:
                    self.env.render()
            
            rewards.append(episode_reward)
            wins += 1 if self.env.won else 0
            if self.env.won:
                attempts.append(self.env.current_attempt)
        
        win_rate = wins / num_episodes
        avg_reward = np.mean(rewards)
        avg_attempts = np.mean(attempts) if attempts else 0
        
        print(f"Evaluation Results:")
        print(f"Win Rate: {win_rate:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Attempts (when won): {avg_attempts:.2f}")
        
        return win_rate
    
    def plot_metrics(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        window_size = min(100, len(self.episode_rewards))
        rewards_smoothed = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(rewards_smoothed)
        ax1.set_title('Smoothed Episode Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        
        eval_interval = len(self.episode_rewards) // len(self.win_rates) if self.win_rates else 100
        episodes = np.arange(eval_interval, len(self.episode_rewards) + 1, eval_interval)
        ax2.plot(episodes, self.win_rates)
        ax2.set_title('Win Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()

class WordleTrainingCallback(BaseCallback):
    """
    Custom callback for logging training progress in the Wordle environment.
    """
    
    def __init__(self, eval_env, check_freq=1000, log_dir=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rewards = []
        self.win_rates = []
        self.eval_episodes = 20
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.eval_episodes
            )
            
            env = self.eval_env
            wins = 0
            for _ in range(self.eval_episodes):
                obs = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
            raw_env = getattr(self.eval_env, "venv", self.eval_env)
            raw_env = raw_env.envs[0] if hasattr(raw_env, "envs") else raw_env
            wins += int(raw_env.unwrapped.won)
            
            win_rate = wins / self.eval_episodes
            
            self.rewards.append(mean_reward)
            self.win_rates.append(win_rate)
            if self.verbose > 0:
                print(f"Step {self.n_calls} | Mean reward: {mean_reward:.2f} | Win rate: {win_rate:.2f}")
            
            if self.log_dir is not None:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(os.path.join(self.log_dir, 'metrics.csv'), 'a') as f:
                    f.write(f"{self.n_calls},{mean_reward},{win_rate}\n")
        
        return True
    
    def plot_metrics(self):
        """Plot training metrics."""
        if not self.rewards:
            print("No metrics to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        steps = [i * self.check_freq for i in range(len(self.rewards))]
        
        ax1.plot(steps, self.rewards)
        ax1.set_title('Mean Reward During Training')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Reward')
        
        ax2.plot(steps, self.win_rates)
        ax2.set_title('Win Rate During Training')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()