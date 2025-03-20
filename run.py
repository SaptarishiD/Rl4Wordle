from collections import defaultdict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os
import random

from environments import WordleEnv, WordleEnvMarkov
from ppo_agents import WordleFeatureExtractor, WordleFeatureExtractor_Markov
from trainer import WordleTrainingCallback
from heuristics import HeuristicWordleAgent
from utils import convert_feedback_to_wordle_format
from q_agents import QLearningAgent
from trainer import WordleTrainer
from datetime import datetime

def make_env(word_list_path, rank=0):
    """
    Create a WordleEnv environment for Stable Baselines3.
    
    Args:
        word_list_path: Path to the word list file
        rank: Environment rank (for vectorized environments)
        
    Returns:
        A function that creates an instance of the environment
    """
    def _init():
        env = WordleEnvMarkov(word_list_path, max_attempts=6, word_length=5)
        env = Monitor(env)
        return env
    return _init


def train_ppo_agent(word_list_path, total_timesteps=100000, log_dir='./logs'):
    """
    Train a PPO agent on the Wordle environment.
    
    Args:
        word_list_path: Path to the word list file
        total_timesteps: Number of training steps
        log_dir: Directory to save logs
        
    Returns:
        The trained PPO model
    """
    env = DummyVecEnv([make_env(word_list_path)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    eval_env = DummyVecEnv([make_env(word_list_path)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, training=False)
    
    policy_kwargs = {
        'features_extractor_class': WordleFeatureExtractor_Markov,
        'features_extractor_kwargs': {'features_dim': 256}
    }
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=log_dir
    )
    
    callback = WordleTrainingCallback(eval_env, check_freq=5000, log_dir=log_dir)
    
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    callback.plot_metrics()
    
    model.save(os.path.join(log_dir, "ppo_wordle"))
    
    return model


def evaluate_ppo_agent(model, word_list_path, num_episodes=50, render=True):
    """
    Evaluate a trained PPO agent on the Wordle environment.
    
    Args:
        model: The trained PPO model
        word_list_path: Path to the word list file
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        Evaluation results
    """
    eval_env = WordleEnvMarkov(word_list_path, max_attempts=6, word_length=5, render_mode="human" if render else None)
    
    try:
        with open(word_list_path, 'r') as f:
            words = [w.strip().lower() for w in f.readlines() if len(w.strip()) == 5 and w.strip().isalpha()]
    except FileNotFoundError:
        print(f"Warning: {word_list_path} not found. Using a small sample of words.")
        words = [
            'apple', 'baker', 'child', 'dance', 'early', 'first', 'grand', 'house', 'input',
            'jolly', 'knife', 'light', 'mouse', 'night', 'ocean', 'piano', 'queen', 'river',
            'sound', 'table', 'under', 'value', 'water', 'xenon', 'youth', 'zebra'
        ]
    
    wins = 0
    rewards = []
    attempts = []
    
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        used_words = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            word = words[action]
            if word in used_words:
                unused_words = [w for w in words if w not in used_words]
                if unused_words:
                    word = random.choice(unused_words)
                    action = words.index(word)
                
            used_words.append(word)
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if render:
                eval_env.render()
        
        rewards.append(episode_reward)
        won = eval_env.won
        wins += 1 if won else 0
        if won:
            attempts.append(eval_env.current_attempt)
        
        if render:
            print(f"Episode {episode+1}/{num_episodes} | " +
                  f"Result: {'Won' if won else 'Lost'} | " +
                  f"Word: {eval_env.target_word} | " +
                  f"Attempts: {eval_env.current_attempt}/{eval_env.max_attempts}")
    
    win_rate = wins / num_episodes
    avg_reward = sum(rewards) / num_episodes
    avg_attempts = sum(attempts) / len(attempts) if attempts else 0
    
    print(f"\nEvaluation Results:")
    print(f"Win Rate: {win_rate:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Attempts (when won): {avg_attempts:.2f}")
    
    return {
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'avg_attempts': avg_attempts
    }


def compare_with_heuristic(model, word_list_path, num_episodes=50):
    """
    Compare the PPO agent with a heuristic agent.
    
    Args:
        model: The trained PPO model
        word_list_path: Path to the word list file
        num_episodes: Number of episodes for comparison
        
    Returns:
        Comparison results
    """
    try:
        with open(word_list_path, 'r') as f:
            words = [w.strip().lower() for w in f.readlines() if len(w.strip()) == 5 and w.strip().isalpha()]
    except FileNotFoundError:
        print(f"Warning: {word_list_path} not found. Using a small sample of words.")
        words = [
            'apple', 'baker', 'child', 'dance', 'early', 'first', 'grand', 'house', 'input',
            'jolly', 'knife', 'light', 'mouse', 'night', 'ocean', 'piano', 'queen', 'river',
            'sound', 'table', 'under', 'value', 'water', 'xenon', 'youth', 'zebra'
        ]
    
    heuristic_agent = HeuristicWordleAgent(words)
    
    env = WordleEnvMarkov(word_list_path, max_attempts=6, word_length=5)
    
    target_words = random.sample(words, min(num_episodes, len(words)))
    
    results = {
        'ppo': {'wins': 0, 'attempts': []},
        'heuristic': {'wins': 0, 'attempts': []}
    }
    
    for i, target_word in enumerate(target_words):
        print(f"\nGame {i+1}/{num_episodes} - Target word: {target_word}")
        
        obs, _ = env.reset(options={'target_word': target_word})
        done = False
        used_words = []
        
        print("PPO agent playing...")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            word = words[action]
            
            if word in used_words:
                unused_words = [w for w in words if w not in used_words]
                if unused_words:
                    word = random.choice(unused_words)
                    action = words.index(word)
            
            used_words.append(word)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        ppo_won = env.won
        ppo_attempts = env.current_attempt
        results['ppo']['wins'] += 1 if ppo_won else 0
        if ppo_won:
            results['ppo']['attempts'].append(ppo_attempts)
        
        print(f"PPO result: {'Won' if ppo_won else 'Lost'} in {ppo_attempts} attempts")
        
        heuristic_agent.reset()
        env.reset(options={'target_word': target_word})
        
        print("Heuristic agent playing...")
        for attempt in range(env.max_attempts):
            guess = heuristic_agent.get_action()
            action = words.index(guess) if guess in words else 0
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            feedback = []
            for j in range(env.word_length):
                if env.board[attempt, j, env.CORRECT] == 1:
                    feedback.append(2)
                elif env.board[attempt, j, env.PRESENT] == 1:
                    feedback.append(1)
                else:
                    feedback.append(0)
            heuristic_agent.update(guess, feedback)
            
            if done:
                break
        
        heuristic_won = env.won
        heuristic_attempts = env.current_attempt
        results['heuristic']['wins'] += 1 if heuristic_won else 0
        if heuristic_won:
            results['heuristic']['attempts'].append(heuristic_attempts)
        
        print(f"Heuristic result: {'Won' if heuristic_won else 'Lost'} in {heuristic_attempts} attempts")
    
    ppo_win_rate = results['ppo']['wins'] / num_episodes
    heuristic_win_rate = results['heuristic']['wins'] / num_episodes
    
    ppo_avg_attempts = sum(results['ppo']['attempts']) / len(results['ppo']['attempts']) if results['ppo']['attempts'] else 0
    heuristic_avg_attempts = sum(results['heuristic']['attempts']) / len(results['heuristic']['attempts']) if results['heuristic']['attempts'] else 0
    
    print("\nComparison Results:")
    print(f"PPO Win Rate: {ppo_win_rate:.2f}")
    print(f"Heuristic Win Rate: {heuristic_win_rate:.2f}")
    print(f"PPO Average Attempts (when won): {ppo_avg_attempts:.2f}")
    print(f"Heuristic Average Attempts (when won): {heuristic_avg_attempts:.2f}")
    
    return results
def play_wordle_interactive(agent, word_list, target_word=None):
    """
    Play Wordle interactively with an agent.
    
    Args:
        agent: The Wordle agent
        word_list (list): List of valid words
        target_word (str): Optional target word (if None, one is selected randomly)
    """
    agent.reset()
    
    if target_word is None:
        target_word = random.choice(word_list)
    
    word_length = len(target_word)
    max_attempts = 6
    
    print(f"Playing Wordle (target word hidden, {word_length} letters)")
    
    for attempt in range(1, max_attempts + 1):
        guess = agent.get_action()
        
        if guess not in word_list:
            print(f"Invalid word: {guess}")
            continue
        
        feedback = []
        letter_counts = defaultdict(int)
        for letter in target_word:
            letter_counts[letter] += 1
        
        remaining_letters = letter_counts.copy()
        for i, letter in enumerate(guess):
            if letter == target_word[i]:
                feedback.append(2)  # Correct
                remaining_letters[letter] -= 1
            else:
                feedback.append(-1)  # To be determined
        
        for i, letter in enumerate(guess):
            if feedback[i] == -1:
                if letter in target_word and remaining_letters[letter] > 0:
                    feedback[i] = 1
                    remaining_letters[letter] -= 1
                else:
                    feedback[i] = 0
        
        print(f"Attempt {attempt}: {guess} {convert_feedback_to_wordle_format(feedback)}")
        
        agent.update(guess, feedback)
        
        if guess == target_word:
            print(f"Won in {attempt} attempts! The word was: {target_word}")
            return True
        
        if hasattr(agent, 'possible_solutions'):
            remaining = len(agent.possible_solutions)
            print(f"Possibilities remaining: {remaining}")
            if remaining <= 5:
                print(f"Possible words: {', '.join(agent.possible_solutions)}")
    
    print(f"Game over! The word was: {target_word}")
    return False


# def main():
#     """Main function to demonstrate the Wordle environment and agents"""
#     word_list_path = "target_words.txt"
    
#     sample_words = [
#         'apple', 'baker', 'child', 'dance', 'early', 'first', 'grand', 'house', 'input',
#         'jolly', 'knife', 'light', 'mouse', 'night', 'ocean', 'piano', 'queen', 'river',
#         'sound', 'table', 'under', 'value', 'water', 'xenon', 'youth', 'zebra'
#     ]
    
#     try:
#         with open(word_list_path, 'r') as f:
#             words = [w.strip().lower() for w in f.readlines() if len(w.strip()) == 5 and w.strip().isalpha()]
#     except FileNotFoundError:
#         print(f"Warning: {word_list_path} not found. Using a small sample of words.")
#         words = sample_words
    
#     env = WordleEnvMarkov(word_list_path, max_attempts=6, word_length=5, render_mode="human")
    
#     agent = QLearningAgent(
#         action_space=env.action_space,
#         learning_rate=0.1,
#         discount_factor=0.95,
#         exploration_rate=1.0,
#         exploration_decay=0.995,
#         min_exploration_rate=0.01
#     )
    
#     trainer = WordleTrainer(env, agent)
    
#     print("Training Q-learning agent...")
#     trainer.train(num_episodes=1000, eval_interval=100)
    
#     trainer.plot_metrics()
    
#     print("\nEvaluating trained agent:")
#     trainer.evaluate(num_episodes=50, render=True)
    
#     print("\nComparing with heuristic agent:")
#     heuristic_agent = HeuristicWordleAgent(words)
    
#     print("\nExample game with heuristic agent:")
#     play_wordle_interactive(heuristic_agent, words)


# if __name__ == "__main__":
#     main()
    
if __name__ == "__main__":
    word_list_path = "target_words.txt"

    log_dir = "./logs/ppo_wordle"
    os.makedirs(log_dir, exist_ok=True)
    print("Training PPO agent...")
    model = train_ppo_agent(word_list_path, total_timesteps=1000000, log_dir=log_dir)

    current_date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model.save(f"ppo_wordle{current_date_and_time}")