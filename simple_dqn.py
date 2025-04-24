#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[28]:


import random
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from collections import Counter


from environments import WordleEnv, WordleEnvMarkov
def nested_list_to_tuple(nested_list):
    return tuple(nested_list_to_tuple(i) if isinstance(i, list) else i for i in nested_list)

# -----------------------------
# Q-Learning Agent Definition
# -----------------------------
import random
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from collections import deque

def train_q_learning(env, num_episodes=1000, alpha=0.5, gamma=0.9, epsilon=0.2, log_dir="./logs/wordle_standard_q"):
    """
    A simple tabular Q-learning algorithm that trains on the Wordle environment with TensorBoard logging.
    
    The state is defined as a tuple: (attempt_number, board_state)
    where board_state is a tuple of length word_length.
    
    This version prevents the agent from guessing the same word twice in a single episode.
    """
    writer = SummaryWriter(log_dir=f"{log_dir}_{time.strftime('%Y%m%d-%H%M%S')}")
    
    Q = {}
    
    episode_rewards = []
    episode_lengths = []
    win_rate_window = deque(maxlen=100)
    q_values_history = []
    exploration_rates = []
    unique_states_count = []
    
    epsilon_start = epsilon
    epsilon_end = 0.01
    epsilon_decay = 0.995

    def get_state(observation):
        """Extract information state from observation"""
        attempt = observation["attempt"].item()
        if "board" in observation and observation["board"] is not None:
            board = nested_list_to_tuple(observation)
        else:
            board = tuple()
            
        if attempt == 0:
            return "initial"
        return (attempt, board)
    
    def choose_action(state, guessed_actions, current_epsilon):
        """Choose action using epsilon-greedy policy"""
        allowed_actions = [a for a in range(env.action_space.n) if a not in guessed_actions]
        
        if not allowed_actions:
            return env.action_space.sample(), True
            
        if random.random() < current_epsilon or state not in Q:
            return random.choice(allowed_actions), True
        else:
            q_values = Q[state]
            allowed_q = {a: q_values[a] for a in allowed_actions}
            return max(allowed_q, key=allowed_q.get), False
    
    def update_Q(state, action, reward, next_state, done):
        """Update Q-value using Q-learning update rule"""
        if state not in Q:
            Q[state] = {a: 0 for a in range(env.action_space.n)}
        if not done and next_state not in Q:
            Q[next_state] = {a: 0 for a in range(env.action_space.n)}
            
        best_next = max(Q[next_state].values()) if not done else 0
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
        
        return Q[state][action]
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = get_state(observation)
        guessed_actions = set() 
        done = False
        
        episode_reward = 0
        episode_step = 0
        exploration_count = 0
        
        current_epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        episode_q_values = []
        
        while not done:
            action, is_exploration = choose_action(state, guessed_actions, current_epsilon)
            guessed_actions.add(action)
            if is_exploration:
                exploration_count += 1
            
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = get_state(next_observation)
            
            new_q_value = update_Q(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_step += 1
            episode_q_values.append(new_q_value)
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        win_rate_window.append(1 if env.won else 0)
        exploration_rates.append(exploration_count / episode_step if episode_step > 0 else 0)
        
        avg_q_value = sum(episode_q_values) / len(episode_q_values) if episode_q_values else 0
        q_values_history.append(avg_q_value)
        
        unique_states_count.append(len(Q))
        
        writer.add_scalar('Metrics/Reward', episode_reward, episode)
        writer.add_scalar('Metrics/Episode_Length', episode_step, episode)
        writer.add_scalar('Metrics/Win_Rate', sum(win_rate_window) / len(win_rate_window), episode)
        writer.add_scalar('Metrics/Exploration_Rate', exploration_rates[-1], episode)
        writer.add_scalar('Metrics/Average_Q_Value', avg_q_value, episode)
        writer.add_scalar('Metrics/Unique_States', unique_states_count[-1], episode)
        writer.add_scalar('Hyperparameters/Epsilon', current_epsilon, episode)
        writer.add_scalar('Hyperparameters/Learning_Rate', alpha, episode)
        
        if hasattr(env, 'won') and env.won and hasattr(env, 'target_word') and episode % 50 == 0:
            writer.add_text('Examples/Won_Games', 
                           f"Episode {episode}: Solved '{env.target_word}' in {episode_step} attempts", 
                           episode)
        
        if (episode + 1) % 100 == 0:
            win_rate = sum(win_rate_window) / len(win_rate_window) if win_rate_window else 0
            print(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f} | "
                  f"Steps: {episode_step} | Win Rate: {win_rate:.2f} | "
                  f"Epsilon: {current_epsilon:.4f} | Q-states: {len(Q)}")
    
    writer.add_histogram('Histograms/Episode_Rewards', np.array(episode_rewards), 0)
    writer.add_histogram('Histograms/Episode_Lengths', np.array(episode_lengths), 0)
    writer.add_histogram('Histograms/Q_Values', np.array(q_values_history), 0)
    
    writer.close()
    
    return Q


# -----------------------------
# Testing the Trained Agent with TensorBoard logging
# -----------------------------
def test_agent(env, Q, log_dir="./logs/wordle_standard_test"):
    """Test the trained agent and log results to TensorBoard"""
    writer = SummaryWriter(log_dir=f"{log_dir}_{time.strftime('%Y%m%d-%H%M%S')}")
    
    observation, _ = env.reset()
    target_word = env.target_word if hasattr(env, 'target_word') else "Unknown"
    
    state = get_state(observation)
    guessed_actions = set()
    done = False
    
    step = 0
    total_reward = 0
    guesses = []
    
    print("\nTesting trained agent:")
    while not done:
        allowed_actions = [a for a in range(env.action_space.n) if a not in guessed_actions]
        if not allowed_actions:
            action = env.action_space.sample()
        elif state in Q:
            allowed_q = {a: Q[state][a] for a in allowed_actions if a in Q[state]}
            if allowed_q:
                action = max(allowed_q, key=allowed_q.get)
            else:
                action = random.choice(allowed_actions)
        else:
            action = random.choice(allowed_actions)
        
        guessed_actions.add(action)
        
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        word = env.valid_words[action] if hasattr(env, 'valid_words') else f"Action_{action}"
        board = observation['board'] if 'board' in observation else None
        
        guesses.append(word)
        total_reward += reward
        step += 1
        
        writer.add_scalar('Test/Step_Reward', reward, step)
        
        print(f"Guess: {word}, Board: {board}, Reward: {reward}")
        
        state = get_state(observation)
    
    success = hasattr(env, 'won') and env.won
    writer.add_scalar('Test/Total_Reward', total_reward, 0)
    writer.add_scalar('Test/Steps', step, 0)
    writer.add_scalar('Test/Success', 1 if success else 0, 0)
    
    summary = f"Target word: {target_word}\n"
    summary += f"Success: {success}\n"
    summary += f"Steps: {step}\n"
    summary += f"Guesses: {', '.join(guesses)}\n"
    summary += f"Total reward: {total_reward}"
    
    writer.add_text('Test/Game_Summary', summary, 0)
    
    if hasattr(env, 'render'):
        env.render()
    
    writer.close()
    return step, success, total_reward

def get_state(observation):
    """Helper function to extract state from observation (used in testing)"""
    attempt = observation["attempt"].item()
    if "board" in observation and observation["board"] is not None:
        board = nested_list_to_tuple(observation)
    else:
        board = tuple()
        
    if attempt == 0:
        return "initial"
        
    return (attempt, board)

def simulate_game_with_target(env, Q, target_word, writer):
    """
    Simulate a single game with the target word fixed to target_word with TensorBoard logging.
    Uses the learned Q-values to choose actions. Returns the number of moves
    taken to solve the word if successful, or 7 if the agent fails within 6 moves.
    """
    observation, _ = env.reset()
    
    if hasattr(env, 'target'):
        env.target = target_word
    elif hasattr(env, 'target_word'):
        env.target_word = target_word
    
    state = get_state(observation)
    guessed_actions = set()
    done = False
    step = 0
    guesses = []
    total_reward = 0

    while not done:
        step += 1
        allowed_actions = [a for a in range(env.action_space.n) if a not in guessed_actions]
        if not allowed_actions:
            action = env.action_space.sample()
        elif state in Q:
            allowed_q = {a: Q[state][a] for a in allowed_actions if a in Q[state]}
            if allowed_q:
                action = max(allowed_q, key=allowed_q.get)
            else:
                action = random.choice(allowed_actions)
        else:
            action = random.choice(allowed_actions)
            
        guessed_actions.add(action)
        
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if writer:
            word = env.valid_words[action] if hasattr(env, 'valid_words') else f"Action_{action}"
            guesses.append(word)
            writer.add_scalar('Simulation/Step_Reward', reward, step)
            
        state = get_state(observation)
    
    success = False
    if hasattr(env, 'won'):
        success = env.won
    elif hasattr(env, 'success'):
        success = env.success
    else:
        success = total_reward > 0
    
    attempts = 0
    if hasattr(env, 'attempt'):
        attempts = env.attempt
    else:
        attempts = step
    
    result = attempts if success else 7
    
    if writer:
        writer.add_scalar('Simulation/Total_Reward', total_reward, 0)
        writer.add_scalar('Simulation/Success', 1 if success else 0, 0)
        writer.add_scalar('Simulation/Attempts', attempts, 0)
        
        summary = f"Target word: {target_word}\n"
        summary += f"Success: {success}\n"
        summary += f"Attempts: {attempts}\n"
        summary += f"Guesses: {', '.join(guesses)}\n"
        summary += f"Total reward: {total_reward}"
        
        writer.add_text('Simulation/Game_Summary', summary, 0)
        writer.close()
    
    return result

def evaluate_agent(env, Q, valid_words=None, num_words=100, log_dir="./logs/wordle_evaluation"):
    """
    Evaluate the trained agent on multiple target words and log aggregate statistics.
    
    Args:
        env: Wordle environment
        Q: Trained Q-table
        valid_words: List of target words to evaluate on (if None, uses random words)
        num_words: Number of words to evaluate (if valid_words is None)
        log_dir: Directory for TensorBoard logs
    
    Returns:
        results: Dictionary of evaluation metrics
    """
    writer = SummaryWriter(log_dir=f"{log_dir}_{time.strftime('%Y%m%d-%H%M%S')}")
    
    if valid_words is None:
        if hasattr(env, 'valid_words'):
            valid_words = random.sample(env.valid_words, min(num_words, len(env.valid_words)))
        else:
            valid_words = [f"word_{i}" for i in range(num_words)]
    
    results = []
    success_count = 0
    attempts_by_success = []
    
    for i, target_word in enumerate(valid_words):
        attempts = simulate_game_with_target(env, Q, target_word)
        
        success = attempts < 7  # 7 means failure
        if success:
            success_count += 1
            attempts_by_success.append(attempts)
        
        results.append((target_word, attempts, success))
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(valid_words)} words | Success rate: {success_count/(i+1):.2f}")
        
        writer.add_scalar('Evaluation/Success', 1 if success else 0, i)
        writer.add_scalar('Evaluation/Attempts', attempts if attempts < 7 else 7, i)
    
    success_rate = success_count / len(valid_words)
    avg_attempts = sum(attempts_by_success) / max(1, success_count)
    
    attempts_dist = {i: attempts_by_success.count(i) for i in range(1, 7)}
    
    writer.add_scalar('Evaluation/Overall_Success_Rate', success_rate, 0)
    writer.add_scalar('Evaluation/Average_Attempts_When_Successful', avg_attempts, 0)
    
    for attempts, count in attempts_dist.items():
        if count > 0:
            percentage = count / max(1, success_count)
            writer.add_scalar('Evaluation/Attempts_Distribution', percentage, attempts)
    
    writer.add_histogram('Evaluation/Attempts_Histogram', np.array(attempts_by_success), 0)
    
    summary = f"Evaluation Results:\n"
    summary += f"Total words: {len(valid_words)}\n"
    summary += f"Success rate: {success_rate:.2f}\n"
    summary += f"Average attempts when successful: {avg_attempts:.2f}\n\n"
    summary += f"Attempts distribution:\n"
    for attempts, count in sorted(attempts_dist.items()):
        percentage = count / max(1, success_count) * 100
        summary += f"  {attempts} attempts: {count} words ({percentage:.1f}%)\n"
    
    writer.add_text('Evaluation/Summary', summary, 0)
    
    writer.close()
    
    return {
        "success_rate": success_rate,
        "avg_attempts": avg_attempts,
        "attempts_distribution": attempts_dist,
        "results": results
    }


# In[29]:


env = WordleEnv(word_list_path="target_words.txt")
Q = train_q_learning(env, num_episodes=1000)
move_counts = []
writer = SummaryWriter(log_dir="./logs/wordle_q_learning")
for target_word in tqdm(env.valid_words):
    moves = simulate_game_with_target(env, Q, target_word, writer)
    move_counts.append(moves)


# In[30]:


def train_q_learning_finite_horizon(env, num_episodes=1000, alpha=0.5, gamma=0.9, epsilon=0.2, word_length=5, 
                                    num_letters=26, max_attempts=6, log_dir="./logs/wordle_q_learning_information"):
    """
    Finite-horizon Q-learning algorithm for Wordle, using an information state.
    This implementation explicitly accounts for the remaining attempts in the Q-table.

    Args:
        env: The Wordle environment
        num_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        word_length: Length of the target word
        num_letters: Number of possible letters (26 for English alphabet)
        max_attempts: Maximum number of attempts allowed in Wordle (typically 6)
        log_dir: Directory for TensorBoard logs

    Returns:
        Q: The learned Q-table, indexed by (state, attempts_remaining)
    """
    writer = SummaryWriter(log_dir=f"{log_dir}_{time.strftime('%Y%m%d-%H%M%S')}")
    
    Q = {}  # Q-table indexed by (state, attempts_remaining)
    
    episode_rewards = []
    episode_lengths = []
    win_rate_window = deque(maxlen=100)
    q_values_history = []
    exploration_rates = []
    unique_states_count = []
    
    epsilon_start = epsilon
    epsilon_end = 0.01
    epsilon_decay = 0.995

    def get_information_state(observation):
        """
        Extract a compact information state from the observation, mimicking WordleFeatureExtractor_Markov.
        This version directly mirrors the logic of the `forward` method, adapted for a single environment.
        """

        state = torch.zeros((word_length, num_letters))
        greens = {}  # {letter_idx: [positions]}
        yellows = {}  # {letter_idx: [positions]}
        blacks = {}  # {letter_idx: [positions]}  for truly absent letters
        missing_letters = {pos: [] for pos in range(word_length)}

        attempt_idx = observation['attempt'].item()

        if attempt_idx == 0:
            return "initial"

        for guess_idx in range(attempt_idx):
            last_feedback = observation['board'][guess_idx]
            last_guess = observation['guesses'][guess_idx]

            if (last_guess < 0).any():
                continue  
            for idx, (feed, letter) in enumerate(zip(last_feedback, last_guess)):
                letter_item = letter.item()
                if feed == 2:  # Green
                    if letter_item not in greens:
                        greens[letter_item] = []
                    greens[letter_item].append(idx)

                elif feed == 1:  # Yellow
                    if letter_item not in yellows:
                        yellows[letter_item] = []
                    yellows[letter_item].append(idx)
                elif feed == 0:  # Black (Gray)
                    if letter_item not in blacks:
                        blacks[letter_item] = []
                    blacks[letter_item].append(idx)

        # Process green positions
        for letter_idx, positions in greens.items():
            for pos in positions:
                state[pos, letter_idx] = 1
                for other_letter in range(num_letters):
                    if other_letter != letter_idx:
                        state[pos, other_letter] = -1

        # Process yellows *after* greens
        for letter_idx, positions in yellows.items():

            # Exclude yellows from the positions, blacks for that letter and greens from being candidates
            candidate_positions = [p for p in range(word_length) if p not in positions and p not in greens.get(letter_idx,[]) and 
                                    (p not in blacks.get(letter_idx, []))]
            for pos in positions:
                state[pos, letter_idx] = -1
                if letter_idx not in missing_letters[pos]:
                    missing_letters[pos].append(letter_idx)

            if candidate_positions:
                yellow_value = min(1.0, len(positions) / len(candidate_positions))

                for pos in candidate_positions:
                  state[pos, letter_idx] = yellow_value
                  if yellow_value == 1:  #yellow confirmed at position
                    for other_letter in range(num_letters):
                      if other_letter != letter_idx:
                        state[pos, other_letter] = -1

        # Process blacks *after* greens and yellows
        for letter_idx, positions in blacks.items():
            has_positive_info = (state[:, letter_idx] > 0).any()

            if has_positive_info:
                # If we have green or yellow info, just mark black positions as impossible
                for pos in positions:
                    state[pos, letter_idx] = -1
                    if letter_idx not in missing_letters[pos]:
                      missing_letters[pos].append(letter_idx)
            else:
                # No positive info, the letter is absent
                for pos in range(word_length):
                    state[pos, letter_idx] = -1
                    if letter_idx not in missing_letters[pos]:
                        missing_letters[pos].append(letter_idx)

        return state.flatten().numpy().tobytes()

    def choose_action(state, attempts_remaining, guessed_actions, current_epsilon):
        """
        Choose an action based on the current state and attempts remaining, 
        using epsilon-greedy strategy.
        """
        allowed_actions = [a for a in range(env.action_space.n) if a not in guessed_actions]

        if not allowed_actions:
            return env.action_space.sample()

        state_time_key = (state, attempts_remaining)
        
        if random.random() < current_epsilon or state_time_key not in Q:
            return random.choice(allowed_actions), True
        else:
            q_values = Q[state_time_key]
            allowed_q = {a: q_values[a] for a in allowed_actions}
            return max(allowed_q, key=allowed_q.get), False

    def update_Q(state, attempts_remaining, action, reward, next_state, next_attempts_remaining, done):
        """
        Update the Q-table using the Q-learning update rule for finite-horizon MDPs.
        The Q-value now depends on both state and time step (attempts remaining).
        """
        state_time_key = (state, attempts_remaining)
        next_state_time_key = (next_state, next_attempts_remaining)
        
        if state_time_key not in Q:
            Q[state_time_key] = {a: 0 for a in range(env.action_space.n)}
        
        if not done and next_state_time_key not in Q:
            Q[next_state_time_key] = {a: 0 for a in range(env.action_space.n)}

        best_next_action_value = max(Q[next_state_time_key].values()) if not done else 0
        Q[state_time_key][action] += alpha * (reward + gamma * best_next_action_value - Q[state_time_key][action])
        
        return Q[state_time_key][action]

    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = get_information_state(observation)
        attempts_remaining = max_attempts
        done = False
        guessed_actions = set()
        
        episode_reward = 0
        episode_step = 0
        exploration_count = 0
        
        current_epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        episode_q_values = []
        
        while not done:
            action, is_exploration = choose_action(state, attempts_remaining, guessed_actions, current_epsilon)
            guessed_actions.add(action)
            if is_exploration:
                exploration_count += 1
            
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = get_information_state(next_observation)
            
            next_attempts_remaining = attempts_remaining - 1
            
            new_q_value = update_Q(state, attempts_remaining, action, reward, next_state, 
                                   next_attempts_remaining, done)
            
            episode_reward += reward
            episode_step += 1
            episode_q_values.append(new_q_value)
            
            state = next_state
            attempts_remaining = next_attempts_remaining

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        win_rate_window.append(1 if env.won else 0)
        exploration_rates.append(exploration_count / episode_step if episode_step > 0 else 0)
        
        avg_q_value = sum(episode_q_values) / len(episode_q_values) if episode_q_values else 0
        q_values_history.append(avg_q_value)
        
        unique_states_count.append(len(Q))
        
        writer.add_scalar('Metrics/Reward', episode_reward, episode)
        writer.add_scalar('Metrics/Episode_Length', episode_step, episode)
        writer.add_scalar('Metrics/Win_Rate', sum(win_rate_window) / len(win_rate_window), episode)
        writer.add_scalar('Metrics/Exploration_Rate', exploration_rates[-1], episode)
        writer.add_scalar('Metrics/Average_Q_Value', avg_q_value, episode)
        writer.add_scalar('Metrics/Unique_States', unique_states_count[-1], episode)
        writer.add_scalar('Hyperparameters/Epsilon', current_epsilon, episode)
        
        if env.won and hasattr(env, 'target_word') and episode % 50 == 0:
            writer.add_text('Examples/Won_Games', 
                           f"Episode {episode}: Solved '{env.target_word}' in {episode_step} attempts", 
                           episode)
        
        if episode % 100 == 0:
            win_rate = sum(win_rate_window) / len(win_rate_window) if win_rate_window else 0
            print(f"Episode {episode}/{num_episodes} | Reward: {episode_reward:.2f} | " 
                  f"Steps: {episode_step} | Win Rate: {win_rate:.2f} | "
                  f"Epsilon: {current_epsilon:.4f} | Q-states: {len(Q)}")
    
    writer.add_histogram('Histograms/Episode_Rewards', np.array(episode_rewards), 0)
    writer.add_histogram('Histograms/Episode_Lengths', np.array(episode_lengths), 0)
    writer.add_histogram('Histograms/Q_Values', np.array(q_values_history), 0)
    
    writer.close()
    
    return Q


def test_agent_finite_horizon(env, Q, num_test_episodes=100, word_length=5, num_letters=26, max_attempts=6, log_dir="./logs/wordle_test"):
    """
    Test the trained finite-horizon agent with TensorBoard logging.

    Args:
        env: The Wordle environment
        Q: The learned Q-table indexed by (state, attempts_remaining)
        num_test_episodes: Number of test episodes to run
        word_length: Length of the target word
        num_letters: Number of possible letters (26 for English alphabet)
        max_attempts: Maximum number of attempts allowed in Wordle (typically 6)
        log_dir: Directory for TensorBoard logs

    Returns:
        results: List of (target_word, num_attempts, success) tuples
    """
    writer = SummaryWriter(log_dir=f"{log_dir}_{time.strftime('%Y%m%d-%H%M%S')}")
    
    def get_information_state(observation):
        """
        Extract a compact information state from the observation, mimicking WordleFeatureExtractor_Markov.
        This version directly mirrors the logic of the `forward` method, adapted for a single environment.
        """

        state = torch.zeros((word_length, num_letters))
        greens = {} 
        yellows = {} 
        blacks = {}
        missing_letters = {pos: [] for pos in range(word_length)}

        attempt_idx = observation['attempt'].item()

        if attempt_idx == 0:
            return "initial" 
        for guess_idx in range(attempt_idx):
            last_feedback = observation['board'][guess_idx]
            last_guess = observation['guesses'][guess_idx]

            if (last_guess < 0).any():
                continue  
            for idx, (feed, letter) in enumerate(zip(last_feedback, last_guess)):
                letter_item = letter.item()
                if feed == 2:
                    if letter_item not in greens:
                        greens[letter_item] = []
                    greens[letter_item].append(idx)

                elif feed == 1:
                    if letter_item not in yellows:
                        yellows[letter_item] = []
                    yellows[letter_item].append(idx)
                elif feed == 0:
                    if letter_item not in blacks:
                        blacks[letter_item] = []
                    blacks[letter_item].append(idx)

        for letter_idx, positions in greens.items():
            for pos in positions:
                state[pos, letter_idx] = 1
                for other_letter in range(num_letters):
                    if other_letter != letter_idx:
                        state[pos, other_letter] = -1

        for letter_idx, positions in yellows.items():

            candidate_positions = [p for p in range(word_length) if p not in positions and p not in greens.get(letter_idx,[])]
            for pos in positions:
                state[pos, letter_idx] = -1
                if letter_idx not in missing_letters[pos]:
                    missing_letters[pos].append(letter_idx)

            if candidate_positions:
                yellow_value = min(1.0, len(positions) / len(candidate_positions))

                for pos in candidate_positions:
                  state[pos, letter_idx] = yellow_value
                  if yellow_value == 1:
                    for other_letter in range(num_letters):
                      if other_letter != letter_idx:
                        state[pos, other_letter] = -1

        for letter_idx, positions in blacks.items():
            has_positive_info = (state[:, letter_idx] > 0).any()

            if has_positive_info:
                for pos in positions:
                    state[pos, letter_idx] = -1
                    if letter_idx not in missing_letters[pos]:
                      missing_letters[pos].append(letter_idx)
            else:
                for pos in range(word_length):
                    state[pos, letter_idx] = -1
                    if letter_idx not in missing_letters[pos]:
                        missing_letters[pos].append(letter_idx)

        return state.flatten().numpy().tobytes()
    
    results = []
    total_reward = 0
    wins = 0
    attempts_distribution = {i: 0 for i in range(1, max_attempts + 1)}
    
    for episode in range(num_test_episodes):
        observation, _ = env.reset()
        target_word = env.target_word
        state = get_information_state(observation)
        attempts_remaining = max_attempts
        done = False
        guessed_actions = set()
        attempts_used = 0
        episode_reward = 0
        
        while not done:
            attempts_used += 1
            allowed_actions = [a for a in range(env.action_space.n) if a not in guessed_actions]
            
            state_time_key = (state, attempts_remaining)
            
            if not allowed_actions:
                attempts_used = max_attempts
                action = env.action_space.sample()
                done = True
            elif state_time_key in Q:
                q_values = Q[state_time_key]
                allowed_q = {a: q_values[a] for a in allowed_actions}
                action = max(allowed_q, key=allowed_q.get)
            else:
                action = random.choice(allowed_actions)

            guessed_actions.add(action)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            next_state = get_information_state(next_observation)
            attempts_remaining -= 1
            state = next_state

        success = env.won
        if success:
            wins += 1
            attempts_distribution[attempts_used] += 1
        
        total_reward += episode_reward
        results.append((target_word, attempts_used, success))
        
        writer.add_scalar('Test/Reward', episode_reward, episode)
        writer.add_scalar('Test/Attempts', attempts_used, episode)
        writer.add_scalar('Test/Success', 1 if success else 0, episode)
        
        if episode % 20 == 0:
            print(f"Tested {episode} / {num_test_episodes} | Win rate: {wins/(episode+1):.2f}")
    
    win_rate = wins / num_test_episodes
    avg_attempts = sum(result[1] for result in results) / num_test_episodes
    avg_reward = total_reward / num_test_episodes
    
    writer.add_scalar('Test/Overall_Win_Rate', win_rate, 0)
    writer.add_scalar('Test/Average_Attempts', avg_attempts, 0)
    writer.add_scalar('Test/Average_Reward', avg_reward, 0)
    
    for attempts, count in attempts_distribution.items():
        if count > 0:
            writer.add_scalar('Test/Success_Distribution', count / wins, attempts)
    
    for i, (word, attempts, success) in enumerate(results[:10]):
        writer.add_text('Test/Examples', 
                       f"Word: {word} | Attempts: {attempts} | Success: {success}", 
                       i)
    
    writer.close()
    return results


# In[31]:


env = WordleEnvMarkov("target_words.txt")
Q = train_q_learning_finite_horizon(env, num_episodes=10000)

results = test_agent_finite_horizon(env, Q, num_test_episodes=100)

total_episodes = len(results)
successful_episodes = sum(1 for _, _, success in results if success)
success_rate = successful_episodes / total_episodes
print(f"Success rate: {success_rate:.4f}")

move_counts = [attempts for _, attempts, success in results if success]
if move_counts:
    average_moves = sum(move_counts) / len(move_counts)
    print(f"Average moves for successful games: {average_moves:.2f}")
else:
    print("No successful games to calculate average moves.")

