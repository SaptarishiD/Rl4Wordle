import pandas as pd
import numpy as np
import gymnasium as gym
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assume target_words.txt exists and contains a list of 5-letter words
# Example target_words.txt content:
# admit
# about
# above
# ...

class WordleQEnv():
    def __init__(self, debug=False, word_list_path='target_words.txt'):
        with open(word_list_path, 'r') as f:
            self.word_list = f.read().splitlines()
        self.attempts = 0
        self.max_attempts = 6
        self.word_len = 5
        self.target_word = random.choice(self.word_list)
        self.guessed_words = []

        self.letters_correct = [] # letters guessed correctly AND in the correct position (greens) - This seems redundant with pos_guessed_correctly
        self.letters_present = [] # unique letters present in the target word but not necessarily in the correct position (yellows)
        self.letters_absent = [] # unique letters absent in the target word (blacks)
        self.pos_guessed_correctly = [None]*self.word_len # keep track of the positions guessed correctly for the whole board (greens)
        self.pos_yellow = defaultdict(list) # letter -> list of positions where it was yellow

        # these 3 are feedback for the current row/guess
        self.row_correct = [None]*self.word_len # greens for the current guess
        self.row_present = [None]*self.word_len # yellows for the current guess
        self.row_absent = [None]*self.word_len  # blacks for the current guess

        self.won_game = ''


    def make_guess(self, word):
        # Ensure word is valid and not guessed before in this environment instance
        if word not in self.word_list:
             # Penalize invalid guess or handle as error? For RL, maybe a penalty.
             # Or just force re-guessing. Let's assume valid words for now based on agent logic.
             pass # Agent should choose from word_list anyway

        if word in self.guessed_words:
            # Penalize repeated guess? Or just process as usual.
            # Let's just process as usual, the agent will learn not to repeat words if penalized via reward.
            pass

        self.attempts += 1

        # Reset row feedback for the new guess
        self.row_correct = [None]*self.word_len
        self.row_present = [None]*self.word_len
        self.row_absent = [None]*self.word_len

        self.guessed_words.append(word)

        # Count occurrences of letters in the target word to handle duplicates correctly
        target_letter_counts = defaultdict(int)
        for letter in self.target_word:
            target_letter_counts[letter] += 1

        # First pass: find greens
        feedback = [None] * self.word_len
        for i, (guessed_letter, target_letter) in enumerate(zip(word, self.target_word)):
            if guessed_letter == target_letter:
                self.row_correct[i] = guessed_letter
                self.pos_guessed_correctly[i] = guessed_letter
                feedback[i] = 'green'
                # Decrement count for matched green letter
                target_letter_counts[guessed_letter] -= 1

        # Second pass: find yellows and blacks
        for i, guessed_letter in enumerate(word):
            if feedback[i] is None: # Not already marked green
                if guessed_letter in target_letter_counts and target_letter_counts[guessed_letter] > 0:
                    self.row_present[i] = guessed_letter
                    # Update global letters_present if new yellow letter is found
                    if guessed_letter not in self.letters_present:
                         self.letters_present.append(guessed_letter)
                    # Store position where it was yellow
                    if i not in self.pos_yellow[guessed_letter]:
                        self.pos_yellow[guessed_letter].append(i)
                    feedback[i] = 'yellow'
                    # Decrement count for matched yellow letter
                    target_letter_counts[guessed_letter] -= 1
                else:
                    self.row_absent[i] = guessed_letter
                    # Update global letters_absent if new black letter is found
                    if guessed_letter not in self.letters_absent:
                        self.letters_absent.append(guessed_letter)
                    feedback[i] = 'black'

        # Update global state lists (letters_correct is redundant with pos_guessed_correctly)
        # self.letters_correct = [letter for letter in self.pos_guessed_correctly if letter is not None] # Should be based on unique letters from green positions

        number_of_greens = sum(1 for x in self.row_correct if x is not None)
        number_of_yellows = sum(1 for x in self.row_present if x is not None)
        number_of_blacks = sum(1 for x in self.row_absent if x is not None)

        # Check for game completion
        if self.target_word == word:
            self.won_game = 'yes'
            # print(f"YOU WON! In {self.attempts} moves")
            # Game ends immediately upon winning
            # Return final feedback for this guess
            return number_of_greens, number_of_yellows, number_of_blacks

        if self.attempts >= self.max_attempts: # Use >= in case attempts exceeds max_attempts in some logic flow
            if self.target_word != word: # Only lose if target word wasn't guessed on the last attempt
                 self.won_game = 'no'
                 # print("ATTEMPTS FINISHED!")
            else: # This case is theoretically covered by the check above, but good for robustness
                 self.won_game = 'yes'

            # Return final feedback for the last guess
            return number_of_greens, number_of_yellows, number_of_blacks


        # If game is not over, return feedback for the current guess
        return number_of_greens, number_of_yellows, number_of_blacks




class MyAgent():
    def __init__(self, debug=False, word_list_path='target_words.txt'):
        self.agent_guesses = [] # Keep track of words already guessed by the agent *across all games*
        self.debug = debug
        with open(word_list_path, 'r') as f:
            self.word_list = f.read().splitlines()

    def _filter_by_absent(self, candidates, absent_letters):
         # Keep words that DO NOT contain any of the absent letters
        return [word for word in candidates if not any(letter in word for letter in absent_letters)]

    def _filter_by_greens(self, candidates, green_positions):
         # Keep words that match the green letters at the specified positions
        filtered = []
        for word in candidates:
            match = True
            for i, letter in enumerate(green_positions):
                if letter is not None and word[i] != letter:
                    match = False
                    break
            if match:
                filtered.append(word)
        return filtered

    def _filter_by_present(self, candidates, present_letters):
         # Keep words that contain ALL of the required present letters (yellows)
         # This is a necessary but not sufficient check, as it doesn't consider positions
        if not present_letters:
            return candidates
        return [word for word in candidates if all(yellow in word for yellow in present_letters)]

    def _filter_by_yellow_positions(self, candidates, yellow_positions):
        # Keep words where yellow letters are NOT in the positions they were marked yellow
        filtered = []
        for word in candidates:
            valid = True
            for letter, positions in yellow_positions.items():
                for pos in positions:
                    if pos < len(word) and word[pos] == letter:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                filtered.append(word)
        return filtered


    def randomly(self):
        # Choose a random word from the full list that hasn't been guessed by the agent
        available_words = [word for word in self.word_list if word not in self.agent_guesses]
        if not available_words:
            # If all words have been guessed, just pick one randomly
            return random.choice(self.word_list)
        return random.choice(available_words)

    def rand_not_absent(self, absent_letters):
        # Choose a random word that doesn't contain any of the absent letters
        candidates = self._filter_by_absent(self.word_list, absent_letters)
        available_words = [word for word in candidates if word not in self.agent_guesses]
        if not available_words:
             # Fallback: try candidates regardless of previous guesses
            if candidates:
                return random.choice(candidates)
            # Fallback: random word (may contain absent letters or be a repeat)
            return self.randomly()
        return random.choice(available_words)

    def rand_green_not_absent(self, green_positions, absent_letters):
        # Choose a random word matching greens and not containing absent letters
        candidates = self._filter_by_absent(self.word_list, absent_letters)
        candidates = self._filter_by_greens(candidates, green_positions)
        available_words = [word for word in candidates if word not in self.agent_guesses]

        if not available_words:
             # Fallback: try candidates regardless of previous guesses
            if candidates:
                return random.choice(candidates)
            # Fallback: use less strict filter
            return self.rand_not_absent(absent_letters)

        return random.choice(available_words)

    def letter_frequency_guess(self):
        # Choose the word with the most common letters that hasn't been guessed

        # Define common letters - can be refined based on frequency analysis of wordle lists
        # https://www.wordfinders.com/word-facts/letter-frequency/wordle/
        common_letters = "esiarntolcdupmghbyfvkwzxqj" # Order based on Wordle frequency

        word_scores = []
        available_words = [word for word in self.word_list if word not in self.agent_guesses]
        if not available_words:
             available_words = self.word_list # If all words guessed, score all

        for word in available_words:
            unique_letters = set(word)
            # Score based on presence of common letters
            score = sum(len(common_letters) - common_letters.index(letter) for letter in unique_letters if letter in common_letters)
            # Add uniqueness bonus (e.g., more unique letters is better for early guesses)
            uniqueness_bonus = len(unique_letters) * 0.1 # Small bonus

            word_scores.append((word, score + uniqueness_bonus))

        if not word_scores:
            return self.randomly()

        # Sort by score, descending
        word_scores.sort(key=lambda x: x[1], reverse=True)

        # Return the highest scoring word
        return word_scores[0][0]


    def smart_guess(self, green_positions, yellows, absent_letters):
        # Smart guess combines green, yellow (presence), and absent filtering,
        # then potentially scores remaining candidates (original scoring logic here was flawed)

        candidates = self.word_list[:] # Start with full list

        # Apply absent filter
        candidates = self._filter_by_absent(candidates, absent_letters)
        if not candidates: # If filtering removes all words, something is wrong or state is impossible
             # Fallback: random guess not containing absent letters (might repeat guessed words)
             return self.rand_not_absent(absent_letters)


        # Apply green filter
        candidates = self._filter_by_greens(candidates, green_positions)
        if not candidates: # If filtering removes all words, fallback
             # Fallback: use less strict filter (absent only)
             return self.rand_not_absent(absent_letters)

        # Apply yellow presence filter (just checks if all required yellows are in the word)
        candidates = self._filter_by_present(candidates, yellows)
        # Note: This smart_guess doesn't use yellow positional info, which is less strict but faster

        # If candidates are filtered down to 0 at this point, it suggests the combined constraints
        # from the environments are contradictory or no word matches.
        # This can happen if, for example, env1 has 'A' green at pos 0 and env2 has 'B' green at pos 0.
        # Or if absent/yellow constraints make a green word impossible.
        # If this happens, fall back to a less strict filter.
        if not candidates:
             # Fallback: try greens + absent filter (ignore yellow presence)
             candidates = self._filter_by_greens(self._filter_by_absent(self.word_list, absent_letters), green_positions)
             if not candidates:
                 # Fallback: try absent filter only
                 return self.rand_not_absent(absent_letters)


        # Prioritize un-guessed words first
        available_words = [word for word in candidates if word not in self.agent_guesses]

        if available_words:
            # Score the available candidates - maybe prioritize words with unique letters or common letters
            # The original scoring logic was a bit basic. Let's use a simple uniqueness/commonality score.
            word_scores = []
            common_letters = "esiarntolcdupmghbyfvkwzxqj" # Based on Wordle frequency
            for word in available_words:
                 unique_letters = set(word)
                 score = sum(len(common_letters) - common_letters.index(letter) for letter in unique_letters if letter in common_letters)
                 uniqueness_bonus = len(unique_letters) * 0.5 # Increased bonus for uniqueness
                 word_scores.append((word, score + uniqueness_bonus))

            if word_scores:
                word_scores.sort(key=lambda x: x[1], reverse=True)
                return word_scores[0][0]
            # If scoring somehow resulted in no scores (shouldn't happen if available_words wasn't empty)
            return random.choice(available_words)

        # If no un-guessed words are available within candidates, pick a random candidate (will be a repeat)
        if candidates:
             return random.choice(candidates)

        # Final fallback if all filters and attempts failed
        return self.randomly()


    def yellow_position_tracking(self, green_positions, yellows, absent_letters, yellow_positions):
        # This strategy uses the most complete set of constraints: green, yellow (presence),
        # absent, AND yellow (positional exclusion).

        candidates = self.word_list[:] # Start with full list

        # Apply absent filter
        candidates = self._filter_by_absent(candidates, absent_letters)
        if not candidates:
             return self.rand_not_absent(absent_letters) # Fallback

        # Apply green filter
        candidates = self._filter_by_greens(candidates, green_positions)
        if not candidates:
             return self.rand_not_absent(absent_letters) # Fallback

        # Apply yellow presence filter
        candidates = self._filter_by_present(candidates, yellows)
        # Note: Filtering candidates here first might be more efficient before positional check

        # Apply yellow positional exclusion filter
        candidates = self._filter_by_yellow_positions(candidates, yellow_positions)

        # Check if filtering resulted in zero candidates
        if not candidates:
            # Fallback to smart_guess logic (less strict) if strict yellow positional logic fails
            if self.debug:
                 print("Yellow positional tracking filter returned no candidates, falling back to smart_guess logic.")
            # Re-calculate candidates using smart_guess logic filters
            candidates = self._filter_by_absent(self.word_list, absent_letters)
            candidates = self._filter_by_greens(candidates, green_positions)
            candidates = self._filter_by_present(candidates, yellows) # smart_guess yellow check

            # If this fallback still yields no candidates, fall back further
            if not candidates:
                 return self.rand_not_absent(absent_letters) # Fallback to absent only

            # Otherwise, continue with scoring from the smart_guess candidate list
            if self.debug:
                 print(f"Smart_guess fallback found {len(candidates)} candidates.")


        # Prioritize un-guessed words
        available_words = [word for word in candidates if word not in self.agent_guesses]

        if available_words:
             # Score available candidates - similar scoring as smart_guess
            word_scores = []
            common_letters = "esiarntolcdupmghbyfvkwzxqj" # Based on Wordle frequency
            for word in available_words:
                 unique_letters = set(word)
                 score = sum(len(common_letters) - common_letters.index(letter) for letter in unique_letters if letter in common_letters)
                 uniqueness_bonus = len(unique_letters) * 0.5 # Increased bonus for uniqueness
                 word_scores.append((word, score + uniqueness_bonus))

            if word_scores:
                word_scores.sort(key=lambda x: x[1], reverse=True)
                return word_scores[0][0]
            # Fallback if scoring fails
            return random.choice(available_words)

        # If no un-guessed words, pick a random one from candidates (will be a repeat)
        if candidates:
             return random.choice(candidates)

        # Final fallback
        return self.randomly()


class WordleMetaEnvMulti():
    def __init__(self, debug=False, word_list_path='target_words.txt'):
        self.won_game_reward = 10
        self.lose_cost = -10
        self.green_reward = 5
        self.yellow_reward = 3
        self.black_cost = -1
        self.debug = debug

        self.word_path = word_list_path
        # Agent is shared across environments to maintain combined state knowledge (like guessed_words)
        self.agent = MyAgent(debug=self.debug, word_list_path=self.word_path)

        # Initialize two separate Wordle environments
        self.env1 = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
        self.env2 = WordleQEnv(debug=self.debug, word_list_path=self.word_path)

        # Define the action space
        self.action_space = [0, 1, 2, 3, 4, 5] # Ensure this line is present

    def reset(self):
        # Reset both environments and the agent's state
        self.agent = MyAgent(debug=self.debug, word_list_path=self.word_path) # Reset agent's guessed words
        self.env1 = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
        self.env2 = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
        self.guesses_made = 0

        # Return combined state (greens1, yellows1, blacks1, greens2, yellows2, blacks2) after reset (initial state)
        return (0, 0, 0, 0, 0, 0)

    def step(self, action):
        self.guesses_made += 1

        # --- Collect combined state information from both environments ---
        combined_absent = list(set(self.env1.letters_absent) | set(self.env2.letters_absent))
        combined_present = list(set(self.env1.letters_present) | set(self.env2.letters_present))

        combined_greens = [None] * self.env1.word_len # Assuming env1 and env2 have same word length
        for i in range(self.env1.word_len):
            g1 = self.env1.pos_guessed_correctly[i]
            g2 = self.env2.pos_guessed_correctly[i]
            if g1 is not None:
                combined_greens[i] = g1
            elif g2 is not None:
                combined_greens[i] = g2

        combined_yellow_pos = defaultdict(list)
        all_yellow_letters = set(self.env1.pos_yellow.keys()) | set(self.env2.pos_yellow.keys())
        for letter in all_yellow_letters:
            combined_yellow_pos[letter] = list(set(self.env1.pos_yellow[letter]) | set(self.env2.pos_yellow[letter]))


        # --- Agent selects guess using the action and combined environment information ---
        guess = None
        if action == 0:
            guess = self.agent.randomly()
        elif action == 1:
            guess = self.agent.rand_not_absent(combined_absent)
        elif action == 2:
            guess = self.agent.rand_green_not_absent(combined_greens, combined_absent)
        elif action == 3:
            guess = self.agent.letter_frequency_guess()
        elif action == 4:
            guess = self.agent.smart_guess(green_positions=combined_greens,
                                          yellows=combined_present,
                                          absent_letters=combined_absent)
        elif action == 5:
            guess = self.agent.yellow_position_tracking(yellows=combined_present,
                                                      absent_letters=combined_absent,
                                                      yellow_positions=combined_yellow_pos,
                                                      green_positions=combined_greens)

        # Add the chosen guess to the agent's history (important for strategies that avoid repeats)
        self.agent.agent_guesses.append(guess)

        # --- Apply the same guess to both environments ---
        greens1, yellows1, blacks1 = self.env1.make_guess(guess)
        greens2, yellows2, blacks2 = self.env2.make_guess(guess)

        # --- Calculate reward ---
        reward1 = (self.green_reward * greens1 +
                   self.yellow_reward * yellows1 +
                   self.black_cost * blacks1)

        reward2 = (self.green_reward * greens2 +
                   self.yellow_reward * yellows2 +
                   self.black_cost * blacks2)

        reward = reward1 + reward2

        # --- Determine combined state for Q-learning ---
        state = (greens1, yellows1, blacks1, greens2, yellows2, blacks2)

        # --- Check game completion conditions ---
        game1_complete = self.env1.won_game in ['yes', 'no']
        game2_complete = self.env2.won_game in ['yes', 'no']
        done = game1_complete and game2_complete

        # Add additional rewards/penalties based on win/loss conditions at the end of the episode
        # These bonuses/penalties are added to the reward of the *final* step of the episode
        # (the step where `done` becomes True).
        if game1_complete:
             if self.env1.won_game == 'yes':
                 reward += self.won_game_reward
             elif self.env1.won_game == 'no':
                 reward += self.lose_cost

        if game2_complete:
             if self.env2.won_game == 'yes':
                 reward += self.won_game_reward
             elif self.env2.won_game == 'no':
                 reward += self.lose_cost


        return state, reward, done
    




    

# Add won_game_reward_added flag to WordleQEnv
def patch_wordle_env():
    original_init = WordleQEnv.__init__
    def new_init(self, debug=False, word_list_path='target_words.txt'):
        original_init(self, debug, word_list_path)
        self.won_game_reward_added = False # Flag to ensure final reward/penalty is added only once per game

    WordleQEnv.__init__ = new_init

patch_wordle_env() # Apply the patch


# The Q-Learning training loop and plotting functions remain the same
# as they interact with the WordleMetaEnvMulti through its public interface (reset, step)

def Q_Learning_Multi(num_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm for the WordleMetaEnvMulti environment, which plays two Wordle games simultaneously.

    Parameters:
    -----------
    num_episodes : int
        Number of episodes to train for
    gamma : float
        Discount factor
    alpha : float
        Learning rate
    epsilon : float
        Exploration probability

    Returns:
    --------
    Q : defaultdict
        Q-table mapping states to action values
    stats : tuple
        Statistics about training (episode_lengths, episode_rewards, wins1, wins2, both_wins)
    """
    env = WordleMetaEnvMulti(debug=False, word_list_path='target_words.txt')
    actions_len = len(env.action_space)
    Q = defaultdict(lambda: np.zeros(actions_len))

    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)
    wins1 = np.zeros(num_episodes)  # Tracking wins for game 1
    wins2 = np.zeros(num_episodes)  # Tracking wins for game 2
    both_wins = np.zeros(num_episodes)  # Tracking when both games are won

    for i in tqdm(range(num_episodes)):
        state = env.reset()

        episode_reward = 0
        episode_length = 0

        while True:
            # Epsilon-greedy policy: choose action based on current state
            # Note: The state used for action selection is the state *before* the action is taken,
            # representing the feedback from the *previous* step.
            # In the first step, the state is (0,0,0,0,0,0).
            action_probs = np.ones(actions_len, dtype=float) * (epsilon / actions_len)
            best_action = np.argmax(Q[state])
            action_probs[best_action] = (1.0 - epsilon + (epsilon / actions_len))
            action = np.random.choice(np.arange(actions_len), p=action_probs / np.sum(action_probs))


            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Update Q-value using the Bellman equation
            opt_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][opt_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            # Move to the next state
            state = next_state
            episode_length += 1

            if done:
                episode_lengths[i] = episode_length
                episode_rewards[i] = episode_reward

                # Record win/loss status for both games at the end of the episode
                if env.env1.won_game == 'yes':
                    wins1[i] = 1
                if env.env2.won_game == 'yes':
                    wins2[i] = 1
                if env.env1.won_game == 'yes' and env.env2.won_game == 'yes':
                    both_wins[i] = 1

                # If both games ended because of max attempts, the length is max_attempts (6)
                # The episode_length increments per step, so it will be 6 if they both end on guess 6.
                # If one ends earlier, the other continues. The episode only finishes when BOTH are done.
                # Max possible length is 6 steps. So episode_length will correctly be 6 if both hit max attempts.
                # The length will be determined by the game that takes the longest.
                # The length should be the number of *agent steps* taken. This is `episode_length`.

                break

    return Q, (episode_lengths, episode_rewards, wins1, wins2, both_wins)


def train_and_evaluate_multi(num_episodes=50000, epsilon=0.1, alpha=0.05, gamma=0.5):
    """
    Train the agent using Q-learning and evaluate its performance.

    Parameters:
    -----------
    num_episodes : int
        Number of episodes to train for
    epsilon : float
        Exploration probability
    alpha : float
        Learning rate
    gamma : float
        Discount factor

    Returns:
    --------
    Q : defaultdict
        Q-table mapping states to action values
    stats : tuple
        Statistics about training
    """
    print(f"Training Q-Learning agent for {num_episodes} episodes...")
    Q, training_stats = Q_Learning_Multi(num_episodes=num_episodes, epsilon=epsilon, alpha=alpha, gamma=gamma)
    print("Training finished.")
    return Q, training_stats


def plot_multi_game_statistics(stats, window_size=500):
    """
    Plot statistics from training the dual Wordle game.

    Parameters:
    -----------
    stats : tuple
        Statistics from training (episode_lengths, episode_rewards, wins1, wins2, both_wins)
    window_size : int
        Window size for moving average calculation
    """
    episode_lengths = stats[0]
    episode_rewards = stats[1]
    wins1 = stats[2]
    wins2 = stats[3]
    both_wins = stats[4]

    num_episodes = len(episode_lengths)

    def moving_average(data, window_size):
        if window_size <= 1:
             return data
        if len(data) < window_size:
             return np.convolve(data, np.ones(len(data)) / len(data), mode='valid')
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Calculate moving averages
    ma_lengths = moving_average(episode_lengths, window_size)
    ma_rewards = moving_average(episode_rewards, window_size)
    ma_wins1 = moving_average(wins1, window_size)
    ma_wins2 = moving_average(wins2, window_size)
    ma_both_wins = moving_average(both_wins, window_size)

    # Determine x-axis range for plots
    x_range_len = len(ma_lengths)
    x_range = np.arange(num_episodes - x_range_len, num_episodes)


    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), sharex=True) # Use 2x2 grid

    # Plot game lengths
    ax1.plot(x_range, ma_lengths, 'steelblue', linewidth=2)
    ax1.set_title('Dual Wordle Game Lengths Over Time', fontsize=14)
    ax1.set_ylabel('Number of Guesses', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 7.5) # Max possible length is 6 + 1 for final state


    # Plot rewards
    ax2.plot(x_range, ma_rewards, 'forestgreen', linewidth=2)
    ax2.set_title('Dual Wordle Game Rewards Over Time', fontsize=14)
    ax2.set_ylabel('Total Reward', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot individual win rates
    ax3.plot(x_range, ma_wins1, 'darkorange', linewidth=2, label='Game 1')
    ax3.plot(x_range, ma_wins2, 'purple', linewidth=2, label='Game 2')
    ax3.set_title('Individual Wordle Win Rates Over Time', fontsize=14)
    ax3.set_ylabel('Win Rate', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1.1) # Win rate is 0-1

    # Plot both-win rate
    ax4.plot(x_range, ma_both_wins, 'crimson', linewidth=2)
    ax4.set_title('Both Games Win Rate Over Time', fontsize=14)
    ax4.set_ylabel('Win Rate', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1) # Win rate is 0-1

    plt.xlabel('Episode', fontsize=12) # Add xlabel to the bottom row

    plt.tight_layout()
    plt.show()

    # --- Additional Statistics ---

    # Distribution of game lengths
    plt.figure(figsize=(10, 6))
    # Only consider lengths for episodes where at least one game was won, or both were lost at 6
    valid_lengths_indices = (both_wins == 1) | (wins1 == 1) | (wins2 == 1) | (episode_lengths == 6)
    valid_lengths = episode_lengths[valid_lengths_indices]

    if len(valid_lengths) > 0:
        labels, counts = np.unique(valid_lengths, return_counts=True)
        plt.bar(labels, counts, align='center', color='steelblue')
        plt.xlabel('Game Length (Number of Guesses)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Distribution of Dual Wordle Game Lengths', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.xticks(labels)
        plt.show()
    else:
        print("No valid episode lengths recorded for plotting distribution.")


    # Print statistics
    win_percentage1 = np.mean(wins1) * 100
    win_percentage2 = np.mean(wins2) * 100
    both_win_percentage = np.mean(both_wins) * 100

    print("-" * 30)
    print("Overall Performance:")
    print("-" * 30)
    print(f"Total Episodes: {num_episodes}")
    print(f"Game 1 Win Rate: {win_percentage1:.2f}%")
    print(f"Game 2 Win Rate: {win_percentage2:.2f}%")
    print(f"Both Games Win Rate: {both_win_percentage:.2f}%")
    print(f"Average Game Length: {np.mean(episode_lengths):.2f} steps (until both games finish)")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print("-" * 30)


    # Additional analysis: Win distribution for successful episodes (both games won)
    winning_episode_lengths = episode_lengths[both_wins == 1]
    if len(winning_episode_lengths) > 0:
        plt.figure(figsize=(10, 6))
        win_labels, win_counts = np.unique(winning_episode_lengths, return_counts=True)
        plt.bar(win_labels, win_counts, align='center', color='forestgreen')
        plt.xlabel('Game Length for Dual Wins (Number of Guesses)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Distribution of Game Lengths for Dual Wins', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.xticks(win_labels)
        plt.show()
    else:
        print("No episodes where both games were won to plot distribution.")

    # Comparison of individual game performance
    plt.figure(figsize=(10, 6))
    labels = ['Game 1', 'Game 2', 'Both Games']
    values = [win_percentage1, win_percentage2, both_win_percentage]
    plt.bar(labels, values, color=['darkorange', 'purple', 'crimson'])
    plt.xlabel('Game', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.title('Win Rate Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Make sure target_words.txt exists or create a dummy one for testing
    try:
        with open('target_words.txt', 'r') as f:
            word_list_check = f.read().splitlines()
        if not word_list_check:
             print("target_words.txt is empty. Creating a dummy list.")
             with open('target_words.txt', 'w') as f:
                 f.write("adieu\n")
                 f.write("crane\n")
                 f.write("slate\n")
                 f.write("salet\n")
                 f.write("trace\n")
                 f.write("tests\n")
                 f.write("hello\n")
                 f.write("world\n")
                 f.write("train\n")
                 f.write("plane\n")
    except FileNotFoundError:
        print("target_words.txt not found. Creating a dummy list.")
        with open('target_words.txt', 'w') as f:
            f.write("adieu\n")
            f.write("crane\n")
            f.write("slate\n")
            f.write("salet\n")
            f.write("trace\n")
            f.write("tests\n")
            f.write("hello\n")
            f.write("world\n")
            f.write("train\n")
            f.write("plane\n")


    # Train the Q-learning agent
    # Reduced episodes for faster testing, increase for better training
    Q, training_stats = train_and_evaluate_multi(num_episodes=10000, epsilon=0.1, alpha=0.05, gamma=0.5)

    # Plot the training statistics
    plot_multi_game_statistics(training_stats, window_size=500)

    # Example of how to save the Q-table (optional)
    # import json
    # # Convert defaultdict keys (tuples) to strings for JSON compatibility
    # Q_str_keys = {str(k): v.tolist() for k, v in Q.items()}
    # with open('wordle_multi_q_table.json', 'w') as f:
    #     json.dump(Q_str_keys, f)
    # print("\nQ-table saved to wordle_multi_q_table.json")

    # Example of how to load the Q-table
    # try:
    #     with open('wordle_multi_q_table.json', 'r') as f:
    #         Q_str_keys_loaded = json.load(f)
    #     # Convert string keys back to tuples and list values back to numpy arrays
    #     Q_loaded = defaultdict(lambda: np.zeros(len(WordleMetaEnvMulti().action_space)))
    #     for k_str, v_list in Q_str_keys_loaded.items():
    #         # Convert string tuple back to tuple of ints
    #         k_tuple = tuple(map(int, k_str.strip('()').split(',')))
    #         Q_loaded[k_tuple] = np.array(v_list)
    #     print("\nQ-table loaded.")
    #     # You can now use Q_loaded for evaluation
    # except FileNotFoundError:
    #     print("\nNo saved Q-table found.")