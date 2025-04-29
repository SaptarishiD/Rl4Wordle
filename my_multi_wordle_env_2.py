import pandas as pd
import numpy as np
import gymnasium as gym
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm




class WordleQEnv():
    def __init__(self, debug=False, word_list_path='target_words.txt'):
        with open(word_list_path, 'r') as f:
            self.word_list = f.read().splitlines()
        self.attempts = 0
        self.max_attempts = 10
        self.word_len = 5
        self.target_word = random.choice(self.word_list)
        self.guessed_words = []

        self.letters_correct = [] # keep track of the letters guessed correctly
        self.letters_present = [] # keep track of the letters present in the target word from the guesses i.e. yellows
        self.letters_absent = [] # keep track of the letters absent in the target word from the guesses
        self.pos_guessed_correctly = [None]*self.word_len # keep track of the positions guessed correctly for the whole board
        self.pos_yellow = defaultdict(list)

        # these 3 are for each row
        self.row_correct = [None]*self.word_len # greens
        self.row_present = [None]*self.word_len # yellows
        self.row_absent = [None]*self.word_len  # blacks

        self.won_game = ''


    def make_guess(self, word):
        self.attempts += 1
        # if word == self.target_word:
            # print("\n ============= You Won =========== \n")

        self.row_correct = [None]*self.word_len 
        self.row_present = [None]*self.word_len # yellows
        self.row_absent = [None]*self.word_len  # blacks

        self.guessed_words.append(word)

        for i, (guessed_letter, target_letter) in enumerate(zip(word, self.target_word)):
            # green
            if guessed_letter == target_letter:
                self.row_correct[i] = target_letter
                self.letters_correct.append(target_letter)
                # self.letters_present.append(target_letter)
                self.pos_guessed_correctly[i] = target_letter

            # yellow
            elif guessed_letter in self.target_word and guessed_letter != target_letter:
                self.row_present[i] = guessed_letter
                if i not in self.pos_yellow[guessed_letter]:
                    self.pos_yellow[guessed_letter].append(i) # this is a dict where vals are list
                if guessed_letter not in self.letters_present:
                    self.letters_present.append(guessed_letter)
            
            else:
                self.row_absent[i] = guessed_letter
                if guessed_letter not in self.letters_absent:
                    self.letters_absent.append(guessed_letter)

        number_of_greens = len([x for x in self.row_correct if x is not None])
        number_of_yellows = len([x for x in self.row_present if x is not None])
        number_of_blacks = len([x for x in self.row_absent if x is not None])

        if self.target_word == word:
            self.won_game = 'yes'
            # print(f"YOU WON! In {self.attempts} moves")
            return number_of_greens, number_of_yellows, number_of_blacks
        
       

        if self.attempts == self.max_attempts:
            self.won_game = 'no'
            self.attempts = 11
            # print("ATTEMPTS FINISHED!")
            return number_of_greens, number_of_yellows, number_of_blacks

        return number_of_greens, number_of_yellows, number_of_blacks
    




class MyAgent():
    def __init__(self, debug=False, word_list_path='target_words.txt'):
        self.agent_guesses = []
        self.agent_attempts = 0
        self.debug = debug
        with open(word_list_path, 'r') as f:
            self.word_list = f.read().splitlines()
        
    def randomly(self):
        rand_word = random.choice(self.word_list)
        while (rand_word in self.agent_guesses):
            rand_word = random.choice(self.word_list)
        return rand_word

    def rand_not_absent(self, absent_letters):
        new_search_space = [word for word in self.word_list if not any(letter in word for letter in absent_letters)]
        if len(new_search_space) < 1:
            return self.randomly()
        rand_word = random.choice(new_search_space)
        while (rand_word in self.agent_guesses):
            rand_word = random.choice(new_search_space)
        return rand_word
    
    def rand_green_not_absent(self, green_positions, absent_letters):
        filtered = []
        new_search_space = [word for word in self.word_list if not any(letter in word for letter in absent_letters)]
        for word in new_search_space:
            match = True
            for i, letter in enumerate(green_positions):
                if letter is not None and word[i] != letter:
                    match = False
                    break
            if match:
                filtered.append(word)
        # print(filtered)
        # print(len(filtered))
        if self.debug:
            print("Filtered list in rand green not absent")
            print(len(filtered))
        if len(filtered) < 1:
            return self.rand_not_absent(absent_letters)
        return random.choice(filtered)
    

    def smart_guess(self, green_positions, yellows, absent_letters):

        # remove words with absent letters
        candidates = [word for word in self.word_list if not any(letter in word for letter in absent_letters)]
        
        # get the words with green pos
        filtered = []
        for word in candidates:
            match = True
            for i, letter in enumerate(green_positions):
                if letter is not None and word[i] != letter:
                    match = False
                    break
            if match:
                filtered.append(word)
        
        if not filtered:
            if candidates:
                return random.choice(candidates)
            return self.randomly()
        
        candidates = filtered
        
        # now check for yellows
        if yellows:
            yellow_filtered = []
            for word in candidates:
                if all(yellow in word for yellow in yellows):
                    valid = True
                    for i, letter in enumerate(green_positions):
                        if letter is None and i < len(word) and word[i] in yellows:
                            # this pos has a yellow letter so shouldnt be in our search space
                            valid = False
                            break
                    if valid:
                        yellow_filtered.append(word)
            
            if yellow_filtered:
                candidates = yellow_filtered
        
        if len(candidates) > 1:

            word_scores = []
            for word in candidates:

                if word in self.agent_guesses:
                    continue
                    
                unique_letters = len(set(word))
            
            if not word_scores:
                return random.choice(candidates)
            
            # we want to prioritize words with unique letters so that search space is reduced more
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            return word_scores[0][0]
        

        candidates = [word for word in candidates if word not in self.agent_guesses]
        if not candidates:
            candidates = filtered
        
        return random.choice(candidates)

    def letter_frequency_guess(self):


        common_letters = "etaoins"
        
        word_scores = []
        for word in self.word_list:
            if word in self.agent_guesses:
                continue
                
            unique_letters = set(word)
            score = sum(1 for letter in unique_letters if letter in common_letters)
            
            uniqueness_bonus = len(unique_letters)
            
            word_scores.append((word, score + 0.1 * uniqueness_bonus))
        
        if not word_scores:
            return self.randomly()
        
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores[0][0]


    def yellow_position_tracking(self, green_positions, yellows, absent_letters, yellow_positions):
        # this additionally uses information about the positions of yellows. This has a stricter, correct interpretation of yellow feedback

        candidates = [word for word in self.word_list if not any(letter in word for letter in absent_letters)]
        
        filtered = []
        for word in candidates:
            match = True
            for i, letter in enumerate(green_positions):
                if letter is not None and word[i] != letter:
                    match = False
                    break
            if match:
                filtered.append(word)
        
        if not filtered:
            if candidates:
                return random.choice(candidates)
            return self.randomly()
        
        candidates = filtered
        

        if yellows:
            yellow_filtered = []
            for word in candidates:

                if all(yellow in word for yellow in yellows):
                    valid = True
                    
                    # check that yellows aren't in locations where we've already guessed that letter
                    for letter, positions in yellow_positions.items():
                        for pos in positions:
                            if pos < len(word) and word[pos] == letter:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        yellow_filtered.append(word)
            
            if yellow_filtered:
                candidates = yellow_filtered
        
        if len(candidates) > 1:
            word_scores = []
            for word in candidates:
                novelty = 0 if word in self.agent_guesses else 2
                unique_letters = len(set(word))
                
                word_scores.append((word, unique_letters + novelty))
            
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            if word_scores:
                return word_scores[0][0]
        
        candidates = [word for word in candidates if word not in self.agent_guesses]
        if not candidates:
            candidates = filtered
        
        return random.choice(candidates)
         


class WordleMetaEnv():
    def __init__(self, debug=False, word_list_path='target_words.txt'):
        self.won_game_reward = 10
        self.lose_cost = -10
        self.green_reward = 5
        self.yellow_reward = 3
        self.black_cost = -1
        self.debug = debug

        self.total_reward = 0
        self.action_space = [0,1,2,3,4,5]

        self.word_path = word_list_path
        self.agent = MyAgent(debug=self.debug, word_list_path=self.word_path)
        self.env = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
    
    def reset(self):
        self.agent = MyAgent(debug=self.debug, word_list_path=self.word_path)
        self.env = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
        self.guesses_made = 0

        return (0,0,0)  # corresponding to greens and yellows and blacks
    
    def step(self, action):
        self.guesses_made += 1

        if action == 0:
            guess = self.agent.randomly()
        elif action == 1:
            guess = self.agent.rand_not_absent(self.env.letters_absent)
        elif action == 2:
            guess = self.agent.rand_green_not_absent(self.env.pos_guessed_correctly, self.env.letters_absent)
        elif action == 3:
            guess = self.agent.letter_frequency_guess()
        elif action == 4:
            guess = self.agent.smart_guess(green_positions=self.env.pos_guessed_correctly, yellows=self.env.letters_present, absent_letters=self.env.letters_absent)
        elif action == 5:
            guess = self.agent.yellow_position_tracking(yellows=self.env.letters_present, absent_letters=self.env.letters_absent,yellow_positions=self.env.pos_yellow, green_positions=self.env.pos_guessed_correctly)
        
        greens, yellows, blacks = self.env.make_guess(guess)
        reward = 0
        reward = self.green_reward * greens + self.yellow_reward * yellows + self.black_cost * blacks

        state = (greens, yellows, blacks)
        
        
        if self.env.won_game == 'yes':
            reward += self.won_game_reward
            return state, reward, True
        elif self.env.won_game == 'no':
            reward += self.lose_cost
            return state, reward, True

        return state, reward, False
    


# what if I just pass the word guessed to the wrapped 4 q wordle environments and flatten everything and get the reward waise




class WordleMetaEnvMulti():
    def __init__(self, debug=False, word_list_path='target_words.txt'):
        self.won_game_reward = 10
        self.lose_cost = -10
        self.green_reward = 5
        self.yellow_reward = 3
        self.black_cost = -1
        self.debug = debug

        self.total_reward = 0
        self.action_space = [0, 1, 2, 3, 4, 5]

        self.word_path = word_list_path
        self.agent = MyAgent(debug=self.debug, word_list_path=self.word_path)
        
        # Initialize two separate Wordle environments
        self.env1 = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
        self.env2 = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
    
    def reset(self):
        self.agent = MyAgent(debug=self.debug, word_list_path=self.word_path)
        self.env1 = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
        self.env2 = WordleQEnv(debug=self.debug, word_list_path=self.word_path)
        self.guesses_made = 0

        # Return combined state (greens1, yellows1, blacks1, greens2, yellows2, blacks2)
        return (0, 0, 0, 0, 0, 0)
    
    def step(self, action):
        self.guesses_made += 1

        # Use the agent to select a guess based on the action and first environment (#TODO change this)
        # if action == 0:
        #     guess = self.agent.randomly()
        # elif action == 1:
        #     guess = self.agent.rand_not_absent(self.env1.letters_absent)
        # elif action == 2:
        #     guess = self.agent.rand_green_not_absent(self.env1.pos_guessed_correctly, self.env1.letters_absent)
        # elif action == 3:
        #     guess = self.agent.letter_frequency_guess()
        # elif action == 4:
        #     guess = self.agent.smart_guess(green_positions=self.env1.pos_guessed_correctly, 
        #                                   yellows=self.env1.letters_present, 
        #                                   absent_letters=self.env1.letters_absent)
        # elif action == 5:
        #     guess = self.agent.yellow_position_tracking(yellows=self.env1.letters_present, 
        #                                               absent_letters=self.env1.letters_absent,
        #                                               yellow_positions=self.env1.pos_yellow, 
        #                                               green_positions=self.env1.pos_guessed_correctly)

        if action == 0:
            guess = self.agent.randomly()
        elif action == 1:
            chosen_env = random.choice([self.env1, self.env2])
            guess = self.agent.rand_not_absent(chosen_env.letters_absent)
        elif action == 2:
            chosen_env = random.choice([self.env1, self.env2])
            guess = self.agent.rand_green_not_absent(chosen_env.pos_guessed_correctly, chosen_env.letters_absent)
        elif action == 3:
            guess = self.agent.letter_frequency_guess()
        elif action == 4:
            chosen_env = random.choice([self.env1, self.env2])
            guess = self.agent.smart_guess(
                green_positions=chosen_env.pos_guessed_correctly,
                yellows=chosen_env.letters_present,
                absent_letters=chosen_env.letters_absent
            )
        elif action == 5:
            chosen_env = random.choice([self.env1, self.env2])
            guess = self.agent.yellow_position_tracking(
                yellows=chosen_env.letters_present,
                absent_letters=chosen_env.letters_absent,
                yellow_positions=chosen_env.pos_yellow,
                green_positions=chosen_env.pos_guessed_correctly
            )
                

        # Apply the same guess to both environments
        greens1, yellows1, blacks1 = self.env1.make_guess(guess)
        greens2, yellows2, blacks2 = self.env2.make_guess(guess)
        
        # Calculate reward for both environments
        reward1 = self.green_reward * greens1 + self.yellow_reward * yellows1 + self.black_cost * blacks1
        reward2 = self.green_reward * greens2 + self.yellow_reward * yellows2 + self.black_cost * blacks2
        
        # Combined state from both environments
        state = (greens1, yellows1, blacks1, greens2, yellows2, blacks2)
        
        # Calculate total reward
        reward = reward1 + reward2
        
        # Check game completion conditions
        game1_complete = self.env1.won_game in ['yes', 'no']
        game2_complete = self.env2.won_game in ['yes', 'no']
        done = game1_complete and game2_complete
        
        # Add additional rewards/penalties based on win/loss conditions
        if self.env1.won_game == 'yes':
            reward += self.won_game_reward
        elif self.env1.won_game == 'no':
            reward += self.lose_cost
            
        if self.env2.won_game == 'yes':
            reward += self.won_game_reward
        elif self.env2.won_game == 'no':
            reward += self.lose_cost

        return state, reward, done
    


f = open('target_words.txt', 'r')
target_words = f.read().splitlines()
f.close()

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

        # Epsilon-greedy policy
        action_probs = np.ones(actions_len, dtype=float) * (epsilon / actions_len)
        action_prob_len = len(action_probs)
        best_action = np.argmax(Q[state])
        action_probs[best_action] = (1.0 - epsilon + (epsilon / actions_len))
        
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Choose action according to probability distribution
            action = np.random.choice(np.arange(action_prob_len), p=action_probs / np.sum(action_probs))
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Update Q-value using the Bellman equation
            opt_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][opt_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            if done:
                episode_lengths[i] = episode_length + 1
                episode_rewards[i] = episode_reward
                
                # Record win/loss status for both games
                if env.env1.won_game == 'yes':
                    wins1[i] = 1
                if env.env2.won_game == 'yes':
                    wins2[i] = 1
                if env.env1.won_game == 'yes' and env.env2.won_game == 'yes':
                    both_wins[i] = 1
                    
                # If both games ended because of max attempts, set length to 7
                if env.env1.won_game == 'no' and env.env2.won_game == 'no':
                    episode_lengths[i] = 7
                    
                break
                
            state = next_state
            episode_length += 1

            # Update action probabilities for next iteration
            action_probs = np.ones(actions_len, dtype=float) * (epsilon / actions_len)
            best_action = np.argmax(Q[state])
            action_probs[best_action] = (1.0 - epsilon + (epsilon / actions_len))
            
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
    return Q_Learning_Multi(num_episodes=num_episodes, epsilon=epsilon, alpha=alpha, gamma=gamma)


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
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    if num_episodes > window_size:
        ma_lengths = moving_average(episode_lengths, window_size)
        ma_rewards = moving_average(episode_rewards, window_size)
        ma_wins1 = moving_average(wins1, window_size)
        ma_wins2 = moving_average(wins2, window_size)
        ma_both_wins = moving_average(both_wins, window_size)
        x_range = np.arange(window_size-1, num_episodes)
    else:
        ma_lengths = episode_lengths
        ma_rewards = episode_rewards
        ma_wins1 = wins1
        ma_wins2 = wins2
        ma_both_wins = both_wins
        x_range = np.arange(num_episodes)
    
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
    
    # Plot game lengths
    ax1.plot(x_range, ma_lengths, 'steelblue', linewidth=2)
    ax1.set_title('Dual Wordle Game Lengths Over Time', fontsize=16)
    ax1.set_ylabel('Number of Guesses', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot rewards
    ax2.plot(x_range, ma_rewards, 'forestgreen', linewidth=2)
    ax2.set_title('Dual Wordle Game Rewards Over Time', fontsize=16)
    ax2.set_ylabel('Total Reward', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot individual win rates
    ax3.plot(x_range, ma_wins1, 'darkorange', linewidth=2, label='Game 1')
    ax3.plot(x_range, ma_wins2, 'purple', linewidth=2, label='Game 2')
    ax3.set_title('Individual Wordle Win Rates Over Time', fontsize=16)
    ax3.set_ylabel('Win Rate', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot both-win rate
    ax4.plot(x_range, ma_both_wins, 'crimson', linewidth=2)
    ax4.set_title('Both Games Win Rate Over Time', fontsize=16)
    ax4.set_ylabel('Win Rate', fontsize=14)
    ax4.set_xlabel('Episode', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Distribution of game lengths
    plt.figure(figsize=(10, 6))
    labels, counts = np.unique(episode_lengths, return_counts=True)
    plt.bar(labels, counts, align='center', color='steelblue')
    plt.xlabel('Game Length (Number of Guesses)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Distribution of Dual Wordle Game Lengths', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(labels)
    plt.show()
    
    # Print statistics
    win_percentage1 = np.mean(wins1) * 100
    win_percentage2 = np.mean(wins2) * 100
    both_win_percentage = np.mean(both_wins) * 100
    
    print(f"Game 1 Win Rate: {win_percentage1:.2f}%")
    print(f"Game 2 Win Rate: {win_percentage2:.2f}%")
    print(f"Both Games Win Rate: {both_win_percentage:.2f}%")
    print(f"Average Game Length: {np.mean(episode_lengths):.2f} guesses")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    
    # Additional analysis: Win distribution for successful episodes
    if np.sum(both_wins) > 0:
        win_lengths = episode_lengths[both_wins == 1]
        plt.figure(figsize=(10, 6))
        win_labels, win_counts = np.unique(win_lengths, return_counts=True)
        plt.bar(win_labels, win_counts, align='center', color='forestgreen')
        plt.xlabel('Game Length for Dual Wins (Number of Guesses)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Distribution of Game Lengths for Dual Wins', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.xticks(win_labels)
        plt.show()
    
    # Comparison of individual game performance
    plt.figure(figsize=(10, 6))
    labels = ['Game 1', 'Game 2', 'Both Games']
    values = [win_percentage1, win_percentage2, both_win_percentage]
    plt.bar(labels, values, color=['darkorange', 'purple', 'crimson'])
    plt.xlabel('Game', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.title('Win Rate Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Train the Q-learning agent
    Q, training_stats = train_and_evaluate_multi(num_episodes=50000, epsilon=0.1, alpha=0.05, gamma=0.5)
    
    # Plot the training statistics
    plot_multi_game_statistics(training_stats)
    
    # Save the Q-table for future use
    # np.save('wordle_multi_q_table.npy', dict(Q))