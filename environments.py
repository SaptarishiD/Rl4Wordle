import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from collections import defaultdict
import os

class WordleEnv(gym.Env):
    """
    A Gymnasium environment for the Wordle game.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    # Color codes for rendering
    CORRECT = 2      # Green: right letter, right position
    PRESENT = 1      # Yellow: right letter, wrong position
    ABSENT = 0       # Gray: letter not in word
    
    def __init__(self, word_list_path, max_attempts=6, word_length=5, render_mode=None, reward_params=None):
        """
        Initialize the Wordle environment.
        
        Args:
            word_list_path (str): Path to a text file containing words
            max_attempts (int): Maximum number of attempts allowed
            word_length (int): Length of the words
            render_mode (str): How to render the environment
        """
        self.word_length = word_length
        self.max_attempts = max_attempts
        self.render_mode = render_mode

        if reward_params is None:
            reward_params = {'win_bonus': 101.53539925680269, 
                             'attempt_bonus': 1.1337170993046657, 
                             'correct_reward': 105.31846089080993, 
                             'present_reward': 12.107504784037516, 
                             'absent_penalty': -10.144106423123457, 
                             'step_penalty': -1.9278451503776404
                             }
        
        self.reward_params = reward_params
        self.load_words(word_list_path)
        
        self.action_space = spaces.Discrete(len(self.valid_words))
        
        board_shape = (max_attempts, word_length, 3)
        letter_tracker_shape = (26, 3)
        
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=1, shape=board_shape, dtype=np.float32),
            'attempt': spaces.Box(
                        low=0, 
                        high=self.max_attempts, 
                        shape=(1,), 
                        dtype=np.float32
                    ),
            'letter_state': spaces.Box(low=0, high=1, shape=letter_tracker_shape, dtype=np.float32)
        })
        self.reset()
    
    def load_words(self, word_list_path):
        """Load words from file"""
        if os.path.exists(word_list_path):
            with open(word_list_path, 'r') as f:
                self.valid_words = [w.strip().lower() for w in f.readlines() 
                                    if len(w.strip()) == self.word_length and w.strip().isalpha()]
        else:
            print(f"Warning: {word_list_path} not found. Using a small sample of words.")
            self.valid_words = ['apple', 'baker', 'child', 'dance', 'early', 
                                'first', 'grand', 'house', 'input', 'jolly']
        
        if not self.valid_words:
            raise ValueError("No valid words found!")
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new game"""
        super().reset(seed=seed)
        
        self.target_word = random.choice(self.valid_words)
        
        self.board = np.zeros((self.max_attempts, self.word_length, 3), dtype=np.float32)
        self.current_attempt = 0
        self.game_over = False
        self.won = False
        
        self.letter_state = np.zeros((26, 3), dtype=np.float32)

        self.guess_history = []
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): Index of the word to guess from valid_words
        
        Returns:
            observation: The current state of the game
            reward: The reward for the action
            terminated: Whether the game is over
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {'warning': 'Game already over'}
        
        if 0 <= action < len(self.valid_words):
            guessed_word = self.valid_words[action]
        else:
            return self._get_observation(), -1.0, False, False, {'warning': 'Invalid action'}
        
        if len(guessed_word) != self.word_length:
            return self._get_observation(), -1.0, False, False, {'warning': 'Invalid word length'}
        
        self.guess_history.append(guessed_word)
        reward = self._process_guess(guessed_word)
        
        self.current_attempt += 1
        
        if self.won or self.current_attempt >= self.max_attempts:
            self.game_over = True
        
        return self._get_observation(), reward, self.game_over, False, {'guessed_word': guessed_word}
    
    def _process_guess(self, guessed_word):
        """Process a guessed word and update the board state"""
        guess_result = np.zeros((self.word_length, 3), dtype=np.float32)
        
        target_char_count = defaultdict(int)
        for c in self.target_word:
            target_char_count[c] += 1
        
        remaining_chars = target_char_count.copy()
        for i, letter in enumerate(guessed_word):
            if letter == self.target_word[i]:
                guess_result[i, self.CORRECT] = 1
                remaining_chars[letter] -= 1
                
                letter_idx = ord(letter) - ord('a')
                self.letter_state[letter_idx, self.CORRECT] = 1
        
        for i, letter in enumerate(guessed_word):
            if guess_result[i, self.CORRECT] == 1:
                continue
                
            letter_idx = ord(letter) - ord('a')
            if letter in self.target_word and remaining_chars[letter] > 0:
                guess_result[i, self.PRESENT] = 1
                remaining_chars[letter] -= 1
                
                if self.letter_state[letter_idx, self.CORRECT] == 0:
                    self.letter_state[letter_idx, self.PRESENT] = 1
            else:
                guess_result[i, self.ABSENT] = 1
                
                if (self.letter_state[letter_idx, self.CORRECT] == 0 and 
                    self.letter_state[letter_idx, self.PRESENT] == 0):
                    self.letter_state[letter_idx, self.ABSENT] = 1
        
        self.board[self.current_attempt] = guess_result
        
        self.won = guessed_word == self.target_word
        
        if self.won:
            reward = self.reward_params["win_bonus"] + self.reward_params["attempt_bonus"] * (self.max_attempts - self.current_attempt)
        else:
            correct_count = np.sum(guess_result[:, self.CORRECT])
            present_count = np.sum(guess_result[:, self.PRESENT])
            absent_count = np.sum(guess_result[:, self.ABSENT])
            reward = (correct_count * self.reward_params["correct_reward"] + 
                      present_count * self.reward_params["present_reward"] + 
                      absent_count * self.reward_params["absent_penalty"])
            reward += self.reward_params["step_penalty"]
        
        return reward
    
    def _get_observation(self):
        return {
        'board': self.board,
        'attempt': np.array([self.current_attempt], dtype=np.float32),  # shape=(1,)
        'letter_state': self.letter_state,
    }

    
    def render(self):
        """Render the current state of the game"""
        if self.render_mode is None:
            return
        
        rendered = []
        rows = min(self.current_attempt + 1, self.max_attempts)
        for i in range(rows):
            row = ""
            for j in range(self.word_length):
                if self.board[i, j, self.CORRECT] == 1:
                    cell = "🟩"
                elif self.board[i, j, self.PRESENT] == 1:
                    cell = "🟨"
                elif self.board[i, j, self.ABSENT] == 1:
                    cell = "⬛"
                else:
                    cell = "⬜"
                row += cell
            rendered.append(row)
            word = self.guess_history[i] if i < len(self.guess_history) else " " * self.word_length
            word = " ".join(word)
            rendered.append(f"{word}\n{row}\n\n")
        
        if self.render_mode == "human":
            for row in rendered:
                print(row)
            print(f"Attempt {self.current_attempt}/{self.max_attempts}")
            if self.game_over:
                if self.won:
                    print(f"Game won! The word was: {self.target_word}")
                else:
                    print(f"Game over! The word was: {self.target_word}")
        
        return rendered
    
class WordleEnvMarkov(gym.Env):
    """
    A Gymnasium environment for the Wordle game. Models states as purely the observable board
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    # Color codes for rendering
    CORRECT = 2      # Green: right letter, right position
    PRESENT = 1      # Yellow: right letter, wrong position
    ABSENT = 0       # Gray: letter not in word
    
    def __init__(self, word_list_path, max_attempts=6, word_length=5, render_mode=None, reward_params=None):
        """
        Initialize the Wordle environment.
        
        Args:
            word_list_path (str): Path to a text file containing words
            max_attempts (int): Maximum number of attempts allowed
            word_length (int): Length of the words
            render_mode (str): How to render the environment
        """
        self.word_length = word_length
        self.max_attempts = max_attempts
        self.render_mode = render_mode

        if reward_params is None:
            # these values obtained from some hyperparameter optimization through optuna
            
            reward_params = {'win_bonus': 101.53539925680269, 
                             'attempt_bonus': 1.1337170993046657, 
                             'correct_reward': 105.31846089080993, 
                             'present_reward': 12.107504784037516, 
                             'absent_penalty': -10.144106423123457, 
                             'step_penalty': -1.9278451503776404
                             }
        
        self.reward_params = reward_params
        self.load_words(word_list_path)
        
        self.action_space = spaces.Discrete(len(self.valid_words))
        
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=2, shape=(max_attempts, word_length), dtype=np.int32),
            'guesses': spaces.Box(low=-1, high=25, shape=(max_attempts, word_length), dtype=np.int32),
            'attempt': spaces.Box(
                        low=0, 
                        high=self.max_attempts,
                        shape=(1,), 
                        dtype=np.int32
                    )
        })
        self.reset()
        
    def load_words(self, word_list_path):
        """Load words from file"""
        if os.path.exists(word_list_path):
            with open(word_list_path, 'r') as f:
                self.valid_words = [w.strip().lower() for w in f.readlines() 
                                    if len(w.strip()) == self.word_length and w.strip().isalpha()]
        else:
            print(f"Warning: {word_list_path} not found. Using a small sample of words.")
            self.valid_words = ['apple', 'baker', 'child', 'dance', 'early', 
                                'first', 'grand', 'house', 'input', 'jolly']
        
        if not self.valid_words:
            raise ValueError("No valid words found!")
    
    def reset(self, seed=None):
        """Reset the environment for a new game"""
        super().reset(seed=seed)
        
        self.target_word = random.choice(self.valid_words)
        
        self.board = np.zeros((self.max_attempts, self.word_length), dtype=np.float32)
        self.current_attempt = 0
        self.guesses = np.array([[-1 for _ in range(self.word_length)] for _ in range(self.max_attempts)], dtype=np.int32)
        self.game_over = False
        self.won = False
        self.guess_history = []
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): Index of the word to guess from valid_words
        
        Returns:
            observation: The current state of the game
            reward: The reward for the action
            terminated: Whether the game is over
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {'warning': 'Game already over'}
        
        if 0 <= action < len(self.valid_words):
            guessed_word = self.valid_words[action]
        else:
            return self._get_observation(), -1.0, False, False, {'warning': 'Invalid action'}
        
        if len(guessed_word) != self.word_length:
            return self._get_observation(), -1.0, False, False, {'warning': 'Invalid word length'}
        
        self.guess_history.append(guessed_word)
        reward = self._process_guess(guessed_word)
        for x in guessed_word:
            self.guesses[self.current_attempt, :] = ord(x) - ord('a')
        
        self.current_attempt += 1
        
        if self.won or self.current_attempt >= self.max_attempts:
            self.game_over = True
        
        return self._get_observation(), reward, self.game_over, False, {'guessed_word': guessed_word}
    
    def _process_guess(self, guessed_word):
        """Process a guessed word and update the board state"""
        guess_result = [0] * self.word_length

        target_char_count = defaultdict(int)
        for c in self.target_word:
            target_char_count[c] += 1

        remaining_chars = target_char_count.copy()
        for i, letter in enumerate(guessed_word):
            if letter == self.target_word[i]:
                guess_result[i] = self.CORRECT
                remaining_chars[letter] -= 1

        for i, letter in enumerate(guessed_word):
            if guess_result[i] == self.CORRECT:
                continue

            if letter in self.target_word and remaining_chars[letter] > 0:
                guess_result[i] = self.PRESENT
                remaining_chars[letter] -= 1
            else:
                guess_result[i] = self.ABSENT

        self.board[self.current_attempt] = guess_result
        self.won = guessed_word == self.target_word

        if self.won:
            reward = self.reward_params["win_bonus"] + self.reward_params["attempt_bonus"] * (
                        self.max_attempts - self.current_attempt)
        else:
            correct_count = guess_result.count(self.CORRECT)
            present_count = guess_result.count(self.PRESENT)
            absent_count = guess_result.count(self.ABSENT)
            reward = (correct_count * self.reward_params["correct_reward"] +
                      present_count * self.reward_params["present_reward"] +
                      absent_count * self.reward_params["absent_penalty"])
            reward += self.reward_params["step_penalty"]

        return reward
    
    def _get_observation(self):
        return {
        'board': self.board,
        'attempt': np.array([self.current_attempt], dtype=np.int32),  # shape=(1,)
        'guesses': self.guesses,
        }

    
    def render(self):
        """Render the current state of the game"""
        if self.render_mode is None:
            return
        
        rendered = []
        rows = min(self.current_attempt + 1, self.max_attempts)
        for i in range(rows):
            row = ""
            for j in range(self.word_length):
                if self.board[i, j] == self.CORRECT:
                    cell = "🟩"
                elif self.board[i, j] == self.PRESENT:
                    cell = "🟨"
                elif self.board[i, j] == self.ABSENT:
                    cell = "⬛"
                else:
                    cell = "⬜"
                row += cell
            rendered.append(row)
            word = self.guess_history[i] if i < len(self.guess_history) else " " * self.word_length
            word = " ".join(word)
            rendered.append(f"{word}\n{row}\n\n")
        
        if self.render_mode == "human":
            for row in rendered:
                print(row)
            print(f"Attempt {self.current_attempt}/{self.max_attempts}")
            if self.game_over:
                if self.won:
                    print(f"Game won! The word was: {self.target_word}")
                else:
                    print(f"Game over! The word was: {self.target_word}")
        
        return rendered
