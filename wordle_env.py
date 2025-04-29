"""
Outline of environment:
    The hidden word is known to the environment,  generated from uniform distribution of valid english words of 5 letters
    The user should be able to input any valid english word (can be changed later to any word)
    Then the input word is checked against the hidden word, and feedback signals of green yellow and grey are given for each letter and position, showing that the letter and position were correct, correct letter wrong position and wrong letter
    The user can do this 6 times and at the end if the user is able to guess the word correctly then reward of +1 otherwise reward of -1
    MDP M = (A, S, R, P, gamma)
        A: Input Words
        S: State of the Wordle Grid with greens and yellows and guessed words
        R: 
        P: ?

Outline of agent:
    The agent should broadly try to learn either of the following strategies:
        1. Guess words that maximize information gain (e.g., minimize entropy of the remaining word list) by guessing unique letter words like flame, brick that draw from information theoretic approaches
        2. Build on previous guesses if some letters were correct and keep those correct green spaces fixed and try to guess the remaining letters
        3. Other (have non RL baseline)
"""
import gymnasium as gym
import numpy as np
import random

# -----------------------------
# Wordle Environment Definition
# -----------------------------
class WordleEnv_Letter_Based_Feedback(gym.Env):
    """
    
    Observation:
      - "attempt": current attempt number (0 to max_attempts)
      - "feedback": a numpy array of length 27, first 26 correspond to letters with last corresponding to coherence
          (each element is: -1 for not present, 0 for not guessed, +1 for each known yellow, 2 for green)
          
    Action: Index of currently guessed word in the word list.
    
    Reward:
      +1 for a correct guess and ending episode
      -1 if max attempts are reached without a correct guess,
      0 otherwise.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, word_list_file=None, max_attempts=6):
        super().__init__()

        with open(word_list_file, "r") as f:
            word_list = f.read().splitlines()
        self.word_list = word_list if word_list is not None else [
            "apple", "berry", "cider", "delta", "eagle", "fuzzy", "gamer"
        ]
        # self.word_list = random.sample(self.word_list, 1000)
        self.max_attempts = max_attempts
        self.word_length = 5
        
        self.action_space = gym.spaces.Discrete(len(self.word_list)) # actions represent indices of words in the word list
        
        # observations represent the current attempt number and feedback for the current guess
        self.observation_space = gym.spaces.Dict({
            "attempt": gym.spaces.Discrete(self.max_attempts + 1),      
            "feedback": gym.spaces.Box(low=-2, high=10, shape=(26,), dtype=np.int8) # can get feedback of -1, 0, 1, 2 representing miss, empty, yellow, green for each of the 5 letters represented by spaces.Box with low=-1, high=2
        })
        self.guessed_words = []
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.attempt = 0
        self.target = random.choice(self.word_list)
        self.last_feedback = tuple([0 for _ in range (27)])
        observation = {"attempt": self.attempt, "feedback": np.array(self.last_feedback, dtype=np.int8)}
        return observation, {}
    
    def step(self, action):
        guess = self.word_list[action]
        self.guessed_words.append(guess)
        feedback = self.get_feedback(guess)
        self.attempt += 1
        self.last_feedback = feedback
        observation = {"attempt": self.attempt, "feedback": np.array(feedback, dtype=np.int8)}
        done = False
        reward = 0
        
        if guess == self.target:
            reward = 1
            done = True
        elif self.attempt >= self.max_attempts:
            reward = -1
            done = True
        return observation, reward, done, False, {}
    
    def get_feedback(self, guess):
        feedback = [x for x in self.last_feedback]
        correct_guesses = 0 # Rewarding concurrence of stringing good guesses together
        # Count occurrences of each letter in target
        letter_counts = {}
        for char in self.target:
            letter_counts[char] = letter_counts.get(char, 0) + 1
        
        # First pass: Mark exact matches (10) and decrement available counts
        for idx, char in enumerate(guess):
            if char == self.target[idx]:
                correct_guesses += 4
                feedback[ord(char) - ord('a')] = 10
                letter_counts[char] -= 1
        
        # Second pass: Mark partial matches (1+) or misses (-1)
        for idx, char in enumerate(guess):
            if char != self.target[idx]:  # Skip exact matches
                char_idx = ord(char) - ord('a')
                if char in letter_counts and letter_counts[char] > 0:
                    # Count how many times this letter has appeared in non-exact positions
                    occurrences = sum(1 for i in range(idx) 
                                    if guess[i] == char and guess[i] != self.target[i])
                    
                    correct_guesses += 1
                    feedback[char_idx] = 1 + occurrences
                    letter_counts[char] -= 1
                else:
                    feedback[char_idx] = -2
        feedback[26] = correct_guesses
        return tuple(feedback)

    
    def render(self):
        print("\nFinal state:")
        print(f"Total Attempts: {self.attempt}")
        print(f"Feedback of last guess: {self.last_feedback}")
        print(f"Target word was: {self.target}")


if __name__ == "__main__":
    env = WordleEnv_Letter_Based_Feedback(word_list_file="target_words.txt")
    observation, _ = env.reset()
    done = False

    print("You have 6 attempts to guess the correct word.\n")
    
    while not done:
        user_guess = input("Enter your guess: ").strip().lower()
        
        if user_guess not in env.word_list:
            print("Invalid word!")
            continue
        
        action = env.word_list.index(user_guess)
        observation, reward, done, _, _ = env.step(action)
        
        print("Feedback (0=gray, 1=yellow, 2=green):", observation["feedback"])
        
        if reward > 0:
            print("\nCongratulations! You've guessed the correct word!")
        elif reward < 0:
            print("\nGame over! You've used all your attempts.")
            print("The correct word was:", env.target)
    
    env.render()
