from collections import defaultdict
from utils import filter_words
class HeuristicWordleAgent:
    """
    A heuristic-based agent for playing Wordle purely based on exploration.
    Tries to maximise number of unique letters tried.
    """
    
    def __init__(self, word_list):
        """
        Initialize the agent with a list of possible words.
        
        Args:
            word_list (list): List of valid words
        """
        self.all_words = word_list
        self.possible_solutions = word_list.copy()
        self.word_length = len(word_list[0]) if word_list else 5
        self.attempts = 0
        self.history = []
    
    def reset(self):
        """Reset the agent for a new game"""
        self.possible_solutions = self.all_words.copy()
        self.attempts = 0
        self.history = []
    
    def get_action(self, observation=None):
        """
        Determine the next word to guess.
        
        Args:
            observation: Optional observation from the environment
            
        Returns:
            str: The word to guess next
        """
        if self.attempts == 0 or len(self.possible_solutions) > 10:
            starters = ['raise', 'slate', 'crate', 'stare', 'roate']
            valid_starters = [w for w in starters if w in self.all_words]
            if valid_starters:
                return valid_starters[0]
        
        if len(self.possible_solutions) == 1:
            return self.possible_solutions[0]
        
        letter_freq = defaultdict(int)
        for word in self.possible_solutions:
            for letter in set(word):
                letter_freq[letter] += 1
        
        best_score = -1
        best_word = None
        
        for word in self.possible_solutions:
            unique_letters = set(word)
            score = sum(letter_freq[letter] for letter in unique_letters)
            
            if score > best_score:
                best_score = score
                best_word = word
        
        return best_word
    
    def update(self, guessed_word, feedback):
        """
        Update the possible solutions based on the feedback.
        
        Args:
            guessed_word (str): The word that was guessed
            feedback (list): List of feedback codes (0=absent, 1=present, 2=correct)
        """
        self.attempts += 1
        self.history.append((guessed_word, feedback))
        
        # Filter the possible solutions
        self.possible_solutions = filter_words(
            self.possible_solutions, guessed_word, feedback, self.word_length
        )
        