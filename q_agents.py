import numpy as np

class QLearningAgent:
    """
    Q-Learning agent for playing Wordle.
    """
    
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, 
                exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize the Q-Learning agent.
        
        Args:
            action_space: The action space of the environment
            learning_rate: The learning rate (alpha)
            discount_factor: The discount factor (gamma)
            exploration_rate: The initial exploration rate (epsilon)
            exploration_decay: The decay rate for exploration
            min_exploration_rate: The minimum exploration rate
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table as a dictionary for sparse representation
        self.q_table = {}
        
    def _get_state_key(self, state):
        """Convert the state to a hashable key"""
        board_state = state['board'].copy()
        attempt = state['attempt']
        letter_state = state['letter_state'].copy()
        
        board_tuple = tuple(map(tuple, np.where(board_state > 0)))
        letter_tuple = tuple(map(tuple, np.where(letter_state > 0)))
        
        return (board_tuple, attempt, letter_tuple)
    
    def get_action(self, state, valid_actions=None):
        """
        Get an action according to an epsilon-greedy policy.
        
        Args:
            state: The current state
            valid_actions: List of valid actions to choose from
        
        Returns:
            The selected action
        """
        if valid_actions is None:
            valid_actions = range(self.action_space.n)
        
        # Exploration: choose a random action
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(valid_actions)
        
        # Exploitation: choose the best action from Q-table
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            # If state not in Q-table, initialize it
            self.q_table[state_key] = np.zeros(self.action_space.n)
        
        q_values = self.q_table[state_key]
        
        valid_q = np.array([q_values[a] for a in valid_actions])
        
        max_q = np.max(valid_q)
        max_actions = [valid_actions[i] for i, q in enumerate(valid_q) if q == max_q]
        
        return np.random.choice(max_actions)
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value for a state-action pair.
        
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state
            done: Whether the episode is done
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        print(state_key)
        print(self.q_table)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)
        
        current_q = self.q_table[state_key][action]
        
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def decay_exploration(self):
        """Decay the exploration rate"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

