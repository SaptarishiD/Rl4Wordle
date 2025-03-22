import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch

class WordleFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Wordle environment.
    Flattens the observation space and processes it through a neural network.
    """
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        board_shape = observation_space.spaces['board'].shape
        board_dim = np.prod(board_shape)
        letter_shape = observation_space.spaces['letter_state'].shape
        letter_dim = np.prod(letter_shape)
        attempt_dim = 1
        
        self.board_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(board_dim, 128),
            nn.ReLU()
        )
        
        self.letter_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(letter_dim, 64),
            nn.ReLU()
        )
        
        self.attempt_net = nn.Sequential(
            nn.Linear(attempt_dim, 16),
            nn.ReLU()
        )
        
        self.combined_net = nn.Sequential(
            nn.Linear(128 + 64 + 16, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        board = torch.as_tensor(observations['board']).float()
        board_features = self.board_net(board)

        letter_state = torch.as_tensor(observations['letter_state']).float()
        letter_features = self.letter_net(letter_state)

        attempt = torch.as_tensor(observations['attempt']).float()
        attempt_features = self.attempt_net(attempt)

        combined = torch.cat([board_features, letter_features, attempt_features], dim=1)
        return self.combined_net(combined)

class WordleFeatureExtractor_Markov(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Markovian Wordle environment.
    Flattens the observation space and processes it through a neural network.
    Extracts information from the game state and feeds it to neural network.
    Supports batched processing for multiple environments.
    """
    
    def __init__(self, observation_space, features_dim=256, num_letters=26, word_length=5):
        super().__init__(observation_space, features_dim)
        
        letter_dim = num_letters * word_length
        self.letter_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(letter_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
        
        # These will now be initialized in reset() based on batch size
        self.last_state = None
        self.missing_letters = None
        self.greens = None
        self.batch_size = None
        self.attempt = None
        # Default word length
        self.word_length = word_length
        self.num_letters = num_letters
    
    def reset(self, batch_size=1):
        """Reset internal state trackers for a new batch of environments"""
        self.batch_size = batch_size
        self.last_state = torch.zeros(batch_size, self.word_length, self.num_letters)
        self.missing_letters = [[[] for _ in range(self.word_length)] for _ in range(batch_size)]
        self.greens = [{} for _ in range(batch_size)]
        self.attempt = None

    def forward(self, observations):
        # Initialize if not already done or if batch size changed
        if self.last_state is None or self.batch_size != observations['board'].shape[0]:
            self.reset(observations['board'].shape[0])
        
        # Create tensor to hold all features for the batch
        batch_features = []
        for b in range(self.batch_size):
            # Get current state for this batch item
            state = self.last_state[b].detach().clone()
            
            # Get the current attempt index
            attempt_idx = observations['attempt'][b].int().item()
            if attempt_idx > 0:  # Only process if there has been at least one attempt
                # Extract feedback and guess for the most recent attempt
                last_feedback = observations['board'][b][attempt_idx-1].int()
                last_guess = observations['guesses'][b][attempt_idx-1].int()
                
                # Skip processing if the guess contains invalid values (-1)
                if not (last_guess < 0).any():
                    # Process information from the latest guess
                    yellows = {}
                    blacks = {}
                    new_greens = {}
                    
                    # Extract information from feedback
                    for idx, (feed, letter) in enumerate(zip(last_feedback, last_guess)):
                        if feed == 2:  # Green
                            letter_item = letter.item()
                            if letter_item not in self.greens[b]:
                                self.greens[b][letter_item] = []
                            if idx not in self.greens[b][letter_item]:
                                self.greens[b][letter_item].append(idx)
                            if letter_item not in new_greens:
                                new_greens[letter_item] = []
                            if idx not in new_greens[letter_item]:
                                new_greens[letter_item].append(idx)
                            
                        elif feed == 1:  # Yellow
                            letter_item = letter.item()
                            if letter_item not in yellows:
                                yellows[letter_item] = []
                            if idx not in yellows[letter_item]:
                                yellows[letter_item].append(idx)
                            
                        elif feed == 0:  # Black
                            letter_item = letter.item()
                            if letter_item not in blacks:
                                blacks[letter_item] = []
                            if idx not in blacks[letter_item]:
                                blacks[letter_item].append(idx)
                    
                    # Process green positions (feed == 2)
                    for letter_idx, positions in self.greens[b].items():
                        for pos in positions:
                            state[pos, letter_idx] = 1
                            # Mark all other letters at this position as impossible
                            for other_letter in range(self.num_letters):
                                if other_letter != letter_idx:
                                    state[pos, other_letter] = -1
                        
                        # Redistribute weight for yellows if this letter became green
                        if letter_idx in new_greens:
                            confirmed_greens = len(new_greens[letter_idx])
                            # Find positions that were candidates for this letter
                            candidate_positions = []
                            for pos in range(self.word_length):
                                if 0 < state[pos, letter_idx] < 1:
                                    candidate_positions.append(pos)
                            
                            # Redistribute weight if there are candidate positions
                            if candidate_positions:
                                weight = state[candidate_positions[0], letter_idx].item()
                                remaining_weight = 1 / (len(candidate_positions) - confirmed_greens) if len(candidate_positions) > confirmed_greens else 0
                                for pos in candidate_positions:
                                    if pos not in positions:  # Not a green position
                                        state[pos, letter_idx] = remaining_weight
                    
                    # Process yellow positions
                    for letter_idx, positions in yellows.items():
                        # For each yellow position, mark it as impossible for this letter
                        for pos in positions:
                            state[pos, letter_idx] = -1
                            if letter_idx not in self.missing_letters[b][pos]:
                                self.missing_letters[b][pos].append(letter_idx)
                        
                        # Find candidate positions
                        candidate_positions = []
                        for pos in range(self.word_length):
                            # Skip if this position is green for any letter
                            if (state[pos] == 1).any():
                                continue
                            # Skip if this position is yellow for this letter
                            if pos in positions:
                                continue
                            # Skip if duplicate letter has black feedback at this position
                            if letter_idx in blacks and pos in blacks[letter_idx]:
                                continue
                            # Skip if this position is already marked impossible for this letter
                            if letter_idx in self.missing_letters[b][pos]:
                                continue
                            candidate_positions.append(pos)
                        
                        # Update candidate positions
                        if candidate_positions:
                            yellow_value = min(1.0, len(positions) / len(candidate_positions))
                            for pos in candidate_positions:
                                state[pos, letter_idx] = yellow_value
                                if yellow_value == 1:
                                    # If this is certain, mark other letters as impossible
                                    for other_letter in range(self.num_letters):
                                        if other_letter != letter_idx:
                                            state[pos, other_letter] = -1
                    
                    # Process black positions
                    for letter_idx, positions in blacks.items():
                        # Check if there's any known information for this letter
                        has_positive_info = (state[:, letter_idx] > 0).any()
                        
                        if has_positive_info:
                            # If we have green or yellow info, just mark black positions as impossible
                            for pos in positions:
                                state[pos, letter_idx] = -1
                                if letter_idx not in self.missing_letters[b][pos]:
                                    self.missing_letters[b][pos].append(letter_idx)
                        else:
                            # No positive info and black feedback => letter is likely absent
                            for pos in range(self.word_length):
                                state[pos, letter_idx] = -1
                                if letter_idx not in self.missing_letters[b][pos]:
                                    self.missing_letters[b][pos].append(letter_idx)
            
            # Update the state for this batch item
            self.last_state[b] = state
            
            # Flatten and add to batch features
            flattened = state.flatten()
            # batch_features.append(torch.cat((flattened, attempt_idx/5)))
            batch_features.append(flattened)
        
        # Stack all batch features and process through the neural network
        batch_tensor = torch.stack(batch_features)
        features = self.letter_net(batch_tensor)
        
        return features