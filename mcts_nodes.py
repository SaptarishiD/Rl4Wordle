from collections import defaultdict
import random
import math
import torch

class MCTSNodeBase:
    def __init__(self, possible_words_tuple, valid_actions_map, parent=None, action=None, attempt=0, max_attempts=6):
        self.state = possible_words_tuple
        self.parent = parent 
        self.action = action 
        self.attempt = attempt # (0-indexed)
        self.max_attempts = max_attempts

        self.children = defaultdict(dict)

        self.action_stats = defaultdict(lambda: {'visits': 0, 'value': 0.0})

        self.visit_count = 0 

        self.word_to_index = valid_actions_map
        self.index_to_word = {v: k for k, v in valid_actions_map.items()}
        self.possible_actions = list(valid_actions_map.values())

    def is_terminal(self):
        """A state is terminal if the game is over (win or loss)."""
        return len(self.state) <= 1 or self.attempt >= self.max_attempts

    def increment_visit(self):
        self.visit_count += 1

    def update_action_stats(self, action_idx, value_increment):
        """Update stats for a specific action taken *from* this node during backpropagation."""
        # action_idx is the index of the guess word
        self.action_stats[action_idx]['visits'] += 1
        self.action_stats[action_idx]['value'] += value_increment


class MCTSNodeVanilla(MCTSNodeBase):
    """Node for vanilla MCTS using possible word sets and random rollouts."""
    def __init__(self, *args, **kwargs):
        """
        Initializes the vanilla MCTS node.
        Derives possible actions and untried actions based on the current state.
        """
        super().__init__(*args, **kwargs)

        current_possible_action_indices = []
        for word in self.state:
            idx = self.word_to_index.get(word)
            if idx is not None:
                current_possible_action_indices.append(idx)
            
        self.possible_actions = list(set(current_possible_action_indices))

        self.untried_actions = list(self.possible_actions)
        random.shuffle(self.untried_actions) 

    def get_current_valid_action_indices(self):
        """Helper to get action indices valid for the *current* state."""
        return set(self.possible_actions) 

    def is_fully_expanded(self):
        """Checks if all actions possible from this state have been tried at least once."""
        return not self.untried_actions

    def select_action_uct(self, exploration_constant):
        """
        Selects the best action from this node using the UCT formula.
        Considers only actions that are currently possible based on the node's state.
        """
        best_score = -float('inf')
        best_actions = []

        current_valid_indices = self.get_current_valid_action_indices()

        tried_valid_actions = []
        for action_idx in self.action_stats.keys():
            if action_idx in current_valid_indices:
                tried_valid_actions.append(action_idx)

        if not tried_valid_actions and not self.untried_actions:
            return -1


        for action_idx in tried_valid_actions:
            stats = self.action_stats[action_idx]

            if stats['visits'] == 0:
                if self.visit_count == 0:
                    score = 0 
                else:
                    score = float('inf')
            else:
                exploit_term = stats['value'] / stats['visits']
                if self.visit_count == 0:
                     explore_term = exploration_constant
                else:
                     explore_term = exploration_constant * math.sqrt(math.log(self.visit_count) / stats['visits'])

                score = exploit_term + explore_term

            if score > best_score:
                best_score = score
                best_actions = [action_idx]
            elif score == best_score:
                best_actions.append(action_idx)

        valid_untried_actions = [action for action in self.untried_actions if action in current_valid_indices]

        if valid_untried_actions:
            return random.choice(valid_untried_actions)
        elif best_actions:
            return random.choice(best_actions)
        else:
            print(f"Warning: No valid action selected by UCT at attempt {self.attempt}. State: {self.state}")
            if current_valid_indices:
                return random.choice(list(current_valid_indices))
            else:
                return -1


class MCTSNodeAlphaZero(MCTSNodeBase):
    """Node for AlphaZero style MCTS, using derived feature tensors."""

    def __init__(self,
                 possible_words_tuple,
                 valid_actions_map,
                 feature_tensor_history,
                 greens,
                 missing_letters,
                 parent,
                 action,
                 attempt: int = 0,
                 max_attempts: int = 6,
                 word_length: int = 5, 
                 alphabet_size: int = 26):

        super().__init__(possible_words_tuple, valid_actions_map,
                         parent=parent, action=action, attempt=attempt,
                         max_attempts=max_attempts)

        self.feature_tensor_history = feature_tensor_history
        self.greens = greens
        self.missing_letters = missing_letters
        self.word_length = word_length
        self.alphabet_size = alphabet_size

        self.policy_probs_dict = None # Policy from network {action_idx: prob}
        self.value = None 

    def get_flattened_state_tensor(self):
        """
        Creates the flattened input tensor for the network from the history.
        Pads with zeros if history is shorter than max_attempts.
        """
        history_len = len(self.feature_tensor_history)
        feature_size_per_step = self.word_length * self.alphabet_size
        total_input_features = self.max_attempts * feature_size_per_step

        flat_tensor = torch.zeros(1, total_input_features, dtype=torch.float32)

        for i, tensor in enumerate(self.feature_tensor_history):
            if i >= self.max_attempts: # Should not happen if logic is correct
                print(f"Warning: History length {history_len} exceeds max_attempts {self.max_attempts}")
                break
            start_idx = i * feature_size_per_step
            end_idx = start_idx + feature_size_per_step
            # Ensure tensor is float and flatten it
            flat_tensor[0, start_idx:end_idx] = tensor.float().flatten()

        return flat_tensor


    def set_evaluation_results(self, value, policy_probs_array):
        """Sets the value and policy probabilities after network evaluation."""
        self.value = value
        # Convert numpy array policy to dictionary format {action_idx: prob}
        # Only store non-zero probabilities to potentially save memory/lookup time
        self.policy_probs_dict = {idx: float(prob) for idx, prob in enumerate(policy_probs_array) if prob > 1e-6} # Use a small threshold


    def select_action_az_uct(self, exploration_constant):
        """Selects an action using AlphaZero style UCT (PUCT formula)."""
        best_score = -float('inf')
        best_actions = [] # To handle ties

        if self.policy_probs_dict is None:
            print("Warning: Selecting action in AZ MCTS node without network policy (not evaluated?). Using random.")
            # Fallback: Choose randomly among possible actions
            if not self.possible_actions:
                print("Error: No possible actions to select from.")
                return -1 # Indicate error or no action possible
            return random.choice(self.possible_actions)

        for action_idx in self.possible_actions:
            stats = self.action_stats.get(action_idx, {'visits': 0, 'value': 0.0})

            Q_sa = stats['value'] / stats['visits'] if stats['visits'] > 0 else 0.0

            P_sa = self.policy_probs_dict.get(action_idx, 0.0) 

            N_sa = stats['visits'] 
            N_s = self.visit_count 

            if N_s > 0:
                explore_term = exploration_constant * P_sa * math.sqrt(N_s) / (1 + N_sa)
            else:
                explore_term = exploration_constant * P_sa

            score = Q_sa + explore_term

            if score > best_score:
                best_score = score
                best_actions = [action_idx]
            elif score == best_score:
                best_actions.append(action_idx)

        if not best_actions:
            print(f"Warning: No best action found in AZ UCT for node attempt {self.attempt}. Possible actions: {len(self.possible_actions)}. Policy Dict: {self.policy_probs_dict is not None}. Choosing random.")
            if not self.possible_actions:
                print("Error: No possible actions exist.")
                return -1 
            return random.choice(self.possible_actions)

        return random.choice(best_actions)

