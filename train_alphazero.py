import random
from mcts_nodes import MCTSNodeAlphaZero
from alphazero_network import AlphaZeroVPNet, ABSENT, PRESENT, CORRECT
from environments import WordleEnvMarkov
from collections import defaultdict, deque
import time
from typing import List, Tuple, Dict 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
writer = SummaryWriter(f'logs/alphazero/{datetime.now().strftime("x")}')

def compute_next_feature_state(
    last_feature_tensor, # (word_length, num_letters)
    last_greens,
    last_missing_letters,
    guess_word, 
    feedback, 
    word_length,
    num_letters
):
    # Use copies to avoid modifying the parent node's state directly here
    state = last_feature_tensor.detach().clone()
    current_greens = {k: list(v) for k, v in last_greens.items()}
    current_missing_letters = [list(sublist) for sublist in last_missing_letters]

    last_guess_indices = [ord(c.lower()) - ord('a') for c in guess_word]
    
    feedback_int = tuple(map(int, feedback))

    yellows = defaultdict(list) 
    blacks = defaultdict(list) 
    new_greens = defaultdict(list) 

    for idx, (feed_item, letter_item) in enumerate(zip(feedback_int, last_guess_indices)):

        if feed_item == CORRECT:  # Green (2)
            if idx not in current_greens.get(letter_item, []):
                if letter_item not in current_greens: current_greens[letter_item] = []
                current_greens[letter_item].append(idx)
                # Track newly identified greens in this specific guess
                new_greens[letter_item].append(idx)

        elif feed_item == PRESENT:  # Yellow (1)
            if idx not in yellows[letter_item]:
                 yellows[letter_item].append(idx)

        elif feed_item == ABSENT:  # Black/Gray (0)
            if idx not in blacks[letter_item]:
                 blacks[letter_item].append(idx)

    for letter_idx, positions in current_greens.items():
        for pos in positions:
            if 0 <= pos < word_length: # Boundary check
                state[pos, letter_idx] = 1.0 # Set green position to 1
                for other_letter in range(num_letters):
                    if other_letter != letter_idx:
                        state[pos, other_letter] = -1.0
                        if other_letter not in current_missing_letters[pos]:
                            current_missing_letters[pos].append(other_letter)

    # for letter_idx, newly_confirmed_positions in new_greens.items():
    #      candidate_yellow_positions = []
    #      for pos_check in range(word_length):
    #           # Check if it was a candidate (e.g., yellow or unknown) before this guess
    #           if 0 < state[pos_check, letter_idx] < 1:
    #                # Exclude positions just confirmed as green in *this* step
    #                if pos_check not in newly_confirmed_positions:
    #                    candidate_yellow_positions.append(pos_check)

    # Process yellow positions (feed == 1)
    for letter_idx, yellow_positions in yellows.items():
        # Mark the exact yellow positions as impossible for this letter
        for pos in yellow_positions:
            if 0 <= pos < word_length: # Boundary check
                state[pos, letter_idx] = -1.0
                if letter_idx not in current_missing_letters[pos]:
                    current_missing_letters[pos].append(letter_idx)

        candidate_green_positions = []
        for pos_check in range(word_length):
            if torch.any(state[pos_check, :] == 1.0): continue
            if pos_check in yellow_positions: continue
            if state[pos_check, letter_idx] == -1.0: continue
            if letter_idx in current_missing_letters[pos_check]: continue
            candidate_green_positions.append(pos_check)

        if candidate_green_positions:
            yellow_prob_value = 1/len(candidate_green_positions)
            for pos in candidate_green_positions:
                if state[pos, letter_idx] != 1.0 and state[pos, letter_idx] >=0:
                    state[pos, letter_idx] = max(state[pos, letter_idx], yellow_prob_value) # Keep higher prob if exists


    # Process black positions (feed == 0)
    for letter_idx, black_positions in blacks.items():
        has_positive_info = False
        if letter_idx in current_greens: has_positive_info = True
        if letter_idx in yellows: has_positive_info = True
        # for pos_check in range(word_length):
        #     if state[pos_check, letter_idx] > 0: # Check for >0 or ==1? Let's use >0
        #         has_positive_info = True
        #         break

        if has_positive_info:
            for pos in black_positions:
                 if 0 <= pos < word_length: # Boundary check
                     state[pos, letter_idx] = -1.0
                     if letter_idx not in current_missing_letters[pos]:
                         current_missing_letters[pos].append(letter_idx)
        else:
            for pos in range(word_length):
                state[pos, letter_idx] = -1.0
                if letter_idx not in current_missing_letters[pos]:
                    current_missing_letters[pos].append(letter_idx)

    return state, current_greens, current_missing_letters


def get_feedback(guess: str, target: str) -> Tuple[int, ...]:
    n = len(guess)
    
    feedback = [ABSENT for _ in range(n)]
    target_counts = defaultdict(int)
    guess_indices = list(range(n))
    for i in range(n):
        if guess[i] == target[i]:
            feedback[i] = CORRECT
            target_counts[target[i]] += 1
            guess_indices.remove(i)

    remaining_target_counts = defaultdict(int)
    for i in range(n):
        if feedback[i] != CORRECT:
            remaining_target_counts[target[i]] += 1


    for i in guess_indices:
        char = guess[i]
        if remaining_target_counts.get(char, 0) > 0:
            feedback[i] = PRESENT
            remaining_target_counts[char] -= 1

    return tuple(feedback)


def filter_words(possible_words: List[str], guess: str, feedback: Tuple[int, ...]) -> List[str]:
    """Filters a list of words based on a guess and its feedback."""
    new_possible = []
    for word in possible_words:
        if get_feedback(guess, word) == feedback:
            new_possible.append(word)
    return new_possible

def visit_counts_to_policy(visit_counts: Dict[int, int], all_action_indices: List[int], temperature: float = 1.0) -> np.ndarray:
    """
    Converts MCTS root node visit counts into a probability distribution (policy target).
    Args:
        visit_counts: Dictionary {action_idx: visits} from the root node.
        all_action_indices: List of all possible action indices (0 to num_total_valid_words-1).
        temperature: Temperature parameter for softening the distribution. Tau=1 is common.

    Returns:
        A numpy array representing the policy probability distribution over all action indices.
    """
    num_actions = len(all_action_indices)
    action_idx_to_pos = {action_idx: i for i, action_idx in enumerate(all_action_indices)}
    policy_target = np.zeros(num_actions, dtype=np.float32)

    if not visit_counts or num_actions == 0:
        if num_actions > 0:
             policy_target.fill(1.0 / num_actions) # Uniform if no visits or no actions
        return policy_target

    actions_visited = list(visit_counts.keys())
    visits = np.array([visit_counts[a] for a in actions_visited], dtype=np.float32)

    if temperature == 0:
        max_visits = np.max(visits)
        best_action_indices_in_visits = np.where(visits == max_visits)[0]
        original_best_actions = [actions_visited[i] for i in best_action_indices_in_visits]
        prob = 1.0 / len(original_best_actions)
        for action_idx in original_best_actions:
             pos = action_idx_to_pos.get(action_idx)
             if pos is not None:
                 policy_target[pos] = prob
    else:
        epsilon = 1e-9
        scaled_visits = (visits + epsilon) ** (1.0 / temperature)
        sum_scaled_visits = np.sum(scaled_visits)

        if sum_scaled_visits == 0:
            prob = 1.0 / len(actions_visited) if actions_visited else 0
            for action_idx in actions_visited:
                pos = action_idx_to_pos.get(action_idx)
                if pos is not None:
                    policy_target[pos] = prob
        else:
            policy_probs_visited = scaled_visits / sum_scaled_visits
            for i, action_idx in enumerate(actions_visited):
                pos = action_idx_to_pos.get(action_idx)
                if pos is not None:
                    policy_target[pos] = policy_probs_visited[i]

    policy_sum = np.sum(policy_target)
    if policy_sum > 0:
        policy_target /= policy_sum
    elif num_actions > 0:
        policy_target.fill(1.0 / num_actions)

    return policy_target


class ExperienceReplayBuffer:
    """Stores experiences (state_tensor, policy_target, value_outcome)."""
    def __init__(self, max_size):
        self.buffer = deque([], maxlen=max_size)

    def add(self, experience: Tuple[torch.Tensor, np.ndarray, float]):
        """Adds a single experience tuple."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a batch of experiences."""
        actual_batch_size = min(batch_size, len(self.buffer))
        if actual_batch_size == 0:
             raise ValueError("Cannot sample from an empty buffer.")

        batch = random.sample(list(self.buffer), actual_batch_size)
        states, policies, outcomes = zip(*batch)

        states_tensor = torch.stack(states)
        policies_tensor = torch.tensor(np.stack(policies), dtype=torch.float32)
        outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1)

        return states_tensor, policies_tensor, outcomes_tensor

    def __len__(self):
        return len(self.buffer)

def run_mcts_alphazero_train(
    current_feature_tensor_history: List[torch.Tensor],
    current_greens: Dict[int, List[int]],
    current_missing_letters: List[List[int]],
    current_possible_words: List[str],
    valid_actions_map: Dict[str, int],
    model: AlphaZeroVPNet,
    device: torch.device,
    iterations: int,
    exploration_constant: float = 1.414,
    max_attempts: int = 6,
    word_length: int = 5,
    num_letters: int = 26,
    current_attempt: int = 0,
) -> Tuple[int, np.ndarray]:
    """
    Runs MCTS for AlphaZero using the feature tensor state representation.
    Returns the chosen action and the MCTS policy distribution for training.
    """
    index_to_word = {v: k for k, v in valid_actions_map.items()}
    num_total_valid_words = len(valid_actions_map)

    initial_state_tuple = tuple(sorted(current_possible_words))

    root_node = MCTSNodeAlphaZero(
        initial_state_tuple,
        valid_actions_map,
        feature_tensor_history=list(current_feature_tensor_history), 
        greens=dict(current_greens),
        missing_letters=[list(m) for m in current_missing_letters],
        parent=None,
        action=None,
        attempt=current_attempt,
        max_attempts=max_attempts,
        word_length=word_length,
        alphabet_size=num_letters
    )

    if not root_node.is_terminal():
        try:
            root_state_tensor_flat = root_node.get_flattened_state_tensor().to(device)
            # Evaluate using the network
            value, policy_probs_array = model.evaluate_state(root_state_tensor_flat)
            root_node.set_evaluation_results(value, policy_probs_array) # Store results
        except Exception as e:
            print(f"Error evaluating root node: {e}")
            # Fallback: uniform policy, zero value
            policy_probs_array = np.ones(num_total_valid_words, dtype=np.float32) / num_total_valid_words
            root_node.set_evaluation_results(0.0, policy_probs_array)


    for _ in range(iterations):
        node = root_node
        path = [node]

        # Selection
        while not node.is_terminal():
            if node.policy_probs_dict is None:
                break

            action_idx = node.select_action_az_uct(exploration_constant)

            if action_idx == -1:
                print("Warning: MCTS selected invalid action (-1). Breaking simulation.")
                break # End this simulation iteration

            guess_word = index_to_word.get(action_idx)
            if guess_word is None:
                 print(f"Error: Invalid action index {action_idx} selected.")
                 break # Invalid action

            # Simulate feedback with random word
            if node.state:
                random_target = random.sample(node.state, 1)[0]
                simulated_feedback = get_feedback(guess_word, random_target)
            else:
                simulated_feedback = tuple([ABSENT] * word_length)


            child_node = node.children.get(action_idx, {}).get(simulated_feedback)

            if child_node is None:
                # Expansion
                next_possible_words = filter_words(list(node.state), guess_word, simulated_feedback)
                next_state_tuple = tuple(sorted(next_possible_words))

                last_feature_tensor = node.feature_tensor_history[-1] if node.feature_tensor_history else torch.zeros(word_length, num_letters)

                next_feature_tensor, next_greens, next_missing_letters = compute_next_feature_state(
                    last_feature_tensor,
                    node.greens,
                    node.missing_letters,
                    guess_word,
                    simulated_feedback,
                    word_length,
                    num_letters
                )

                next_feature_tensor_history = node.feature_tensor_history + [next_feature_tensor]

                child_node = MCTSNodeAlphaZero(
                    next_state_tuple,
                    valid_actions_map,
                    feature_tensor_history=next_feature_tensor_history,
                    greens=next_greens,
                    missing_letters=next_missing_letters,
                    parent=node,
                    action=action_idx,
                    attempt=node.attempt + 1,
                    max_attempts=max_attempts,
                    word_length=word_length,
                    alphabet_size=num_letters
                )

                if action_idx not in node.children: node.children[action_idx] = {}
                node.children[action_idx][simulated_feedback] = child_node

                if not child_node.is_terminal():
                    child_state_tensor_flat = child_node.get_flattened_state_tensor().to(device)
                    value, policy_probs_array = model.evaluate_state(child_state_tensor_flat)
                    child_node.set_evaluation_results(value, policy_probs_array)
                    

                value_to_backpropagate = 0.0
                if child_node.is_terminal():
                    is_win = len(child_node.state) == 1 and child_node.attempt <= max_attempts
                    value_to_backpropagate = 1.0 if is_win else 0.0
                elif child_node.value is not None:
                    value_to_backpropagate = child_node.value

                temp_node = child_node
                while temp_node is not None:
                    temp_node.increment_visit()
                    if temp_node.parent is not None:
                        action_taken_from_parent = temp_node.action
                        if action_taken_from_parent is not None:
                             temp_node.parent.update_action_stats(action_taken_from_parent, value_to_backpropagate)
                    temp_node = temp_node.parent
                break

            else:
                node = child_node
                path.append(node)
        
        # If simulation ended by reaching a terminal node without expansion
        if node.is_terminal() and node == path[-1]: # Ensure it's the last node added
            is_win = len(node.state) == 1 and node.attempt <= max_attempts
            value_to_backpropagate = 1.0 if is_win else 0.0
            # Backpropagate from this terminal node
            temp_node = node
            while temp_node is not None:
                temp_node.increment_visit()
                if temp_node.parent is not None:
                    action_taken_from_parent = temp_node.action
                    if action_taken_from_parent is not None:
                         temp_node.parent.update_action_stats(action_taken_from_parent, value_to_backpropagate)
                temp_node = temp_node.parent

    action_visits = {a: stats['visits'] for a, stats in root_node.action_stats.items()}

    if not action_visits:
        print("Warning: Root node has no action visits after MCTS. Choosing random action.")
        chosen_action_idx = random.choice(root_node.possible_actions) if root_node.possible_actions else -1
        mcts_policy_target = np.ones(num_total_valid_words, dtype=np.float32) / num_total_valid_words if num_total_valid_words > 0 else np.zeros(num_total_valid_words, dtype=np.float32)
    else:
        chosen_action_idx = max(action_visits, key=action_visits.get)

        # Calculate the policy target for training (using temperature=1)
        mcts_policy_target = visit_counts_to_policy(
            action_visits,
            list(valid_actions_map.values()), # Pass all possible action indices
            temperature=1.0 # Use temperature=1 for training data generation
        )

    if chosen_action_idx == -1 and root_node.possible_actions:
        print("Warning: chosen_action_idx is -1 despite possible actions. Choosing random.")
        chosen_action_idx = random.choice(root_node.possible_actions)


    return chosen_action_idx, mcts_policy_target


def train_alphazero(
    num_self_play_games=1000,
    mcts_iterations_per_move=50, 
    erb_size=10000,
    batch_size=64, 
    train_steps_per_game=5, 
    learning_rate=1e-4, 
    exploration_constant=1.414,
    max_attempts=6,
    word_length=5,
    num_letters=26, 
    word_list_path='target_words.txt', 
    model_save_path='alphazero_wordle_model_feat.pt',
    start_from_checkpoint=False
):
    print("\n--- AlphaZero Training Setup (Feature Tensor)")
    print(f"Self-Play Games: {num_self_play_games}")
    print(f"MCTS Iterations per move: {mcts_iterations_per_move}")
    print(f"ERB Size: {erb_size}")
    print(f"Training Batch Size: {batch_size}")
    print(f"Train Steps per Game: {train_steps_per_game}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Exploration Constant (MCTS): {exploration_constant}")
    print(f"Max Attempts: {max_attempts}")
    print(f"Word Length: {word_length}")
    print(f"Model Save Path: {model_save_path}")
    print(f"Start from Checkpoint: {start_from_checkpoint}")

    env = WordleEnvMarkov(word_list_path, max_attempts=max_attempts, word_length=word_length, render_mode=None)
    valid_actions_map = env.word_to_index
    index_to_word = env.index_to_word
    num_total_valid_words = len(valid_actions_map)
    print(f"Loaded environment with {num_total_valid_words} valid words.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AlphaZeroVPNet(word_length, max_attempts, num_total_valid_words, num_letters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    erb = ExperienceReplayBuffer(max_size=erb_size)

    start_game_i = 0
    if start_from_checkpoint:
        print(f"Loading model from {model_save_path}")
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_game_i = checkpoint.get('game_i', 0)
        # erb_state = checkpoint.get('erb_buffer') # Optionally load ERB state
        # if erb_state: erb.buffer = deque(erb_state, maxlen=erb_size)
        print(f"Checkpoint loaded successfully. Resuming from game {start_game_i + 1}.")
        

    total_steps = 0
    win_rates = deque(maxlen=100) 
    attempts_on_wins = deque(maxlen=100) 
    
    for game_i in range(start_game_i, num_self_play_games):
        game_start_time = time.time()
        obs, info = env.reset() 
        terminated = False
        current_possible_words = list(env.valid_words) 

        current_feature_tensor_history: List[torch.Tensor] = []
        initial_feature_tensor = torch.zeros(word_length, num_letters, dtype=torch.float32)
        current_feature_tensor_history.append(initial_feature_tensor)

        current_greens: Dict[int, List[int]] = {} # No greens initially
        current_missing_letters: List[List[int]] = [[] for _ in range(word_length)]

        game_experiences = [] 
        game_won = False

        print(f"\n--- Game {game_i + 1}/{num_self_play_games} (Target: {env.target_word})")

        while not terminated and env.current_attempt < max_attempts:
            current_attempt = env.current_attempt 
            current_node_for_mcts = MCTSNodeAlphaZero( 
                tuple(current_possible_words), valid_actions_map,
                current_feature_tensor_history, current_greens, current_missing_letters, None, None,
                attempt=current_attempt, max_attempts=max_attempts,
                word_length=word_length, alphabet_size=num_letters,
            )
            state_tensor_flat_for_erb = current_node_for_mcts.get_flattened_state_tensor().squeeze(0) 

            mcts_start_time = time.time()
            chosen_action_idx, mcts_policy_target = run_mcts_alphazero_train(
                current_feature_tensor_history,
                current_greens,
                current_missing_letters,
                current_possible_words,
                valid_actions_map,
                model,
                device,
                mcts_iterations_per_move,
                exploration_constant,
                max_attempts,
                word_length,
                num_letters,
                current_attempt
            )
            mcts_end_time = time.time()
            suggested_word = index_to_word.get(chosen_action_idx, 'INVALID ACTION')

            if chosen_action_idx == -1 or suggested_word == 'INVALID ACTION':
                print(f"Attempt {current_attempt + 1}: MCTS returned invalid action. Ending game as loss.")
                terminated = True
                game_won = False
                # Don't store experience if MCTS failed
                break

            # Store the state (tensor before action) and the MCTS policy target
            # We'll add the outcome later
            game_experiences.append((state_tensor_flat_for_erb.cpu(), mcts_policy_target))

            # print(f"Attempt {current_attempt + 1}/{max_attempts} ({len(current_possible_words)} possible): MCTS ({mcts_end_time - mcts_start_time:.2f}s) Suggests: {suggested_word}")

            obs, _, terminated, truncated, info = env.step(chosen_action_idx)
            game_won = env.won 

            if not terminated:
                guessed_word_env = info.get('guessed_word')
                if 'board' in obs and obs['board'].shape == (max_attempts, word_length):
                    feedback_array = obs['board'][current_attempt] # Use current_attempt index
                    feedback = tuple(map(int, feedback_array)) # Convert to tuple of ints
                else:
                    # print("Error: Could not extract feedback from environment observation.")
                    feedback = get_feedback(suggested_word, env.target_word)

                if guessed_word_env != suggested_word:
                    # print(f"Warning: Env guess '{guessed_word_env}' differs from MCTS guess '{suggested_word}'")
                    guessed_word_for_update = suggested_word
                else:
                    guessed_word_for_update = guessed_word_env


                current_possible_words = filter_words(current_possible_words, guessed_word_for_update, feedback)

                last_feature_tensor = current_feature_tensor_history[-1]
                next_feature_tensor, next_greens, next_missing_letters = compute_next_feature_state(
                    last_feature_tensor,
                    current_greens,
                    current_missing_letters,
                    guessed_word_for_update,
                    feedback,
                    word_length,
                    num_letters
                )
                current_feature_tensor_history.append(next_feature_tensor)
                current_greens = next_greens
                current_missing_letters = next_missing_letters

            if truncated:
                # print("Warning: Episode truncated unexpectedly.")
                terminated = True
                game_won = False


        attempts_made = env.current_attempt
        game_outcome = 1.0 if game_won else 0.0

        # print(f"Game Over. Result: {'Win!' if game_won else 'Loss.'} Target: {env.target_word}, Attempts: {attempts_made}")

        win_rates.append(game_won)
        if game_won:
            attempts_on_wins.append(attempts_made)

        num_exp = len(game_experiences)
        for state_tensor_flat, policy_target in game_experiences:
            erb.add((state_tensor_flat, policy_target, game_outcome)) 
        # print(f"Added {num_exp} experiences to ERB (current size: {len(erb)})")

        if len(erb) >= batch_size:
            model.train() 
            train_loss_total = 0.0
            policy_loss_total = 0.0
            value_loss_total = 0.0

            # print(f"Starting training for {train_steps_per_game} steps...")
            for train_step in range(train_steps_per_game):
                states, policy_targets, value_targets = erb.sample(batch_size)
                states, policy_targets, value_targets = states.to(device), policy_targets.to(device), value_targets.to(device)

                optimizer.zero_grad()

                predicted_values, policy_logits = model(states)

                # Policy loss: compare network logits with MCTS policy target
                # print(policy_targets.shape, policy_logits.shape)
                policy_loss = policy_loss_fn(policy_logits, policy_targets) 
                # Value loss: compare network value with actual game outcome
                value_loss = value_loss_fn(predicted_values, value_targets)

                total_loss = policy_loss + value_loss

                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                current_total_step = total_steps + train_step
                train_loss_total += total_loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                writer.add_scalar('Loss/Total_Step', total_loss.item(), current_total_step)
                writer.add_scalar('Loss/Policy_Step', policy_loss.item(), current_total_step)
                writer.add_scalar('Loss/Value_Step', value_loss.item(), current_total_step)

            avg_train_loss = train_loss_total / train_steps_per_game
            avg_policy_loss = policy_loss_total / train_steps_per_game
            avg_value_loss = value_loss_total / train_steps_per_game
            writer.add_scalar('Loss/Total_Avg_Per_TrainPhase', avg_train_loss, game_i + 1)
            writer.add_scalar('Loss/Policy_Avg_Per_TrainPhase', avg_policy_loss, game_i + 1)
            writer.add_scalar('Loss/Value_Avg_Per_TrainPhase', avg_value_loss, game_i + 1)
            # print(f"Training Step Avg Loss = {avg_train_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")
            total_steps += train_steps_per_game
        else:
            pass
            # print(f"ERB size ({len(erb)}) less than batch size ({batch_size}), skipping training step.")

        game_end_time = time.time()
        # print(f"Game {game_i + 1} finished in {game_end_time - game_start_time:.2f} seconds.")

        if (game_i + 1) % 50 == 0 or (game_i + 1) == num_self_play_games:
            print(f"\n--- Saving Model after Game {game_i + 1}")
            try:
                torch.save({
                    'game_i': game_i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'erb_buffer': list(erb.buffer), # Saving ERB can make files large
                }, model_save_path)
                print(f"Model saved to {model_save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

            # Log Training Stats
            current_win_rate = sum(win_rates) / len(win_rates) * 100 if len(win_rates) > 0 else 0
            current_avg_attempts = sum(attempts_on_wins) / len(attempts_on_wins) if len(attempts_on_wins) > 0 else 0
            writer.add_scalar('WinRate/Overall_100Games', current_win_rate, game_i + 1)
            if current_avg_attempts > 0:
                 writer.add_scalar('Performance/Avg_Attempts_On_Wins_100Games', current_avg_attempts, game_i + 1)
            writer.add_scalar('Progress/Total_Training_Steps', total_steps, game_i + 1) # Log total steps vs game number too
            print(f"\n--- Training Stats (Last {len(win_rates)} games)")
            print(f"Win Rate: {current_win_rate:.2f}%")
            if current_avg_attempts > 0:
                 print(f"Avg Attempts on Wins: {current_avg_attempts:.2f}")
            print(f"Total Training Steps: {total_steps}")
            print(f"ERB Size: {len(erb)}")
            print("-" * 30)
    env.close()
    print("\n--- AlphaZero Training Completed")


# Main Execution Guard
if __name__ == "__main__":
    train_alphazero(
        num_self_play_games=100000, 
        mcts_iterations_per_move=50, 
        erb_size=10000, 
        batch_size=128, 
        train_steps_per_game=16, 
        learning_rate=0.0001, 
        exploration_constant=1.4, 
        max_attempts=6, 
        word_length=5, 
        num_letters=26, 
        word_list_path='target_words.txt', 
        model_save_path='alphazero_wordle_model_feat.pt', 
        start_from_checkpoint=False 
    )
