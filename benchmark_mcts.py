import torch
import numpy as np
import random
import time
import os
from typing import List, Dict
import matplotlib.pyplot as plt
from environments import WordleEnvMarkov
from alphazero_network import AlphaZeroVPNet
from mcts_nodes import MCTSNodeAlphaZero
from mcts import mcts, get_feedback, filter_words
from train_alphazero import compute_next_feature_state
import pickle

ABSENT = 0
MAX_ATTEMPTS = 6 
WORD_LENGTH = 5 
ALPHABET_SIZE = 26 

def run_mcts_alphazero_inference(
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
) -> int:
    """
    Runs MCTS for AlphaZero using the feature tensor state for INFERENCE.
    Returns only the chosen action index. Does not compute policy targets.
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
            with torch.no_grad():
                 value, policy_probs_array = model.evaluate_state(root_state_tensor_flat)
            root_node.set_evaluation_results(value, policy_probs_array) 
        except Exception as e:
            print(f"Error evaluating root node during inference: {e}")
            policy_probs_array = np.ones(num_total_valid_words, dtype=np.float32) / num_total_valid_words
            root_node.set_evaluation_results(0.0, policy_probs_array)

    for _ in range(iterations):
        node = root_node
        path = [node] 

        # Selection
        while not node.is_terminal():
            if node.policy_probs_dict is None:
                # This node hasn't been expanded/evaluated yet
                break # Move to expansion/simulation

            action_idx = node.select_action_az_uct(exploration_constant)

            if action_idx == -1:
                break 

            guess_word = index_to_word.get(action_idx)
            if guess_word is None:
                print(f"Error: Invalid action index {action_idx} selected during AZ inference.")
                break 

            if node.state:
                random_target = random.choice(node.state) 
                simulated_feedback = get_feedback(guess_word, random_target)
            else:
                simulated_feedback = tuple([ABSENT] * word_length)


            child_node = node.children.get(action_idx, {}).get(simulated_feedback)

            if child_node is None:
                # Expansion
                next_possible_words = filter_words(list(node.state), guess_word, simulated_feedback)
                next_state_tuple = tuple(sorted(next_possible_words))

                last_feature_tensor = node.feature_tensor_history[-1] if node.feature_tensor_history else torch.zeros(word_length, num_letters, dtype=torch.float32)

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

                if action_idx not in node.children: 
                    node.children[action_idx] = {}
                node.children[action_idx][simulated_feedback] = child_node
                path.append(child_node) 

                value_to_backpropagate = 0.0
                if not child_node.is_terminal():
                    try:
                        child_state_tensor_flat = child_node.get_flattened_state_tensor().to(device)
                        with torch.no_grad():
                            value, policy_probs_array = model.evaluate_state(child_state_tensor_flat)
                        child_node.set_evaluation_results(value, policy_probs_array)
                        value_to_backpropagate = value
                    except Exception as e:
                        print(f"Error evaluating expanded child node during inference: {e}")
                        policy_probs_array = np.ones(num_total_valid_words, dtype=np.float32) / num_total_valid_words
                        child_node.set_evaluation_results(0.0, policy_probs_array)
                        value_to_backpropagate = 0.0
                else:
                    is_win = len(child_node.state) == 1 and child_node.attempt <= max_attempts
                    value_to_backpropagate = 1.0 if is_win else 0.0

                # Backpropagation
                temp_node = child_node
                while temp_node is not None:
                    temp_node.increment_visit()
                    if temp_node.parent is not None:
                        action_taken = temp_node.action
                        if action_taken is not None:
                            temp_node.parent.update_action_stats(action_taken, value_to_backpropagate)
                    temp_node = temp_node.parent
                break 

            else:
                node = child_node
                path.append(node)
                # Continue selection from the child

        if node.is_terminal() and node == path[-1]: 
            is_win = len(node.state) == 1 and node.attempt <= max_attempts
            value_to_backpropagate = 1.0 if is_win else 0.0
            temp_node = node
            while temp_node is not None:
                temp_node.increment_visit()
                if temp_node.parent is not None:
                    action_taken = temp_node.action
                    if action_taken is not None:
                         temp_node.parent.update_action_stats(action_taken, value_to_backpropagate)
                temp_node = temp_node.parent


    action_visits = {a: stats['visits'] for a, stats in root_node.action_stats.items()}

    if not action_visits:
        print("Warning: AZ Root node has no action visits after MCTS inference. Choosing random action.")
        chosen_action_idx = random.choice(root_node.possible_actions) if root_node.possible_actions else -1
    else:
        # Select the action with the maximum visit count
        chosen_action_idx = max(action_visits, key=action_visits.get)

    if chosen_action_idx == -1 and root_node.possible_actions:
        print("Warning: chosen_action_idx is -1 despite possible actions in AZ inference. Choosing random.")
        chosen_action_idx = random.choice(root_node.possible_actions)

    return chosen_action_idx

def play_game_alphazero_feature(model, env, mcts_iterations, exploration_constant, device):
    """Plays one game using AlphaZero with feature tensors."""
    model.eval() 
    obs, info = env.reset()
    terminated = False
    truncated = False
    game_won = False
    target_word = env.target_word 

    current_possible_words = list(env.valid_words) 
    valid_actions_map = env.word_to_index
    index_to_word = env.index_to_word

    current_feature_tensor_history = []
    initial_feature_tensor = torch.zeros(WORD_LENGTH, ALPHABET_SIZE, dtype=torch.float32)
    current_feature_tensor_history.append(initial_feature_tensor)

    current_greens = {} 
    current_missing_letters = [[] for _ in range(WORD_LENGTH)] 

    start_time = time.time()

    while not terminated and not truncated:
        current_attempt = env.current_attempt

        action_idx = run_mcts_alphazero_inference(
            current_feature_tensor_history,
            current_greens,
            current_missing_letters,
            current_possible_words,
            valid_actions_map,
            model,
            device,
            mcts_iterations,
            exploration_constant,
            MAX_ATTEMPTS,
            WORD_LENGTH,
            ALPHABET_SIZE,
            current_attempt
        )

        suggested_word = index_to_word.get(action_idx, 'INVALID')
        
        if action_idx == -1 or suggested_word == 'INVALID':
            print(f"  AZ Feat Error: MCTS returned invalid action. Game lost.")
            game_won = False
            break 

        obs, _, terminated, truncated, info = env.step(action_idx)
        game_won = env.won

        if not terminated and not truncated:
            guessed_word = info.get('guessed_word', suggested_word) 

            attempt_idx = obs['attempt'][0] - 1
            feedback = tuple(map(int, obs['board'][attempt_idx]))
            
            current_possible_words = filter_words(current_possible_words, guessed_word, feedback)

            last_feature_tensor = current_feature_tensor_history[-1]
            next_feature_tensor, next_greens, next_missing_letters = compute_next_feature_state(
                last_feature_tensor,
                current_greens,
                current_missing_letters,
                guessed_word,
                feedback,
                WORD_LENGTH,
                ALPHABET_SIZE
            )
            current_feature_tensor_history.append(next_feature_tensor)
            current_greens = next_greens
            current_missing_letters = next_missing_letters

    end_time = time.time()
    return game_won, env.current_attempt, end_time - start_time


def play_game_mcts(env, mcts_iterations, exploration_constant):
    obs, info = env.reset()
    terminated = False
    truncated = False
    game_won = False
    target_word = env.target_word # For debug

    current_possible_words = list(env.valid_words)
    valid_actions_map = env.word_to_index
    index_to_word = env.index_to_word

    start_time = time.time()

    while not terminated and not truncated:
        current_attempt = env.current_attempt

        if not current_possible_words:
            print("  Vanilla MCTS Error: No possible words left. Game lost.")
            game_won = False
            break

        action_idx = mcts(
            current_possible_words,
            valid_actions_map,
            iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            max_attempts=MAX_ATTEMPTS,
            current_attempt=current_attempt
        )

        suggested_word = index_to_word.get(action_idx, 'INVALID')
        # print(f"  Vanilla Attempt {current_attempt+1}: Suggests {suggested_word}") # Optional debug print


        if action_idx == -1 or suggested_word == 'INVALID':
            print(f"  Vanilla MCTS Error: MCTS returned invalid action. Game lost.")
            game_won = False
            break # End game

        obs, _, terminated, truncated, info = env.step(action_idx)
        game_won = env.won

        if not terminated and not truncated:
            guessed_word = info.get('guessed_word', suggested_word)

            attempt_idx = obs['attempt'][0] - 1
            if attempt_idx >= 0:
                feedback = tuple(map(int, obs['board'][attempt_idx]))
            else:
                # print("Warning: Could not extract feedback from observation, calculating manually.")
                feedback = get_feedback(guessed_word, target_word)

            current_possible_words = filter_words(current_possible_words, guessed_word, feedback)

    end_time = time.time()
    return game_won, env.current_attempt, end_time - start_time


if __name__ == "__main__":
    WORD_LIST_PATH = 'target_words.txt' 
    MODEL_PATH = 'alphazero_wordle_model_feat_full_train.pt' 
    NUM_BENCHMARK_GAMES = 1000 
    MCTS_ITERATIONS = 100 
    EXPLORATION_CONSTANT = 1.414 

    print("--- Starting Wordle Solver Benchmark")
    print(f"Number of Games per Method: {NUM_BENCHMARK_GAMES}")
    print(f"MCTS Iterations per Move: {MCTS_ITERATIONS}")
    print(f"Word List: {WORD_LIST_PATH}")
    print(f"AlphaZero Model: {MODEL_PATH}")

    env = WordleEnvMarkov(WORD_LIST_PATH, max_attempts=MAX_ATTEMPTS, word_length=WORD_LENGTH, render_mode=None)
    num_total_valid_words = len(env.valid_words)
    print(f"Environment loaded with {num_total_valid_words} words.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"AlphaZero model not found at {MODEL_PATH}")

    model = AlphaZeroVPNet(WORD_LENGTH, MAX_ATTEMPTS, num_total_valid_words, ALPHABET_SIZE).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Assume the checkpoint *is* the state dict
    print("AlphaZero model loaded successfully.")

    print("\n--- Benchmarking AlphaZero (Feature Tensor)")
    az_wins = 0
    az_total_attempts_on_win = 0
    az_total_time = 0.0
    az_game_times = []
    az_win_attempts = []

    # for i in range(NUM_BENCHMARK_GAMES):
    #     # print(f"  Playing AZ Game {i+1}/{NUM_BENCHMARK_GAMES}...")
    #     won, attempts, duration = play_game_alphazero_feature(model, env, MCTS_ITERATIONS, EXPLORATION_CONSTANT, device)
    #     az_total_time += duration
    #     az_game_times.append(duration)
    #     if won:
    #         az_wins += 1
    #         az_total_attempts_on_win += attempts
    #         az_win_attempts.append(attempts)
    #     print(f"  Result: {'Win' if won else 'Loss'} in {attempts} attempts ({duration:.2f}s)")

    print("\n--- Benchmarking Vanilla MCTS (Possible Words)")
    vanilla_wins = 0
    vanilla_total_attempts_on_win = 0
    vanilla_total_time = 0.0
    vanilla_game_times = []
    vanilla_win_attempts = []

    for i in range(NUM_BENCHMARK_GAMES):
        # print(f"  Playing Vanilla MCTS Game {i+1}/{NUM_BENCHMARK_GAMES}...")
        won, attempts, duration = play_game_mcts(env, MCTS_ITERATIONS, EXPLORATION_CONSTANT)
        vanilla_total_time += duration
        vanilla_game_times.append(duration)
        if won:
            vanilla_wins += 1
            vanilla_total_attempts_on_win += attempts
            vanilla_win_attempts.append(attempts)
        print(f"  Result: {'Win' if won else 'Loss'} in {attempts} attempts ({duration:.2f}s)")

    print("\n--- Benchmark Results")

    az_win_rate = (az_wins / NUM_BENCHMARK_GAMES) * 100 if NUM_BENCHMARK_GAMES > 0 else 0
    az_avg_attempts = az_total_attempts_on_win / az_wins if az_wins > 0 else 0
    az_avg_time = az_total_time / NUM_BENCHMARK_GAMES if NUM_BENCHMARK_GAMES > 0 else 0
    az_time_std = np.std(az_game_times) if len(az_game_times) > 1 else 0
    az_attempts_std = np.std(az_win_attempts) if len(az_win_attempts) > 1 else 0

    print("\nAlphaZero (Feature Tensor):")
    print(f"  Win Rate: {az_win_rate:.2f}% ({az_wins}/{NUM_BENCHMARK_GAMES})")
    if az_wins > 0:
        print(f"  Avg. Attempts on Win: {az_avg_attempts:.2f} (StdDev: {az_attempts_std:.2f})")
    else:
        print("  Avg. Attempts on Win: N/A (0 wins)")
    print(f"  Avg. Time per Game: {az_avg_time:.3f}s (StdDev: {az_time_std:.3f}s)")

    vanilla_win_rate = (vanilla_wins / NUM_BENCHMARK_GAMES) * 100 if NUM_BENCHMARK_GAMES > 0 else 0
    vanilla_avg_attempts = vanilla_total_attempts_on_win / vanilla_wins if vanilla_wins > 0 else 0
    vanilla_avg_time = vanilla_total_time / NUM_BENCHMARK_GAMES if NUM_BENCHMARK_GAMES > 0 else 0
    vanilla_time_std = np.std(vanilla_game_times) if len(vanilla_game_times) > 1 else 0
    vanilla_attempts_std = np.std(vanilla_win_attempts) if len(vanilla_win_attempts) > 1 else 0
    with open('successes.pkl', 'wb') as f:
        pickle.dump(vanilla_win_attempts, f)
    
    print("\nVanilla MCTS (Possible Words):")
    print(f"  Win Rate: {vanilla_win_rate:.2f}% ({vanilla_wins}/{NUM_BENCHMARK_GAMES})")
    if vanilla_wins > 0:
        print(f"  Avg. Attempts on Win: {vanilla_avg_attempts:.2f} (StdDev: {vanilla_attempts_std:.2f})")
    else:
        print("  Avg. Attempts on Win: N/A (0 wins)")
    print(f"  Avg. Time per Game: {vanilla_avg_time:.3f}s (StdDev: {vanilla_time_std:.3f}s)")
    plt.bar([x+1 for x in range(6)], vanilla_win_attempts)
    plt.savefig('mcts_successs_plot.png')
    plt.close()
    env.close()
    print("\n--- Benchmark Complete")
