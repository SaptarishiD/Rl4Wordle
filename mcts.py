import random
from environments import WordleEnvMarkov
from collections import defaultdict
import time
from mcts_nodes import MCTSNodeVanilla
ABSENT, PRESENT, CORRECT = 0, 1, 2

def get_feedback(guess, target):
    n = len(guess)
    feedback = [ABSENT] * n
    target_counts = defaultdict(int)
    for char in target:
        target_counts[char] += 1

    # Greens pass
    remaining_counts = target_counts.copy()
    for i in range(n):
        if guess[i] == target[i]:
            feedback[i] = CORRECT
            remaining_counts[guess[i]] -= 1

    # Yellows pass
    for i in range(n):
        if feedback[i] == CORRECT:
            continue
        if guess[i] in target and remaining_counts[guess[i]] > 0:
            feedback[i] = PRESENT
            remaining_counts[guess[i]] -= 1
    return tuple(feedback) # Return as tuple for hashing/comparison

def filter_words(possible_words, guess, feedback):
    new_possible = []
    for word in possible_words:
        if get_feedback(guess, word) == feedback:
            new_possible.append(word)
    return new_possible

def mcts(initial_possible_words, valid_actions_map, iterations, exploration_constant=1.414, max_attempts=6, current_attempt=0):
    """
    Runs MCTS using MCTSNodeVanilla (possible word sets state, random rollouts).
    State transition relies on sampling a word from the current set during simulation
    to determine feedback.

    Args:
        initial_possible_words: List/set of words possible at the start of this turn.
        valid_actions_map: Dictionary mapping word -> index for all valid guesses.
        iterations: Number of MCTS simulations to run.
        exploration_constant: UCT exploration parameter (e.g., sqrt(2)).
        max_attempts: Total attempts allowed in the game.
        current_attempt: The attempt number for the *current* turn (0-indexed).

    Returns:
        The index of the best action found by MCTS.
    """

    initial_state_tuple = tuple(sorted(initial_possible_words))
    
    root_node = MCTSNodeVanilla(
        possible_words_tuple=initial_state_tuple,
        valid_actions_map=valid_actions_map,
        parent=None,
        action=None,
        attempt=current_attempt,
        max_attempts=max_attempts
    )
    index_to_word = root_node.index_to_word 

    for _ in range(iterations):
        node = root_node 
        path_nodes = [node] 

        # Selection Phase
        # Descend with UCT until leaf or unexpanded node
        while not node.is_terminal():
            if not node.is_fully_expanded():
                # Expansion
                action_to_expand = node.untried_actions.pop(random.randrange(len(node.untried_actions))) 
                
                guess_word = index_to_word[action_to_expand]

                if not node.state:
                    simulated_feedback = tuple([ABSENT for _ in range(len(guess_word))])
                else:
                    sampled_target_word = random.choice(node.state)
                    simulated_feedback = get_feedback(guess_word, sampled_target_word)

                next_possible_words = filter_words(list(node.state), guess_word, simulated_feedback)
                next_state_tuple = tuple(sorted(next_possible_words))

                child_node = MCTSNodeVanilla(
                    possible_words_tuple=next_state_tuple,
                    valid_actions_map=valid_actions_map,
                    parent=node,
                    action=action_to_expand,
                    attempt=node.attempt + 1,
                    max_attempts=max_attempts
                )

                if action_to_expand not in node.children:
                    node.children[action_to_expand] = {}
                node.children[action_to_expand][simulated_feedback] = child_node

                simulation_start_node = child_node
                path_nodes.append(child_node) 
                break # Exit selection/expansion loop

            else:
                action_selected = node.select_action_uct(exploration_constant)

                guess_word = index_to_word[action_selected]
                if not node.state:
                    simulated_feedback = tuple([ABSENT] * len(guess_word))
                else:
                    sampled_target_word = random.choice(node.state)
                    simulated_feedback = get_feedback(guess_word, sampled_target_word)

                child_node = node.children.get(action_selected, {}).get(simulated_feedback)

                if child_node is None:
                    # print(f"Warning: Child node not found for action {action_selected} and feedback {simulated_feedback} in supposedly fully expanded node. Attempt: {node.attempt}")
                    simulation_start_node = node
                    break
                else:
                    # Descend to the selected child
                    node = child_node
                    path_nodes.append(node)
                    # Continue selecting from child

        else:
            # Terminal node selected
            simulation_start_node = node

        # Simulation by Random Rollout
        simulation_value = 0.0
        if simulation_start_node.is_terminal():
            # If outcome is determined
            is_win = len(simulation_start_node.state) == 1 and simulation_start_node.attempt <= max_attempts
            simulation_value = 1.0 if is_win else 0.0
        else:
            current_possible_words_sim = list(simulation_start_node.state)
            current_attempt_sim = simulation_start_node.attempt
            sim_won = False

            if not current_possible_words_sim:
                sim_won = False
            elif len(current_possible_words_sim) == 1:
                sim_won = True # Win if within attempts
            else:
                 # Choose a secret word for the rollout simulation
                secret_word_sim = random.choice(current_possible_words_sim)

                while current_attempt_sim < max_attempts:
                    # Choose a random valid word as the guess for the rollout
                    # Note: Using *any* valid word, not just from possible_words_sim for rollout
                    rollout_guess_idx = random.choice(list(index_to_word.keys()))
                    rollout_guess = index_to_word[rollout_guess_idx]

                    if rollout_guess == secret_word_sim:
                        sim_won = True
                        break # Won the rollout

                    feedback = get_feedback(rollout_guess, secret_word_sim)
                    current_possible_words_sim = filter_words(current_possible_words_sim, rollout_guess, feedback)
                    current_attempt_sim += 1

                    if not current_possible_words_sim:
                        sim_won = False # Contradiction, lost rollout
                        break
                    if len(current_possible_words_sim) == 1:
                        if current_possible_words_sim[0] == secret_word_sim:
                            if current_attempt_sim < max_attempts:
                                sim_won = True 
                            else:
                                sim_won = False 
                        else:
                             sim_won = False 
                        break 

                else:
                    sim_won = False 

            simulation_value = 1.0 if sim_won else 0.0


        # Backpropagation
        node_to_update = path_nodes.pop() 
        while node_to_update is not None:
            node_to_update.increment_visit()
            if node_to_update.parent is not None:
                action_taken_from_parent = node_to_update.action
                if action_taken_from_parent is not None:
                    node_to_update.parent.update_action_stats(action_taken_from_parent, simulation_value)

            if path_nodes:
                node_to_update = path_nodes.pop()
            else:
                node_to_update = None


    # Select from most visited node
    best_action = -1
    max_visits = -1

    if not root_node.action_stats:
        # print("Warning: MCTS root has no action stats after iterations. Choosing random action.")
        if root_node.possible_actions:
            best_action = random.choice(root_node.possible_actions)
        else:
            # print("Error: No possible actions at root.")
            return -1 # Indicate error
        return best_action

    for action_idx, stats in root_node.action_stats.items():
        if stats['visits'] > max_visits:
            max_visits = stats['visits']
            best_action = action_idx

    if best_action == -1:
        # print("Warning: No action found with visits > 0. Choosing random explored action or any possible action.")
        explored_actions = list(root_node.action_stats.keys())
        if explored_actions:
            best_action = random.choice(explored_actions)
        elif root_node.possible_actions:
            best_action = random.choice(root_node.possible_actions)
        else:
            # print("Error: No possible actions at root and no explored actions.")
            return -1

    return best_action


if __name__ == "__main__":
    WORD_LIST_PATH = 'target_words.txt' 
    MAX_ATTEMPTS = 6 
    WORD_LENGTH = 5 
    MCTS_ITERATIONS = 100 
    EXPLORATION = 1.414 

    env = WordleEnvMarkov(WORD_LIST_PATH, max_attempts=MAX_ATTEMPTS, word_length=WORD_LENGTH, render_mode='human')
    obs, info = env.reset()
    terminated = False
    total_reward = 0

    current_possible_words = list(env.valid_words)
    start_time = time.time()

    print(f"Starting Vanilla MCTS Game. Target Word: {env.target_word} (Keep it secret!)")
    print("-" * 20)

    while not terminated:
        env.render() # Show the board state
        print(f"Attempt {env.current_attempt + 1}/{MAX_ATTEMPTS}")
        print(f"Possible words remaining: {len(current_possible_words)}")
        # Optionally print a few possible words if the list is small
        if 0 < len(current_possible_words) < 15:
            print(f"({', '.join(sorted(current_possible_words))})")

        valid_actions_map = env.word_to_index

        mcts_start_time = time.time()
        action = mcts(
            current_possible_words,
            valid_actions_map,
            iterations=MCTS_ITERATIONS,
            exploration_constant=EXPLORATION,
            max_attempts=MAX_ATTEMPTS,
            current_attempt=env.current_attempt
        )
        mcts_end_time = time.time()

        suggested_word = env.index_to_word.get(action, 'INVALID ACTION')
        print(f"MCTS ({mcts_end_time - mcts_start_time:.2f}s) Suggests: {suggested_word} (Index: {action})")

        if action == -1 or suggested_word == 'INVALID ACTION':
                print("Error: MCTS returned invalid action. Stopping.")
                break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if not terminated:
            guessed_word = info.get('guessed_word')
            current_attempt_idx = obs['attempt'][0] - 1
            if current_attempt_idx >= 0:
                feedback = tuple(map(int, obs['board'][current_attempt_idx])) # Convert row to tuple of ints
                print(f"Feedback Received: {feedback}")

                # Filter the possible words based on the actual feedback received
                if guessed_word and feedback is not None:
                    current_possible_words = filter_words(current_possible_words, guessed_word, feedback)
                else:
                    print("Warning: Could not reliably get guess/feedback from env info/obs. Word filtering skipped.")
            else:
                 print("Warning: Could not determine feedback index. Word filtering skipped.")

        print(f"Reward this step: {reward:.2f}, Total Reward: {total_reward:.2f}")
        print("-" * 20)

        if truncated:
            print("Game truncated (should not happen in standard Wordle).")
            terminated = True


    env.render()
    end_time = time.time()
    print(f"\n--- Game Over")
    print(f"Result: {'Success!' if env.won else 'Failure.'}")
    print(f"Target Word was: {env.target_word}")
    print(f"Attempts Used: {env.current_attempt}")
    print(f"Total Reward: {total_reward:.2f}")
    if not env.won and current_possible_words:
            print(f"Remaining possible words ({len(current_possible_words)}): {', '.join(sorted(current_possible_words)[:10])}{'...' if len(current_possible_words) > 10 else ''}")
    elif not env.won:
            print("Remaining possible words: 0")
    print(f"Total Game Time: {end_time - start_time:.2f} seconds")
    env.close()
