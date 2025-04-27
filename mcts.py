import math
import random
import numpy as np
import os
from environments import WordleEnvMarkov
from collections import defaultdict
import time

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


class MCTSNodePossibleWords:
    def __init__(self, possible_words_tuple, valid_actions_map, parent=None, action=None, attempt=0, max_attempts=6):
        self.state = possible_words_tuple
        self.parent:MCTSNodePossibleWords = parent
        self.action = action
        self.attempt = attempt
        self.max_attempts = max_attempts

        self.children = defaultdict(dict)

        self.action_stats = {}

        self.visit_count = 0

        self.word_to_index = valid_actions_map
        self.index_to_word = {v: k for k, v in valid_actions_map.items()}
        self.possible_actions = list(valid_actions_map.values())
        self.untried_actions = list(self.possible_actions)
        random.shuffle(self.untried_actions)


    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return len(self.state) <= 1 or self.attempt >= self.max_attempts

    def select_action_uct(self, exploration_constant):
        best_score = -float('inf')
        best_action = -1

        tried_actions = list(self.action_stats.keys())
        if not tried_actions:
            return random.choice(self.possible_actions)

        for action_idx in tried_actions:
            stats = self.action_stats[action_idx]
            if stats['visits'] == 0:
                score = float('inf')
            else:
                if self.visit_count == 0: continue
                exploit_term = stats['value'] / stats['visits']
                explore_term = exploration_constant * math.sqrt(math.log(self.visit_count) / stats['visits'])
                score = exploit_term + explore_term

            if score > best_score:
                best_score = score
                best_action = action_idx

        if best_action == -1:
            best_action = random.choice(tried_actions) if tried_actions else random.choice(self.possible_actions)

        return best_action

    def increment_visit(self):
        self.visit_count += 1
        
    def update_action_stats(self, action_idx, value_increment):
        """Update stats for a specific action taken *from* this node."""
        if action_idx not in self.action_stats:
            self.action_stats[action_idx] = {'visits': 0, 'value': 0.0}
        self.action_stats[action_idx]['visits'] += 1
        self.action_stats[action_idx]['value'] += value_increment
        

def mcts_possible_words(initial_possible_words, valid_actions_map, iterations, exploration_constant=1.414, max_attempts=6, current_attempt=0):
    """
    Runs MCTS using possible word sets. State transition relies on sampling a word
    from the current set during simulation to determine feedback.
    Args:
        initial_possible_words: List/set of words possible at the start of this turn.
        valid_actions_map: Dictionary mapping word -> index for all valid guesses.
        iterations: Number of MCTS simulations to run.
        exploration_constant: UCT exploration parameter (e.g., sqrt(2)).
        max_attempts: Total attempts allowed in the game.
        current_attempt: The attempt number for the *current* turn (0-indexed).
    """

    initial_state_tuple = tuple(sorted(initial_possible_words))
    root_node = MCTSNodePossibleWords(
        initial_state_tuple,
        valid_actions_map,
        attempt=current_attempt,
        max_attempts=max_attempts
    )
    index_to_word = root_node.index_to_word

    for _ in range(iterations):
        node = root_node
        path_nodes = [node] 

        # Selection: DFS using UCT until a leaf or expandable node
        while not node.is_terminal():
            if not node.is_fully_expanded():
                # Expansion : Choose random untried action
                action_to_expand = node.untried_actions.pop(random.choice(range(len(node.untried_actions))))
                if action_to_expand not in node.action_stats:
                    node.action_stats[action_to_expand] = {'visits': 0, 'value': 0.0}
                simulation_action = action_to_expand
                simulation_start_node = node
                break
            else:
                action_selected = node.select_action_uct(exploration_constant)

                if not node.state: # Should not happen if not terminal
                    # Treat as terminal if state becomes empty
                    simulation_action = -1
                    simulation_start_node = node
                    break

                sampled_target_word = random.choice(node.state)
                guess_word = index_to_word[action_selected]
                feedback = get_feedback(guess_word, sampled_target_word)

                if feedback not in node.children.get(action_selected, {}):
                    simulation_action = action_selected
                    simulation_start_node = node
                    break
                else:
                    # Move to the existing child node corresponding to this action and feedback.
                    node = node.children[action_selected][feedback]
                    path_nodes.append(node)
                    # Continue selection from the child node.
        else:
            # Terminal Node reached
            simulation_action = -1 # No action to simulate from a terminal node.
            simulation_start_node = node

        # Simulate a game playthrough from 'simulation_start_node' state using random guessing
        simulation_value = 0.0
        if simulation_start_node.is_terminal():
            if len(simulation_start_node.state) == 1: 
                simulation_value = 1.0
            else: 
                simulation_value = 0.0 
        elif simulation_action != -1:
            # An action was chosen (expansion/UCT selection), simulate
            current_possible_words_sim = list(simulation_start_node.state)
            current_attempt_sim = simulation_start_node.attempt
            guess_word = index_to_word[simulation_action]
           
            if not current_possible_words_sim:
                sim_outcome = 0.0
            else:
                secret_word_sim = random.choice(current_possible_words_sim)
                feedback = get_feedback(guess_word, secret_word_sim)
                sim_possible_words = filter_words(current_possible_words_sim, guess_word, feedback)
                sim_attempt = current_attempt_sim + 1

                if guess_word == secret_word_sim:
                    sim_outcome = 1.0 
                elif sim_attempt >= max_attempts:
                    sim_outcome = 0.0 
                else:
                    # Continue rollout using random policy if game isn't over.
                    while sim_attempt < max_attempts and len(sim_possible_words) > 1:
                        rollout_guess_idx = random.choice(list(index_to_word.keys()))
                        rollout_guess = index_to_word[rollout_guess_idx]

                        if rollout_guess == secret_word_sim:
                            sim_outcome = 1.0
                            break

                        feedback = get_feedback(rollout_guess, secret_word_sim)
                        sim_possible_words = filter_words(sim_possible_words, rollout_guess, feedback)
                        sim_attempt += 1
                    else:
                        if sim_attempt < max_attempts and len(sim_possible_words) == 1 and sim_possible_words[0] == secret_word_sim:
                            sim_outcome = 1.0 
                        elif sim_attempt == max_attempts and sim_possible_words and sim_possible_words[0] == secret_word_sim:
                             sim_outcome = 1.0 # Final attempt deduction win.
                        elif sim_attempt == max_attempts and locals().get('rollout_guess') == secret_word_sim:
                             sim_outcome = 1.0 # Win on last guess of rollout loop.
                        else:
                            sim_outcome = 0.0 # Loss (max attempts or >1 possibility left).

            simulation_value = sim_outcome
        else:
            # Simulation started at terminal node with simulation_action -1
            # so value should have been set already based on terminal condition.
            pass


        # Backpropagation
        value_to_propagate = simulation_value
        node_to_update = simulation_start_node
        
        while node_to_update is not None:
            node_to_update.increment_visit()

            if node_to_update.parent is not None:
                parent_node = node_to_update.parent
                action_taken_from_parent = node_to_update.action

                # Update the statistics for the action taken from the parent.
                parent_node.update_action_stats(action_taken_from_parent, value_to_propagate)

                # --- Create Child Node if Expansion Occurred ---
                # If node_to_update was newly reached via a specific feedback, link it.
                # We need the feedback that led to node_to_update.
                # This feedback was generated during the selection/simulation phase *for this iteration*.

                # This linking is complex because the 'true' child depends on the sampled word.
                # A pragmatic approach: If node_to_update was the simulation_start_node
                # and resulted from an expansion (untried action), we don't necessarily
                # create a permanent child link here, as the feedback varies.
                # Alternatively, if selection descended into an existing child based on sampled feedback,
                # that link is already correct.

                # Let's focus on updating stats, child links can be implicitly managed or added if needed.
                # If node_to_update *is* the direct result of parent + action + specific feedback (that we'd need to track),
                # then: parent_node.children[action_taken_from_parent][feedback_leading_to_node] = node_to_update

            # Move up the tree.
            node_to_update = node_to_update.parent


    best_action = -1
    max_visits = -1

    if not root_node.action_stats:
        print("Warning: MCTS root has no action stats after iterations. Choosing random possible word.")
        if root_node.state:
            chosen_word = random.choice(root_node.state)
            best_action = root_node.word_to_index.get(chosen_word, -1)
            if best_action == -1: best_action = random.choice(list(valid_actions_map.values()))
        else:
            best_action = random.choice(list(valid_actions_map.values()))
        return best_action

    sorted_actions = sorted(root_node.action_stats.items(), key=lambda item: item[1]['visits'], reverse=True)

    if sorted_actions:
        best_action = sorted_actions[0][0] 
        max_visits = sorted_actions[0][1]['visits']
        if max_visits == 0:
            # print("Warning: All actions at root have 0 visits after MCTS. Choosing random.")
            best_action = random.choice(list(valid_actions_map.values()))
    else:
        # print("Error: No actions explored in MCTS. Choosing random.")
        best_action = random.choice(list(valid_actions_map.values()))

    return best_action


if __name__ == "__main__":
    WORD_LIST_PATH = 'target_words.txt' 
    MAX_ATTEMPTS = 6
    WORD_LENGTH = 5
    MCTS_ITERATIONS = 1000
    EXPLORATION = 1.414
    
    env = WordleEnvMarkov(WORD_LIST_PATH, max_attempts=MAX_ATTEMPTS, word_length=WORD_LENGTH, render_mode='human')
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    current_possible_words = list(info.get('valid_words', env.valid_words))
    start_time = time.time()

    while not terminated:
        # env.render()
        # print(f"Attempt {env.current_attempt + 1}/{MAX_ATTEMPTS}")
        # print(f"Possible words remaining: {len(current_possible_words)}")
        # if len(current_possible_words) < 15:
        #     print(f"({', '.join(sorted(current_possible_words))})")

        if not current_possible_words:
                print("Error: No possible words left, but game not terminated?")
                break 

        valid_actions_map = env.word_to_index

        action = mcts_possible_words(
            current_possible_words,
            valid_actions_map,
            iterations=MCTS_ITERATIONS,
            exploration_constant=EXPLORATION,
            max_attempts=MAX_ATTEMPTS,
            current_attempt=env.current_attempt 
        )
        suggested_word = env.index_to_word.get(action, 'INVALID ACTION')
        # print(f"MCTS Suggests: {suggested_word} (Index: {action})")

        if action == -1 or suggested_word == 'INVALID ACTION':
                # print("Error: MCTS returned invalid action. Stopping.")
                break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if not terminated:
            guessed_word = info.get('guessed_word')
            feedback = tuple(obs['board'][obs['attempt'][0] - 1])
            if guessed_word and feedback is not None:
                feedback_tuple = tuple(feedback)
                current_possible_words = filter_words(current_possible_words, guessed_word, feedback_tuple)
            else:
                    pass
                    # print("Warning: Environment info missing 'guessed_word' or 'feedback'. Cannot filter words.")

        # print(f"Feedback Received: {feedback}")
        # print(f"Reward this step: {reward:.2f}, Total Reward: {total_reward:.2f}")
        # print("-" * 20)

    # env.render()
    end_time = time.time()
    # print(f"\n--- Game Over ---")
    # print(f"Result: {'Success!' if env.won else 'Failure.'}")
    # print(f"Target Word was: {env.target_word}")
    # print(f"Attempts Used: {env.current_attempt}")
    # print(f"Total Reward: {total_reward:.2f}")
    if not env.won and current_possible_words:
            print(f"Remaining possible words ({len(current_possible_words)}): {', '.join(sorted(current_possible_words)[:10])}{'...' if len(current_possible_words) > 10 else ''}")
    elif not env.won:
            print("Remaining possible words: 0")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
