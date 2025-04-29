import pickle
import numpy as np
import random
import time
from my_multi_wordle_env_2 import WordleMetaEnvMulti, WordleQEnv, MyAgent

class InteractiveWordleDemoMulti:
    def __init__(self,
                 q_table_path='Q_table_multi_wordle_good.pkl',
                 word_list_path='target_words.txt'):
        # Load the Q-table
        with open(q_table_path, 'rb') as f:
            self.Q = pickle.load(f)
        
        # Initialize multi-wordle environment
        self.meta_env = WordleMetaEnvMulti(debug=False, word_list_path=word_list_path)
        self.env1 = self.meta_env.env1
        self.env2 = self.meta_env.env2
        self.agent = self.meta_env.agent
        
        # Load word list
        with open(word_list_path, 'r') as f:
            self.word_list = f.read().splitlines()
        
        # Action names for display
        self.action_names = {
            0: "Random guess",
            1: "Random guess (avoiding absent letters)",
            2: "Random guess (using green positions, avoiding absent letters)",
            3: "Letter frequency-based guess",
            4: "Smart guess (using greens, yellows, and absent letters)",
            5: "Yellow position tracking guess"
        }

    def reset(self, target1=None, target2=None):
        """Reset both games with optional specific target words"""
        self.meta_env.reset()
        if target1:
            if target1 in self.word_list:
                self.env1.target_word = target1
            else:
                print(f"'{target1}' not in word list for game 1. Using random word.")
        if target2:
            if target2 in self.word_list:
                self.env2.target_word = target2
            else:
                print(f"'{target2}' not in word list for game 2. Using random word.")
        
        self.state = (0, 0, 0, 0, 0, 0)
        self.done = False
        self.moves = 0
        
        print("Dual Wordle Game initialized! Targets hidden for both games.")
        print("=" * 60)

    def get_action(self):
        """Select best action from Q-table or random if unseen state"""
        return np.argmax(self.Q.get(self.state, np.ones(len(self.action_names)) / len(self.action_names)))

    def select_guess(self, action):
        """Replicates WordleMetaEnvMulti guess-selection logic"""
        if action == 0:
            return self.agent.randomly()
        elif action == 1:
            # choose one env randomly
            chosen = random.choice([self.env1, self.env2])
            return self.agent.rand_not_absent(chosen.letters_absent)
        elif action == 2:
            chosen = random.choice([self.env1, self.env2])
            return self.agent.rand_green_not_absent(chosen.pos_guessed_correctly,
                                                   chosen.letters_absent)
        elif action == 3:
            return self.agent.letter_frequency_guess()
        elif action == 4:
            chosen = random.choice([self.env1, self.env2])
            return self.agent.smart_guess(
                green_positions=chosen.pos_guessed_correctly,
                yellows=chosen.letters_present,
                absent_letters=chosen.letters_absent
            )
        elif action == 5:
            chosen = random.choice([self.env1, self.env2])
            return self.agent.yellow_position_tracking(
                yellows=chosen.letters_present,
                absent_letters=chosen.letters_absent,
                yellow_positions=chosen.pos_yellow,
                green_positions=chosen.pos_guessed_correctly
            )

    def visualize_guess(self, guess_word, target_word):
        """Return colored feedback string for a single game"""
        result = ""
        for g, t in zip(guess_word, target_word):
            if g == t:
                result += f"\033[42m {g.upper()} \033[0m"
            elif g in target_word:
                result += f"\033[43m {g.upper()} \033[0m"
            else:
                result += f"\033[47m\033[30m {g.upper()} \033[0m"
        return result

    def print_keyboard(self, env, label):
        """Print keyboard status for one game"""
        keyboard = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        print(f"\nKeyboard Status ({label}):")
        for row in keyboard:
            line = ""
            for l in row:
                if l in env.letters_correct:
                    line += f"\033[42m {l.upper()} \033[0m"
                elif l in env.letters_present:
                    line += f"\033[43m {l.upper()} \033[0m"
                elif l in env.letters_absent:
                    line += f"\033[47m\033[30m {l.upper()} \033[0m"
                else:
                    line += f" {l.upper()} "
            print(line)

    def step(self):
        """Perform one action across both games and display feedback"""
        if self.done:
            print("Game is over. Reset to play again.")
            return

        action = self.get_action()
        print(f"Move {self.moves+1}: Strategy -> {self.action_names[action]}")
        guess = self.select_guess(action)
        self.agent.agent_guesses.append(guess)

        # Apply guess to both games
        g1, y1, b1 = self.env1.make_guess(guess)
        g2, y2, b2 = self.env2.make_guess(guess)
        self.state = (g1, y1, b1, g2, y2, b2)
        self.moves += 1

        # Display results
        print(f"Guess: {guess.upper()}")
        print("-- Game 1 --")
        print(self.visualize_guess(guess, self.env1.target_word),
              f"-> {g1}G {y1}Y {b1}B")
        self.print_keyboard(self.env1, label="Game 1")
        print("-- Game 2 --")
        print(self.visualize_guess(guess, self.env2.target_word),
              f"-> {g2}G {y2}Y {b2}B")
        self.print_keyboard(self.env2, label="Game 2")

        # Check terminal condition
        done1 = self.env1.won_game in ['yes','no']
        done2 = self.env2.won_game in ['yes','no']
        if done1 and done2:
            self.done = True
            # report outcomes
            outcome1 = 'WON' if self.env1.won_game=='yes' else 'LOST'
            outcome2 = 'WON' if self.env2.won_game=='yes' else 'LOST'
            print(f"\n🏁 Both games finished: Game1 {outcome1}, Game2 {outcome2}")
            print(f"Total moves: {self.moves}")
        print("="*60)
        return self.done

    def run_full(self, delay=1.0):
        while not self.done:
            self.step()
            if not self.done:
                time.sleep(delay)

# Example interactive loop omitted; integrate similar menu logic as before but instantiate InteractiveWordleDemoMulti


# Demo usage
def main():
    print("Welcome to the Interactive Wordle Q-Learning Demo!")
    print("=" * 50)
    
    # Initialize the demo
    try:
        demo = InteractiveWordleDemoMulti()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the Q-table file and word list file exist in the current directory.")
        return
    
    # Main interaction loop
    while True:
        print("\nMenu:")
        print("1. Start new game (random word)")
        print("2. Start new game (specific word)")
        print("3. Take one step")
        # print("4. Run full game automatically")
        # print("5. Reveal target word")
        # print("6. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            demo.reset()
        elif choice == '2':
            target1 = input("Enter a 5-letter word as target 1: ").lower()
            target2 = input("Enter a 5-letter word as targe 2: ").lower()
            if len(target1) != 5:
                print("Word must be 5 letters long!")
            else:
                demo.reset(target1=target1, target2=target2)
        elif choice == '3':
            demo.step()
        elif choice == '4':
            try:
                delay = float(input("Enter delay between steps (seconds): "))
                demo.run_full_game(delay=delay)
            except ValueError:
                print("Invalid delay value. Using default 1.0 second.")
                demo.run_full_game()
        elif choice == '5':
            demo.reveal_answer()
        elif choice == '6':
            print("Thank you for using the Wordle Q-Learning Demo!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()