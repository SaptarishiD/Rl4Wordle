import pickle
import numpy as np
import random
import time
from myQwordleEnv import WordleMetaEnv, WordleQEnv, MyAgent

class InteractiveWordleDemo:
    def __init__(self, q_table_path='Q_table_no_intermediate_targets.pkl', word_list_path='target_words.txt'):
        # Load the Q-table
        with open(q_table_path, 'rb') as f:
            self.Q = pickle.load(f)
        
        # Initialize environment
        self.meta_env = WordleMetaEnv(debug=False, word_list_path=word_list_path)
        self.env = self.meta_env.env
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
    
    def reset(self, target_word=None):
        """Reset the game with optional specific target word"""
        self.meta_env.reset()
        
        if target_word:
            if target_word in self.word_list:
                self.env.target_word = target_word
            else:
                print(f"'{target_word}' not in word list. Using random target word.")
        
        self.state = (0, 0, 0)  # Initial state (0 greens, 0 yellows, 0 blacks)
        self.done = False
        self.moves = 0
        
        print(f"Game initialized! (Target word hidden)")
        print("=" * 50)
    
    def get_action(self):
        """Get the best action from Q-table or random if state not in Q-table"""
        if self.state in self.Q:
            return np.argmax(self.Q[self.state])
        else:
            return random.choice(range(6))
    
    def visualize_guess(self, guess_word):
        """Visualize the guess with colored feedback"""
        result = ""
        for i, (guessed_letter, target_letter) in enumerate(zip(guess_word, self.env.target_word)):
            if guessed_letter == target_letter:
                # Green - correct letter in correct position
                result += f"\033[42m {guessed_letter.upper()} \033[0m"
            elif guessed_letter in self.env.target_word:
                # Yellow - correct letter in wrong position
                result += f"\033[43m {guessed_letter.upper()} \033[0m"
            else:
                # Gray - letter not in word
                result += f"\033[47m\033[30m {guessed_letter.upper()} \033[0m"
        
        return result
    
    def print_keyboard(self):
        """Print the keyboard with color-coded feedback"""
        keyboard = [
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm"
        ]
        
        print("\nKeyboard Status:")
        for row in keyboard:
            row_display = ""
            for letter in row:
                if letter in self.env.letters_correct:
                    # Green
                    row_display += f"\033[42m {letter.upper()} \033[0m"
                elif letter in self.env.letters_present:
                    # Yellow
                    row_display += f"\033[43m {letter.upper()} \033[0m"
                elif letter in self.env.letters_absent:
                    # Gray
                    row_display += f"\033[47m\033[30m {letter.upper()} \033[0m"
                else:
                    # Unused
                    row_display += f" {letter.upper()} "
            print(row_display)
    
    def step(self):
        """Take one step in the game"""
        if self.done:
            print("Game is already over. Please reset to play again.")
            return
        
        # Get action from Q-table
        action = self.get_action()
        action_name = self.action_names[action]
        
        print(f"Step {self.moves + 1}: Using strategy: {action_name}")
        
        # Get the guess word based on the action
        if action == 0:
            guess = self.agent.randomly()
        elif action == 1:
            guess = self.agent.rand_not_absent(self.env.letters_absent)
        elif action == 2:
            guess = self.agent.rand_green_not_absent(self.env.pos_guessed_correctly, self.env.letters_absent)
        elif action == 3:
            guess = self.agent.letter_frequency_guess()
        elif action == 4:
            guess = self.agent.smart_guess(green_positions=self.env.pos_guessed_correctly, yellows=self.env.letters_present, absent_letters=self.env.letters_absent)
        elif action == 5:
            guess = self.agent.yellow_position_tracking(yellows=self.env.letters_present, absent_letters=self.env.letters_absent, yellow_positions=self.env.pos_yellow, green_positions=self.env.pos_guessed_correctly)
        
        # Add guess to agent's list of guesses
        self.agent.agent_guesses.append(guess)
        
        # Make the guess
        greens, yellows, blacks = self.env.make_guess(guess)
        
        # Update state
        next_state = (greens, yellows, blacks)
        self.state = next_state
        self.moves += 1
        
        # Visualize the guess
        print(f"Guess: {self.visualize_guess(guess)}")
        print(f"Feedback: {greens} greens, {yellows} yellows, {blacks} blacks")
        
        # Print keyboard status
        self.print_keyboard()
        
        # Check if game is done
        if self.env.won_game == 'yes':
            self.done = True
            print(f"\n🎉 GAME WON! The word was '{self.env.target_word.upper()}'")
            print(f"Solved in {self.moves} guesses")
        elif self.env.won_game == 'no':
            self.done = True
            print(f"\n❌ GAME LOST! The word was '{self.env.target_word.upper()}'")
        
        return self.done
    
    def reveal_answer(self):
        """Reveal the target word"""
        print(f"The target word is: {self.env.target_word.upper()}")
    
    def run_full_game(self, delay=1.0):
        """Run the full game automatically with delay between steps"""
        while not self.done:
            self.step()
            if not self.done:
                time.sleep(delay)
            print("-" * 50)

# Demo usage
def main():
    print("Welcome to the Interactive Wordle Q-Learning Demo!")
    print("=" * 50)
    
    # Initialize the demo
    try:
        demo = InteractiveWordleDemo()
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
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            demo.reset()
        elif choice == '2':
            target = input("Enter a 5-letter word as target: ").lower()
            if len(target) != 5:
                print("Word must be 5 letters long!")
            else:
                demo.reset(target_word=target)
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