import pandas as pd
import numpy as np
import gymnasium as gym
import random
from collections import defaultdict




class WordleQEnv():
    def __init__(self, word_list_path='target_words.txt'):
        with open(word_list_path, 'r') as f:
            self.word_list = f.read().splitlines()
        self.attempts = 0
        self.max_attempts = 6
        self.word_len = 5
        self.target_word = random.choice(self.word_list)
        self.guessed_words = []

        self.letters_correct = [] # keep track of the letters guessed correctly
        self.letters_present = [] # keep track of the letters present in the target word from the guesses i.e. yellows
        self.letters_absent = [] # keep track of the letters absent in the target word from the guesses
        self.pos_guessed_correctly = [None]*self.word_len # keep track of the positions guessed correctly for the whole board
        self.pos_yellow = defaultdict(list)

        # these 3 are for each row
        self.row_correct = [None]*self.word_len # greens
        self.row_present = [None]*self.word_len # yellows
        self.row_absent = [None]*self.word_len  # blacks

        self.won_game = ''

        # self.correct

    def make_guess(self, word):
        self.attempts += 1
        # if word == self.target_word:
            # print("\n ============= You Won =========== \n")

        self.row_correct = [None]*self.word_len 
        self.row_present = [None]*self.word_len # yellows
        self.row_absent = [None]*self.word_len  # blacks

        self.guessed_words.append(word)

        for i, (guessed_letter, target_letter) in enumerate(zip(word, self.target_word)):
            # green
            if guessed_letter == target_letter:
                self.row_correct[i] = target_letter
                self.letters_correct.append(target_letter)
                # self.letters_present.append(target_letter)
                self.pos_guessed_correctly[i] = target_letter

            # yellow
            elif guessed_letter in self.target_word and guessed_letter != target_letter:
                self.row_present[i] = guessed_letter
                if i not in self.pos_yellow[guessed_letter]:
                    self.pos_yellow[guessed_letter].append(i) # this is a dict where vals are list
                if guessed_letter not in self.letters_present:
                    self.letters_present.append(guessed_letter)
            
            else:
                self.row_absent[i] = guessed_letter
                if guessed_letter not in self.letters_absent:
                    self.letters_absent.append(guessed_letter)

        number_of_greens = len([x for x in self.row_correct if x is not None])
        number_of_yellows = len([x for x in self.row_present if x is not None])
        number_of_blacks = len([x for x in self.row_absent if x is not None])

        if self.target_word == word:
            self.won_game = 'yes'
            # print(f"YOU WON! In {self.attempts} moves")
            return number_of_greens, number_of_yellows, number_of_blacks
        
       

        if self.attempts == self.max_attempts:
            self.won_game = 'no'
            self.attempts = 7
            # print("ATTEMPTS FINISHED!")
            return number_of_greens, number_of_yellows, number_of_blacks

        return number_of_greens, number_of_yellows, number_of_blacks
    




class MyAgent():
    def __init__(self, word_list_path='target_words.txt'):
        self.agent_guesses = []
        self.agent_attempts = 0
        with open(word_list_path, 'r') as f:
            self.word_list = f.read().splitlines()
        
    def randomly(self):
        rand_word = random.choice(self.word_list)
        while (rand_word in self.agent_guesses):
            rand_word = random.choice(self.word_list)
        return rand_word

    def rand_not_absent(self, absent_letters):
        new_search_space = [word for word in self.word_list if not any(letter in word for letter in absent_letters)]
        if len(new_search_space) < 1:
            return self.randomly()
        rand_word = random.choice(new_search_space)
        while (rand_word in self.agent_guesses):
            rand_word = random.choice(new_search_space)
        return rand_word
    
    def rand_green_not_absent(self, green_positions, absent_letters):
        filtered = []
        new_search_space = [word for word in self.word_list if not any(letter in word for letter in absent_letters)]
        for word in new_search_space:
            match = True
            for i, letter in enumerate(green_positions):
                if letter is not None and word[i] != letter:
                    match = False
                    break
            if match:
                filtered.append(word)
        # print(filtered)
        # print(len(filtered))
        if self.debug:
            print("Filtered list in rand green not absent")
            print(len(filtered))
        if len(filtered) < 1:
            return self.rand_not_absent(absent_letters)
        return random.choice(filtered)
    

    def smart_guess(self, green_positions, yellows, absent_letters):

        # remove words with absent letters
        candidates = [word for word in self.word_list if not any(letter in word for letter in absent_letters)]
        
        # get the words with green pos
        filtered = []
        for word in candidates:
            match = True
            for i, letter in enumerate(green_positions):
                if letter is not None and word[i] != letter:
                    match = False
                    break
            if match:
                filtered.append(word)
        
        if not filtered:
            if candidates:
                return random.choice(candidates)
            return self.randomly()
        
        candidates = filtered
        
        # now check for yellows
        if yellows:
            yellow_filtered = []
            for word in candidates:
                if all(yellow in word for yellow in yellows):
                    valid = True
                    for i, letter in enumerate(green_positions):
                        if letter is None and i < len(word) and word[i] in yellows:
                            # this pos has a yellow letter so shouldnt be in our search space
                            valid = False
                            break
                    if valid:
                        yellow_filtered.append(word)
            
            if yellow_filtered:
                candidates = yellow_filtered
        
        if len(candidates) > 1:

            word_scores = []
            for word in candidates:

                if word in self.agent_guesses:
                    continue
                    
                unique_letters = len(set(word))
            
            if not word_scores:
                return random.choice(candidates)
            
            # we want to prioritize words with unique letters so that search space is reduced more
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            return word_scores[0][0]
        

        candidates = [word for word in candidates if word not in self.agent_guesses]
        if not candidates:
            candidates = filtered
        
        return random.choice(candidates)

    def letter_frequency_guess(self):


        common_letters = "etaoins"
        
        word_scores = []
        for word in self.word_list:
            if word in self.agent_guesses:
                continue
                
            unique_letters = set(word)
            score = sum(1 for letter in unique_letters if letter in common_letters)
            
            uniqueness_bonus = len(unique_letters)
            
            word_scores.append((word, score + 0.1 * uniqueness_bonus))
        
        if not word_scores:
            return self.randomly()
        
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores[0][0]


    def yellow_position_tracking(self, green_positions, yellows, absent_letters, yellow_positions):
        # this additionally uses information about the positions of yellows. This has a stricter, correct interpretation of yellow feedback

        candidates = [word for word in self.word_list if not any(letter in word for letter in absent_letters)]
        
        filtered = []
        for word in candidates:
            match = True
            for i, letter in enumerate(green_positions):
                if letter is not None and word[i] != letter:
                    match = False
                    break
            if match:
                filtered.append(word)
        
        if not filtered:
            if candidates:
                return random.choice(candidates)
            return self.randomly()
        
        candidates = filtered
        

        if yellows:
            yellow_filtered = []
            for word in candidates:

                if all(yellow in word for yellow in yellows):
                    valid = True
                    
                    # check that yellows aren't in locations where we've already guessed that letter
                    for letter, positions in yellow_positions.items():
                        for pos in positions:
                            if pos < len(word) and word[pos] == letter:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        yellow_filtered.append(word)
            
            if yellow_filtered:
                candidates = yellow_filtered
        
        if len(candidates) > 1:
            word_scores = []
            for word in candidates:
                novelty = 0 if word in self.agent_guesses else 2
                unique_letters = len(set(word))
                
                word_scores.append((word, unique_letters + novelty))
            
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            if word_scores:
                return word_scores[0][0]
        
        candidates = [word for word in candidates if word not in self.agent_guesses]
        if not candidates:
            candidates = filtered
        
        return random.choice(candidates)
         


class WordleMetaEnv():
    def __init__(self, debug=False, word_list_path='target_words.txt'):
        self.won_game_reward = 10
        self.lose_cost = -10
        self.green_reward = 5
        self.yellow_reward = 3
        self.black_cost = -1
        self.debug = debug

        self.total_reward = 0
        self.action_space = [0,1,2,3,4,5]

        self.word_path = word_list_path
    
    def reset(self):
        self.agent = MyAgent(debug=self.debug, word_list_path=self.word_path)
        self.env = WordleQEnv(word_list_path=self.word_path)
        self.guesses_made = 0

        return (0,0,0)  # corresponding to greens and yellows and blacks
    
    def step(self, action):
        self.guesses_made += 1

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
            guess = self.agent.yellow_position_tracking(yellows=self.env.letters_present, absent_letters=self.env.letters_absent,yellow_positions=self.env.pos_yellow, green_positions=self.env.pos_guessed_correctly)
        
        greens, yellows, blacks = self.env.make_guess(guess)
        reward = 0
        reward = self.green_reward * greens + self.yellow_reward * yellows + self.black_cost * blacks

        state = (greens, yellows, blacks)
        
        
        if self.env.won_game == 'yes':
            reward += self.won_game_reward
            return state, reward, True
        elif self.env.won_game == 'no':
            reward += self.lose_cost
            return state, reward, True

        return state, reward, False