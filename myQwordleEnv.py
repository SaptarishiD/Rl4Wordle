# <myQwordleEnv.py>
import pandas as pd
import numpy as np
import gymnasium as gym
import random




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

        # these 3 are for each row
        self.row_correct = [None]*self.word_len # greens
        self.row_present = [None]*self.word_len # yellows
        self.row_absent = [None]*self.word_len  # blacks

        self.win = ''



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
                self.letters_present.append(target_letter)
                self.pos_guessed_correctly[i] = target_letter

            # yellow
            elif guessed_letter in word and guessed_letter != target_letter:
                self.row_present[i] = guessed_letter
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
            self.win = True
            # print(f"YOU WON! In {self.attempts} moves")
            return number_of_greens, number_of_yellows, number_of_blacks
        
       

        if self.attempts == self.max_attempts:
            self.win = False
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
        return random.choice(filtered)
        
    


class WordleMetaEnv():
    def __init__(self):
        self.win_reward = 10
        self.lose_cost = -10
        self.green_reward = 5
        self.yellow_reward = 3
        self.black_cost = -1

        self.total_reward = 0
        self.action_space = [0,1,2]
    
    def reset(self):
        self.agent = MyAgent()
        self.env = WordleQEnv()
        self.guesses_made = 0

        return (0,0,0)  # corresponding to greens and yellows
    
    def step(self, action):
        self.guesses_made += 1

        if action == 0:
            guess = self.agent.randomly()
        elif action == 1:
            guess = self.agent.rand_not_absent(self.env.letters_absent)
        elif action == 2:
            guess = self.agent.rand_green_not_absent(self.env.pos_guessed_correctly, self.env.letters_absent)
        
        greens, yellows, blacks = self.env.make_guess(guess)
        
        reward = self.green_reward * greens + self.yellow_reward * yellows - self.black_cost * blacks

        state = (greens, yellows, blacks)
        
        
        if self.env.win:
            reward += self.win_reward
            return state, reward, True
        elif not self.env.win:
            reward -= self.lose_cost
            return state, reward, True

        return state, reward, False



env = WordleQEnv()
print(f"Target Word: {env.target_word}")
env.make_guess("chase")


agent = MyAgent()
print(agent.rand_green_not_absent([None, None,None, None, 's'], []))



# </myQwordleEnv.py>