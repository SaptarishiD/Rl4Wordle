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

        # self.correct

    def make_guess(self, word):
        self.attempts += 1
        if word == self.target_word:
            print("\n ============= You Won =========== \n")




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
        if self.target_word == word:
            print("YOU WON!")
            return
        
        number_of_greens = len([x for x in self.row_correct if x is not None])
        number_of_yellows = len([x for x in self.row_present if x is not None])
        number_of_blacks = len([x for x in self.row_absent if x is not None])


        if self.attempts == self.max_attempts:
            print("ATTEMPTS FINISHED!")
            return 

        return number_of_greens, number_of_yellows, number_of_blacks
    

class MyAgent():
    def __init__(self, word_list_path):
        self.agent_guesses = []
        self.agent_attempts = 0
        with open(word_list_path, 'r') as f:
            self.word_list = f.read().splitlines()
        
    def randomly(self):
        rand_word = random.choice(self.word_list)
        while (rand_word in self.agent_guesses):
            rand_word = random.choice(self.word_list)





        










env = WordleQEnv()
print(f"Target Word: {env.target_word}")
env.make_guess("chase")



