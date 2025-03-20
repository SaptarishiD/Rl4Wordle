from collections import defaultdict
def filter_words(word_list, guessed_word, feedback, word_length=5):
    """
    Filter the word list based on the feedback from a guess.
    
    Args:
        word_list (list): List of possible words
        guessed_word (str): The word that was guessed
        feedback (list): List of feedback codes (0=absent, 1=present, 2=correct)
        word_length (int): Length of the words
        
    Returns:
        list: Filtered word list
    """
    filtered = []
    
    # Count the occurrences of each letter in the guessed word
    letter_counts = defaultdict(int)
    for i, letter in enumerate(guessed_word):
        letter_counts[letter] += 1
    
    # Determine minimum occurrences for each letter based on correct and present feedback
    min_occurrences = defaultdict(int)
    for i, letter in enumerate(guessed_word):
        if feedback[i] in [1, 2]:
            min_occurrences[letter] += 1
    
    # Determine maximum occurrences for each letter
    max_occurrences = {}
    for letter, count in letter_counts.items():
        # Count how many times the letter got 'absent' feedback
        absent_count = sum(1 for i, l in enumerate(guessed_word) 
                          if l == letter and feedback[i] == 0)
        
        if absent_count > 0:
            # If letter appears as absent, max occurrences = number of non-absent occurrences
            max_occurrences[letter] = count - absent_count
        else:
            # If letter never appears as absent, no upper bound
            max_occurrences[letter] = float('inf')
    
    for word in word_list:
        if len(word) != word_length:
            continue
        
        is_valid = True
        
        word_letter_counts = defaultdict(int)
        for letter in word:
            word_letter_counts[letter] += 1
        
        for letter, min_count in min_occurrences.items():
            if word_letter_counts[letter] < min_count:
                is_valid = False
                break
        
        for letter, max_count in max_occurrences.items():
            if word_letter_counts[letter] > max_count:
                is_valid = False
                break
        
        for i, letter in enumerate(guessed_word):
            if feedback[i] == 2:  # Correct position
                if word[i] != letter:
                    is_valid = False
                    break
            elif feedback[i] == 1:  # Present but wrong position
                if word[i] == letter:
                    is_valid = False
                    break
            elif feedback[i] == 0:  # Absent
                # Only check position if the letter is allowed to appear elsewhere
                if max_occurrences[letter] == 0 and letter in word:
                    is_valid = False
                    break
        
        if is_valid:
            filtered.append(word)
    
    return filtered


def convert_feedback_to_wordle_format(feedback):
    """Convert numeric feedback to Wordle format for rendering"""
    result = []
    for code in feedback:
        if code == 2:
            result.append("🟩")  # Correct (green)
        elif code == 1:
            result.append("🟨")  # Present (yellow)
        else:
            result.append("⬛")  # Absent (gray)
    return "".join(result)
