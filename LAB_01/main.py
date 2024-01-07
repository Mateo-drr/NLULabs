# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import os
print(os.getcwd())
# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # -*- coding: utf-8 -*-
    """
    Created on Wed Mar  8 15:38:15 2023
    
    @author: Mateo
    """
    #LOAD DATA
    chars,words,sents = loadData()
    
    ################################################################################
    #COMPUTE STATISTICS Reference
    print('\n- Reference Statistics:')
    word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw = statistics(chars, words, sents)
    printStat(word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw)
    ################################################################################
    
    ################################################################################
    #COMPUTE STATISTICS Spacy
    print('\n- Spacy Statistics:')
    wordsS, sentsS = spacytknz(chars)
    word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw = statistics(chars, wordsS, sentsS)
    printStat(word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw)
    ################################################################################
    
    ################################################################################
    #COMPUTE STATISTICS NLTK
    print('\n- NLTK Statistics:')
    wordsN, sentsN = nltktknz(chars)
    word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw = statistics(chars, wordsN, sentsN)    
    printStat(word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw)
    ################################################################################
    
    ################################################################################
    #LEXICON 
    print('\n- Reference Lexicon size and 5 top frequencies:')
    topFreq(words)
    ################################################################################
    
    ################################################################################
    print('\n- Spacy Lexicon size and 5 top frequencies:')
    topFreq(wordsS)
    ################################################################################
    
    ################################################################################
    print('\n- NLTK Lexicon size and 5 top frequencies:')
    topFreq(wordsN)
    ################################################################################
