# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import os
print(os.getcwd())
# Import everything from functions.py file
from functions import *
import spacy
import nltk

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
    
    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence (characters)', longest_sent)
    print('Longest sentence (words)', l_sentw)
    print('Longest word', longest_word)
    ################################################################################
    
    ################################################################################
    #COMPUTE STATISTICS Spacy
    print('\n- Spacy Statistics:')
    nlp = spacy.load("en_core_web_sm")
    wordsS = nlp(chars)
    sentsS = list(wordsS.sents)
    wordsS = [token.text for token in wordsS]
    sentsS = [sent for sent in sentsS]
    aux = []
    for i in range(0,len(sentsS)):
        for word in sentsS[i]:
            aux.append(word.text)
        sentsS[i] = aux
        aux = []
    
    word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw = statistics(chars, wordsS, sentsS)
    
    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence (characters)', longest_sent)
    print('Longest sentence (words)', l_sentw)
    print('Longest word', longest_word)
    ################################################################################
    
    ################################################################################
    #COMPUTE STATISTICS NLTK
    print('\n- NLTK Statistics:')
    sentsN = nltk.sent_tokenize(chars)
    wordsN = nltk.word_tokenize(chars)
    
    for i in range(0,len(sentsN)):
        #for word in sentsS[i]:
        #aux.append(nltk.word_tokenize(word))
        sentsN[i] = nltk.word_tokenize(sentsN[i])#aux
        #aux = []
    
    word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw = statistics(chars, wordsN, sentsN)
    
    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence (characters)', longest_sent)
    print('Longest sentence (words)', l_sentw)
    print('Longest word', longest_word)
    ################################################################################
    
    
    ################################################################################
    #LEXICON 
    print('\n- Reference Lexicon size and 5 top frequencies:')
    
    #calculate the frequencies with NLKT
    freq_dist = nltk.FreqDist(words)
    #get the n most frequent tokens
    print('Upper case:', len(freq_dist), 'Top 5:', nbest(freq_dist, n=5)) 
    
    #put in lower caps
    lexicon = ([w.lower() for w in words])
    #Calculate freq using counter
    l_freq_dist = nltk.FreqDist(lexicon)
    
    #get the n most frequent tokens
    print("Lower case:", len(l_freq_dist), 'Top 5:', nbest(l_freq_dist, n=5)) 
    ################################################################################
    
    ################################################################################
    print('\n- Spacy Lexicon size and 5 top frequencies:')
    #calculate the frequencies with NLKT
    freq_dist = nltk.FreqDist(wordsS)
    #get the n most frequent tokens
    print('Upper case:', len(freq_dist), 'Top 5:', nbest(freq_dist, n=5)) 
    
    #put in lower caps
    lexicon = ([w.lower() for w in wordsS])
    #Calculate freq using counter
    l_freq_dist = nltk.FreqDist(lexicon) 
    
    #get the n most frequent tokens
    print("Lower case:", len(l_freq_dist), 'Top 5:', nbest(l_freq_dist, n=5)) 
    ################################################################################
    
    ################################################################################
    print('\n- NLTK Lexicon size and 5 top frequencies:')
    #calculate the frequencies with NLKT
    freq_dist = nltk.FreqDist(wordsN)
    #get the n most frequent tokens
    print('Upper case:', len(freq_dist), 'Top 5:', nbest(freq_dist, n=5))
    
    #put in lower caps
    lexicon = ([w.lower() for w in wordsN])
    #Calculate freq using counter
    l_freq_dist = nltk.FreqDist(lexicon)
    
    #get the n most frequent tokens
    print("Lower case:", len(l_freq_dist), 'Top 5:', nbest(l_freq_dist, n=5)) # Change N form 1 to 5
    ################################################################################
