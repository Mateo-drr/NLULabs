# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:38:15 2023

@author: Mateo
"""

import spacy
import nltk

nltk.download("gutenberg")
nltk.download("punkt")

import spacy.cli

# Download the en_core_web_sm model
spacy.cli.download("en_core_web_sm")

# Load the model
nlp = spacy.load("en_core_web_sm")

#LOADING DATA
################################################################################
chars = nltk.corpus.gutenberg.raw('milton-paradise.txt')
words = nltk.corpus.gutenberg.words('milton-paradise.txt')
sents = nltk.corpus.gutenberg.sents('milton-paradise.txt')
################################################################################

#COMPUTE STATISTICS Reference
print('\n- Reference Statistics:')
################################################################################
def statistics(chars, words, sents):

    word_lens = []
    for word in words:
        word_lens.append(len(word))
    
    sent_lens = []
    for sentence in sents:
        sent_lens.append(len(sentence))
    #print(sents[0])
    
    #print(len(sents[1]), sents[1])
    chars_in_sents =[]
    for characters in sents:
        chars_in_sents.append(len("".join(characters)))
    #print(chars_in_sents)
    
    word_per_sent = round(sum(sent_lens) / len(sents))
    char_per_word = round(sum(word_lens) / len(words))
    char_per_sent = round(sum(chars_in_sents) / len(sents))
    
    longest_sentence = max(chars_in_sents)
    l_sentw = max(sent_lens)
    longest_word = max(word_lens)
    
    return word_per_sent, char_per_word, char_per_sent, longest_sentence, longest_word, l_sentw


word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw = statistics(chars, words, sents)


print('Word per sentence', word_per_sent)
print('Char per word', char_per_word)
print('Char per sentence', char_per_sent)
print('Longest sentence (characters)', longest_sent)
print('Longest sentence (words)', l_sentw)
print('Longest word', longest_word)
################################################################################

#COMPUTE STATISTICS Spacy
print('\n- Spacy Statistics:')
################################################################################
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

#COMPUTE STATISTICS NLTK
print('\n- NLTK Statistics:')
################################################################################
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


#LEXICON 
print('\n- Reference Lexicon size and 5 top frequencies:')
################################################################################
def nbest(d, n=1):
    """
    get n max values from a dict
    :param d: input dict (values are numbers, keys are stings)
    :param n: number of values to get (int)
    :return: dict of top n key-value pairs
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

#calculate the frequencies with NLKT
freq_dist = nltk.FreqDist(words)
#get the n most frequent tokens
print('Upper case:', len(freq_dist), 'Top 5:', nbest(freq_dist, n=5)) # Change N form 1 to 5

#put in lower caps
lexicon = ([w.lower() for w in words])
#Calculate freq using counter
l_freq_dist = nltk.FreqDist(lexicon) # Replace X with the word list of the corpus in lower case (see above))

#get the n most frequent tokens
print("Lower case:", len(l_freq_dist), 'Top 5:', nbest(l_freq_dist, n=5)) # Change N form 1 to 5
################################################################################
print('\n- Spacy Lexicon size and 5 top frequencies:')
################################################################################
#calculate the frequencies with NLKT
freq_dist = nltk.FreqDist(wordsS)
#get the n most frequent tokens
print('Upper case:', len(freq_dist), 'Top 5:', nbest(freq_dist, n=5)) # Change N form 1 to 5

#put in lower caps
lexicon = ([w.lower() for w in wordsS])
#Calculate freq using counter
l_freq_dist = nltk.FreqDist(lexicon) # Replace X with the word list of the corpus in lower case (see above))

#get the n most frequent tokens
print("Lower case:", len(l_freq_dist), 'Top 5:', nbest(l_freq_dist, n=5)) # Change N form 1 to 5
################################################################################
print('\n- NLTK Lexicon size and 5 top frequencies:')
################################################################################
#calculate the frequencies with NLKT
freq_dist = nltk.FreqDist(wordsN)
#get the n most frequent tokens
print('Upper case size:', len(freq_dist), 'Top 5:', nbest(freq_dist, n=5)) # Change N form 1 to 5

#put in lower caps
lexicon = ([w.lower() for w in wordsN])
#Calculate freq using counter
l_freq_dist = nltk.FreqDist(lexicon) # Replace X with the word list of the corpus in lower case (see above))

#get the n most frequent tokens
print("Lower case:", len(l_freq_dist), 'Top 5:', nbest(l_freq_dist, n=5)) # Change N form 1 to 5
################################################################################

