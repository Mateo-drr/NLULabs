# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:11:00 2024

@author: Mateo-drr
"""

import random
import spacy
import string
# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")

avoid = ['hp','pc']

def rmvStopwords(sentence):
   # Tokenize and remove stopwords using spaCy
    doc = nlp(sentence)
    non_stop_words = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
    return non_stop_words
    
def findSynonyms(words,word2vec_model):
    synonyms=[]
    for word in words:
        if word not in avoid:
            try:
                similar_words = word2vec_model.most_similar(word, topn=3)
                #remove words that have _, @ :  and punctuation
                similar_words = [x[0] for x in similar_words if '_' not in x[0] and '@' not in x[0] and ':' not in x[0] and '#' not in x[0] and '.' not in x[0] and x[0] not in string.punctuation]
                synonyms.append([word,random.choice(similar_words)])
            except:
                # Handle the case where the word is not in the vocabulary
                synonyms.append([word,word])
    return synonyms

def replaceWords(synonyms,sentence):
    sent=[]
    words = sentence.lower().split(" ")
    for w in words:
       replaced = False
       for s in synonyms:
           if w == s[0]:
               sent.append(s[1])
               replaced = True
               break  # Break the inner loop once a replacement is found
       if not replaced:
           sent.append(w)

    return ' '.join(sent)

def switch(list1, list2):
    # Ensure lists have the same length
    assert len(list1) == len(list2), "Input lists must have the same length"

    num_pairs = random.randint(1, len(list1) // 2)  # Determine the number of pairs
    available_indices = list(range(len(list1)))  # Include all indices
    indices_to_switch = random.sample(available_indices, num_pairs * 2)  # Multiply by 2 to get pairs

    # Switch the positions of the selected pairs in both lists
    for i in range(0, len(indices_to_switch), 2):
        index1, index2 = indices_to_switch[i], indices_to_switch[i + 1]
        list1[index1], list1[index2] = list1[index2], list1[index1]
        list2[index1], list2[index2] = list2[index2], list2[index1]
    
    return list1, list2

# sentence = "The Best Way To Get Started Is To Quit Talking And Begin Doing ."
# nons = rmvStopwords(sentence)
# syn = findSynonyms(nons)
# end = replaceWords(syn, sentence)
# print(end)