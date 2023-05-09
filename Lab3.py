# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:37:33 2023

@author: Mateo-drr
"""

import nltk
nltk.download("gutenberg")
nltk.download("punkt")

from nltk.lm import Vocabulary
from nltk.corpus import gutenberg
from nltk.util import everygrams
from itertools import chain
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm.models import StupidBackoff
from nltk.lm.preprocessing import padded_everygram_pipeline
import numpy as np
from nltk.tokenize import word_tokenize
import math
from scipy.stats import entropy

test_sents1 = ["the king is dead", "the emperor is dead", "may the force be with you"]
test_sents2 = ["welcome to you", "how are you"]
n = 5

# Load data
hamlet_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-macbeth.txt')]
hamlet_words = flatten(hamlet_sents)
# Compute vocab 
lex = Vocabulary(hamlet_words, unk_cutoff=2)

ngrams, flat_text = padded_everygram_pipeline(n, hamlet_sents)

lm = StupidBackoff(0.4, vocabulary=lex, order=n)
lm.fit(ngrams, flat_text)

test_set = [test_sents1, test_sents2]
hents = []
for i in range(0,2):
    ngrams, flat_text = padded_everygram_pipeline(lm.order, [lex.lookup(sent.split()) for sent in test_set[i]])
    ngrams = chain.from_iterable(ngrams)
    
    hent = []
    test =[[],[], []]
    for x in ngrams:
        if len(x) == lm.order:
            hent.append(x)

    print("Entropy:", lm.entropy(hent))
    print("Perplexity:", lm.perplexity(hent), 2**(lm.entropy(hent)))
    hents.append(hent)

print("Unmasked score:", lm.unmasked_score('you', 'may the force be with'))
print("Masked score:", lm.score('you', 'may the force be with'),"\n")
print(hent)

print(hent[0][0], hent[0][0:-1])
#print(len(test[0]),len(test[1]),len(test[2]))
#print(test)

#Custom implementation
def compute_ppl(model, data):
    highest_ngram = model.order
    scores = [] 
    for sentence in data:
        ngrams, flat_text = padded_everygram_pipeline(highest_ngram, [sentence.split()])
        scores.extend([-1 * model.logscore(w[-1], w[0:-1]) for gen in ngrams for w in gen if len(w) == highest_ngram])
    return math.pow(2.0, np.asarray(scores).mean())
#compute_ppl(mle_lm, test_sents2)    

def score(hent, n_gramsS, counts):

    sentlogprob = []
    for i in range(1,len(test[1])): #n-grams for probs
        sentlogprob[i-1] += np.log2(test[1][i])


    
    score = 0
    sentprob = 0
    probs = []
    #test 0 -> unigrams, 1 -> n-1, 2 -> n
    n=3
    for hent in hents:
        for i in range(0,len(hent)): 
            #for j in range(0,len(hent[0])): #loop each ngram
            if hent[i] in n_gramsS[-1]: #check if ngram is in the corpus
                print('Looking for', hent[i])
                for j in range(0,len(n_gramsS[-1])): #loop the corpus to find the ngram
                    if hent[i] == n_gramsS[-1][j]: 
                        print("Found ngram", hent[i], 'with count:', counts[-1][j])
                        break
                for k in range(0,len(n_gramsS[-2])): #loop the corpus to find context n-1 counts
                    if hent[i][0:-1] == n_gramsS[-2][k]:
                        print("Context",hent[i][0:-1], n_gramsS[-2][k], 'with count:', counts[-2][k])
                        #sum up the log prob of each word given context
                        probs.append(float(counts[-1][j]) / counts[-2][k])
                        score += np.log2(float(counts[-1][j]) / counts[-2][k]) #log p(word|context) = p(context+word)/p(context) = p(5gram)/p(4gram)
                        break            
                sentprob += score #log prob of the whole sentence
                print(np.log2(float(counts[-1][j]) / counts[-2][k]), score, sentprob, '\n')
            score = 0 #reset score

        print("Entropy:", -sentprob/n)
        print("Perplexity:", 2**(-sentprob/n))
        sentprob = 0
        n=2

    #return 2**(-sentprob/5) #divide by the number of sentences


hw = [w.lower() for w in gutenberg.words('shakespeare-macbeth.txt')]

def stupidBoff(corpus, word, context, n=2, dv=0.4):

    #Create the ngrams and sort them into separate lists
    ngrams, flat_text = padded_everygram_pipeline(n, corpus)
    ngrams = chain.from_iterable(ngrams)

    n_grams = [[] for _ in range(n)] #empty lists for each n gram
    for x in ngrams:
        for i in range(1,n+1):
            if len(x) == i:
                n_grams[i-1].append(x)

    n_gramsS = [[] for _ in range(n)] #make copies without repeated ngrams
    for i in range(0,n):
        n_gramsS[i] = list(set(n_grams[i]))
    print(n_gramsS[0][0])

    probs = [[] for _ in range(n)] #empty lists for each n gram
    counts = [[] for _ in range(n)] #empty lists for each n gram
    for i in range(0,n):
        for ng in n_gramsS[i]:
            if i == n-1:
                x = n_grams[i].count(ng) #number of times word appears in each ngram list
                counts[i].append(x)
            else:
                x = n_grams[i].count(ng)
                counts[i].append(x)
                x=x*dv
            probs[i].append(x/len(n_grams[i]))
    
    wprob = []
    
    tcontext = word_tokenize(context +" "+ word)
    tcontext = mask(n_gramsS[0], tcontext)
    
    cngrams = list(everygrams(tcontext, max_len=n, pad_left=False, pad_right=False, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    #cngrams, flat_text = padded_everygram_pipeline(n, [[context]])
    #cngrams = chain.from_iterable(ngrams)
    print(tcontext)
    cn_grams = [[] for _ in range(n)] #empty lists for each n gram
    for x in cngrams:
        for i in range(1,n+1):
            if len(x) == i:
                cn_grams[i-1].append(x)
    
    
    #print(cn_grams)

    cn_grams = [x for x in cn_grams if x] #remove empty lists caused by smaller context input than the n gram size used
    #print(cn_grams)

    #print(n_gramsS[0][0], cn_grams[0][-1], n_gramsS[0][0] == cn_grams[0][-1], cn_grams[0][-1] in n_gramsS[0]) 

    for i in range(1,len(cn_grams)+1): #loop in all n available context+word grams backwards
        ngram = cn_grams[len(cn_grams)- i][-1] #get the context as an ngram
        print('Looking for ngrams equal to', ngram)
        if ngram in n_gramsS[len(cn_grams)-i]: #Check if context+word ngram is in the corpus
            for j in range(0,len(n_gramsS[len(cn_grams)-i])): #loop the corpus
                if ngram == n_gramsS[len(cn_grams)-i][j]:
                    print("Found ngram", ngram, 'with count', counts[len(counts)-i][j])#'with prob:', probs[len(probs)-i][j])
                    #found the ngram so ill check the context ex: found 'be with you' now search 'be with'
                    if len(ngram) !=1: #unigram prob is only a simple count
                        for k in range(0,len(n_gramsS[len(ngram)-2])):
                            if n_gramsS[len(ngram)-2][k] == ngram[0:-1]: #context
                                wprob.append(counts[len(counts)-i][j]/counts[len(ngram)-2][k]) #p(w|c)
                                #if len(cn_grams[-1]) != len(ngram): #check if it's doing backoff 
                                #    wprob[-1] = wprob[-1]*0.4
                    else: 
                        wprob.append(probs[len(probs)-i][j]) #get the probability of the unigram
                    break
    
    
      
    extra = 0
    if len(tcontext) > len(n_grams): #apply backoff to the first probability if context+word is longer than corpus n gram 
        wprob[0]*=dv
        extra =1
        
    finalp = 0
    for i in range(0,len(wprob)):
        if wprob[i] != 0:
            finalp = wprob[i]
            break


    print('\nProbability of word [' + word + '] given context ['+context+']:', finalp, "with backoff level of", len(cn_grams) - len(wprob) +extra)
    #print('Always applying backoff:', np.sum(wprob), '\n')
    #print('Perplexity:', score(hent, n_gramsS, counts))
    return n_gramsS, counts
    
#'''        
def mask(unigrams, sentence):
    
    for i in range(0,len(sentence)):
        ok = False
        for unigram in unigrams:
            if sentence[i] == unigram[0]:
                ok = True
                break
        if not ok:
            sentence[i] = '<UNK>'
    print(sentence)
    return sentence
        
#'''            
                


n_gramsS, counts = stupidBoff( hamlet_sents, 'hh','may the force be hh',n, 0.4)

score(hents, n_gramsS, counts)
#print(hamlet_sents[0:2])

