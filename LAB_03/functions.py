# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import nltk
from nltk.corpus import gutenberg
nltk.download("gutenberg")
nltk.download("punkt")
from nltk.lm.preprocessing import flatten
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
from itertools import chain
import math
import numpy as np
from nltk.lm import StupidBackoff

def prepData(cutoff, tests):
    # Dataset in lowercase
    macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-macbeth.txt')]
    macbeth_words=flatten(macbeth_sents)
    #Create the vocabulary with lvl x cutoff
    lex = Vocabulary(macbeth_words, unk_cutoff=cutoff)
    #Apply cutoff
    macbeth_sents_c = list(lex.lookup(macbeth_sents))
    #Remove unk words from test
    testsf = [list(lex.lookup(sent)) for sent in tests]
    return macbeth_sents_c, testsf, lex

def NLTKsbf(ngram, macbeth_sents_c):
    #Get the ngrams
    macbeth_ngrams, flat_text = padded_everygram_pipeline(ngram, macbeth_sents_c)
    #Fit the model    
    lm = StupidBackoff(0.4, order=ngram)
    lm.fit(macbeth_ngrams, flat_text)
    return lm

def testNLTKsbf(testsf, lm):
    #Calculate perplexity using cstm funct
    scores=[]
    for sent in testsf:
        scores.extend([-1 * lm.logscore(sent[-1], sent[0:-1])]) #last word given all previous ones
    #Print results  
    print('\nNLTK IMPLEMENTATION')              
    print("Entropy:", lm.entropy(testsf))
    print("Perplexity:", lm.perplexity(testsf))
    
def CSTMsbf(ngram, macbeth_sents_c):
    #Get the ngrams
    macbeth_ngrams, flat_text = padded_everygram_pipeline(ngram, macbeth_sents_c)
    macbeth_ngrams = chain.from_iterable(macbeth_ngrams)
    
    #Format them into lists for ease of access
    n_grams = [[] for _ in range(ngram)] #empty lists for each n gram
    for x in macbeth_ngrams:
        for i in range(1,ngram+1):
            if len(x) == i:
                n_grams[i-1].append(x)
    
    #Flip the ordering
    ng_list = n_grams[::-1]
    
    #To find the counts we need a list without repeated ngrams
    ngf_list=[]
    for lvl in ng_list:
        ngf_list.append(list(set(lvl)))
    
    #Iteratively look for the test ngrams in the training set
    score = 0
    rep=[[] for _ in range(ngram)]
    for lvl in range(0,ngram):
        for nlvl1 in ngf_list[lvl]: #loop same ngram lvl non repeating ngrams 
            for nlvl2 in ng_list[lvl]:
                if nlvl1 == nlvl2: #ngram found
                    score+=1
            rep[lvl].append({'ngram': nlvl1, 'score':score})
            score=0
    
    return rep, ng_list

 #Search the score of a given ngram
def probSearch(ngram, text, rep, ng_list):
     word = text
     ogw = text
     score=0
     
     #FIND TIMES NGRAM APPEARS
     #Limit max size to max ngram size
     if len(word) > ngram:
         word = word[-ngram:]
     for lvl in range(ngram-len(word),ngram): #start from the length of the test ngram
         for nlvl1 in rep[lvl]:
             if word == nlvl1['ngram']:
                 score=(nlvl1)
         if score != 0: #ngram was found
             break
         else:
             word = word[1:] #remove one word and search again
             if len(word) == 0: #in case the unigram was not found just break
                 print(text, 'NOT FOUND P=0')
                 return 1e-10
     
     #FIND TIMES CONTEXT APPEARS
     counts=0
     if len(score['ngram']) >1:        
         lvl = ngram-len(score['ngram'])
         for nlvl1 in rep[lvl]:
             if score['ngram'][:-1] == nlvl1['ngram'][:-1]:
                 counts+=nlvl1['score']
         prob = score['score']/counts
     else:
         #Calculate the probability
         prob = score['score']/len(ng_list[ngram-len(score['ngram'])])#len(rep[ngram-len(score['ngram'])])
     
     if len(ogw) > len(score['ngram']):
         bkoff = len(ogw) - len(score['ngram'])
         prob = prob*0.4**bkoff
     
     return prob
    
def testCSTMsbf(ngram,testsf,rep,ng_list):
    #Get the ngrams of the test sentences
    test_s, flat = padded_everygram_pipeline(ngram, testsf)
    test_s = chain.from_iterable(test_s)
    
    #Format them into lists for ease of access
    t_grams = [[] for _ in range(ngram)] #empty lists for each n gram
    for x in test_s:
        for i in range(1,ngram+1):
            if len(x) == i:
                t_grams[i-1].append(x)
    
    #Flip the ordering
    t_list = t_grams[::-1]
    #Remove duplicate ngrams;
    tf_list=[]
    for lvl in t_list:
        tf_list.append(list(set(lvl)))
    
    #Loop the ngrams (only the biggest lvl since the function reduces the ngram length automatically)
    perplexity = 1.0
    count = 0
    logscores=[]
    for text in testsf:
        prob = probSearch(ngram,tuple(text),rep,ng_list)
        logscore = math.log(prob,2)
        logscores.append(-1*logscore)
        
    perplexity = math.pow(2.0, np.array(logscores).mean())
    
    #PRINT RESULTS
    print('\nCUSTOM IMPLEMENTATION:')
    print("Entropy:", np.array(logscores).mean())
    print("Perplexity:", perplexity)
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 