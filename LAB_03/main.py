# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Configuration
    ngram=5 #Ngram level 
    cutoff=2 
    tests = ["the king is dead", "the emperor is dead", "may the force be with you", "welcome to you", "how are you",  'wasd wasd wasd']
    tests = [sentence.split() for sentence in tests]
    ###########################################################################
    macbeth_sents_c,testsf, lex = prepData(cutoff, tests)
    lm = NLTKsbf(ngram, macbeth_sents_c)
    testNLTKsbf(testsf, lm)
    rep,ng_list = CSTMsbf(ngram, macbeth_sents_c)
    testCSTMsbf(ngram, testsf, rep,ng_list)
    ###########################################################################

    '''
    #DEBUG
    macbeth_ngrams, flat_text = padded_everygram_pipeline(ngram, macbeth_sents_c)
    aa = nltk.lm.NgramCounter(macbeth_ngrams)
    
    print(probSearch(('<UNK>',),rep), lm.score('<UNK>'), aa[('<UNK>')])
    print(probSearch(('<UNK>','<UNK>'),rep), lm.score('<UNK>',('<UNK>',)), aa[('<UNK>','<UNK>')])
    print(probSearch(('<UNK>','<UNK>','<UNK>'),rep), lm.score('<UNK>',('<UNK>','<UNK>')), aa[('<UNK>','<UNK>','<UNK>')])
    
    '''
    '''
    print(probSearch(('is',),rep), lm.score('is'), aa[('is')])
    print(probSearch(('he',),rep), lm.score('he'), aa[('he')])
    print(probSearch(('he','is'),rep), lm.score('is', ('he',)), aa[('he','is')])
    print(probSearch(('i','am'),rep), lm.score('am', ('i',)), aa[('i','am')])
    print(probSearch(('i','will'),rep), lm.score('will', ('i',)), aa[('i','will')])
    print(probSearch(('i','can'),rep), lm.score('can', ('i',)), aa[('i','can')])
    print(probSearch(('i',),rep), lm.score('i'), aa[('i')])
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    