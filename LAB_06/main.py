# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    # Spacy version 
    from nltk.parse import DependencyEvaluator
    
    #Initialize the data, spacy and stanza
    stanza, spcy, sents, psents = ini()
    
    #Lists for the joined sentences into one string
    jsents = []
    stz = []
    spy = []

    dpz = []
    dpy = [] 
    
    #Get dependency graphs
    dpy,dpz = depGraphs(stanza, spcy, sents, jsents, stz, spy, dpz, dpy)
    
    de = DependencyEvaluator(dpy, psents)
    las, uas = de.eval()
    print('\nSpacy: ')
    print("LAS:", las)
    print("UAS:", uas)
    de = DependencyEvaluator(dpz, psents)
    las, uas = de.eval()
    print("\nStanza: ")
    print("LAS:", las)
    print("UAS:", uas)