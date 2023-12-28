# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions

    
    
    # test setenteces, changed some words so they are in the treebank vocab
    test_sents = [
        "there is a plane stuck in a tree", #"There is a rabbit stuck in a tree", 
        "I want to return it to the wild so it can lay its things",#"I want to return it to the wild so it can lay its eggs",
        "I can not control the weather",
        ]
    
    #get grammar from treebank
    grammar = getGrammar()
    
    #get custom grammar
    cstm_grammar = cstmGrammar()
    
    #print tree from grammar
    print('---------------------------------------')
    printTrees(grammar,test_sents)
    
    #print tree from cstm grammar
    print('------------------cstm-----------------')
    printTrees(cstm_grammar,test_sents)
    
    #Generate sentences
    print('\nGenerated sentences using nltk.parse.generate:')
    for sent in generate(cstm_grammar, depth=10, n=3):
        print(sent)

    print('\nGenerated sentences using PCFG:')
    cstm_grammar2 = PCFG.fromstring(weighted_rules)
    for sent in cstm_grammar2.generate(10):
        print(sent)
