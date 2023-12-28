# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    
    '''
    #Mapping NLTK and SPACY
    mapping_spacy_to_NLTK = {
        'ADJ': 'JJ',
        'ADP': 'IN',
        'ADV': 'RB',
        'AUX': 'MD',
        'CONJ': 'CC',
        'DET': 'DT',
        'INTJ': 'UH',
        'NOUN': 'NN',
        'NUM': 'CD',
        'PART': 'RP',
        'PRON': 'PRP',
        'PROPN': 'NNP',
        'PUNCT': '.',
        'SCONJ': 'IN',
        'SYM': 'LS',
        'VERB': 'VB',
        'X': 'FW',
        'SPACE': 'SP'
    }
    '''
    #This other mapping gives the same result
    
    mapping_spacy_to_NLTK = {
        "ADJ": "ADJ",
        "ADP": "ADP",
        "ADV": "ADV",
        "AUX": "VERB",
        "CCONJ": "CONJ",
        "DET": "DET",
        "INTJ": "X",
        "NOUN": "NOUN",
        "NUM": "NUM",
        "PART": "PRT",
        "PRON": "PRON",
        "PROPN": "NOUN",
        "PUNCT": ".",
        "SCONJ": "CONJ",
        "SYM": "X",
        "VERB": "VERB",
        "X": "X"
    }
    
    
    #Load spacy and nltk data, including the sentences 
    sents, usents, nlp = loadData()
    #Get the trained data tagged for spacy
    spacy_train_sents = spacyTag(usents, nlp, mapping_spacy_to_NLTK)
    #test with the untagged sentences
    nk, sp = testTag(sents, usents[idx:], spacy_train_sents)
    #print accurcy results
    res(nk, sp)
    
    #Running spacy without nltk
    print('NLTK best ACC:', max(max(nk)), 'vs SPACY ACC:',spacyAcc(spacy_train_sents))
    
