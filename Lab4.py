# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:27:14 2023

@author: Mateo-drr
"""

# See abore for further details
'''mapping_spacy_to_NLTK = {
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
#'''

import nltk
nltk.download('treebank')

from nltk.corpus import treebank
import spacy
from spacy.tokenizer import Tokenizer
from nltk.tag import NgramTagger, untag
from nltk.metrics import accuracy
import spacy.cli
from itertools import chain

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
# We overwrite the spacy tokenizer with a custom one, that split by whitespace only
nlp.tokenizer = Tokenizer(nlp.vocab) # Tokenize by whitespace
# Sanity check
for id_sent, sent in enumerate(treebank.sents()):
    doc = nlp(" ".join(sent))
    if len([x.text for x in doc]) != len(sent):
        print(id_sent, sent)
        
print([x.text for x in nlp("we don't do that.")])


n=3

sents = treebank.tagged_sents()
usents = treebank.sents()

ngram_tagger = NgramTagger(n, train=sents, cutoff=0)
tsents = ngram_tagger.tag_sents(usents) #tag the words
#ngram_accuracy = accuracy(sents, tsents) list(chain.from_iterable(sents))
ngram_accuracy = accuracy(list(chain.from_iterable(sents)), list(chain.from_iterable(tsents)))
print("NLTK: not flattened:", accuracy(sents, tsents))
print("NLTK: Accuracy of NgramTagger:", ngram_accuracy)


#Spacy tagger
# Tag the test set with SpaCy and convert to NLTK format
spacy_test_sents = []
for sent in usents:
    #print(sent)
    spacy_sent = []
    for word in nlp(' '.join(sent)):
        spacy_tag = mapping_spacy_to_NLTK.get(word.pos_, 'XX')
        spacy_sent.append((word.text, spacy_tag))
    spacy_test_sents.append(spacy_sent)
    
spacy_tagger = nltk.UnigramTagger(train=spacy_test_sents, cutoff=0)
spacy_tsents = spacy_tagger.tag_sents(usents)
# Evaluate the SpaCy tagger
print("SPACY not flattened:", accuracy(spacy_test_sents,spacy_tsents))
print("SPACY:", accuracy(list(chain.from_iterable(spacy_test_sents)),
                         list(chain.from_iterable(spacy_tsents))))



