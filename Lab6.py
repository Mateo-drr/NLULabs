# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:54:58 2023

@author: Mateo
"""

# Spacy version 
from nltk.parse.dependencygraph import DependencyGraph
from spacy.tokenizer import Tokenizer
import spacy 
import spacy_conll
from nltk.corpus import dependency_treebank
import spacy_stanza
from nltk.parse import DependencyEvaluator
import nltk
nltk.download('dependency_treebank')

stanza = spacy_stanza.load_pipeline("en", verbose=False, tokenize_pretokenized=True)
config = {"ext_names": {"conll_pd": "pandas"},
          "conversion_maps": {"deprel": {"nsubj": "subj", "root":"ROOT"}}}
stanza.add_pipe("conll_formatter", config=config, last=True)

spcy = spacy.load("en_core_web_sm")
config = {"ext_names": {"conll_pd": "pandas"},
          "conversion_maps": {"deprel": {"nsubj": "subj"}}}
spcy.add_pipe("conll_formatter", config=config, last=True)
spcy.tokenizer = Tokenizer(spcy.vocab) 

sents = dependency_treebank.sents()[-100:]
psents = dependency_treebank.parsed_sents()[-100:]
jsents = []
stz = []
spy = []

tmpz = []
tmpy = []

dpz = []
dpy = []

for i,item in enumerate(sents):
    jsents.append(" ".join(item))
    stz.append(stanza(jsents[i]))
    spy.append(spcy(jsents[i]))
    
    stz[i] = stz[i]._.pandas#["DEPREL"].replace({"root": "ROOT"})
    spy[i] = spy[i]._.pandas
    
    stz[i]['DEPREL'] = stz[i]['DEPREL'].replace({"root": "ROOT"})
    
    tmpz.append( stz[i][["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False))
    tmpy.append(spy[i][["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False))
    
    dpz.append(DependencyGraph(tmpz[i]))
    dpy.append(DependencyGraph(tmpy[i]))
    
    if i== 99:
        
        break
    
de = DependencyEvaluator(dpy, psents)
las, uas = de.eval()
print('Spacy: ')
print("LAS:", las)
print("UAS:", uas)
de = DependencyEvaluator(dpz, psents)
las, uas = de.eval()
print("\nStanza: ")
print("LAS:", las)
print("UAS:", uas)






