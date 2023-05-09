# -*- coding: utf-8 -*-
"""
Created on Tue May  9 20:34:38 2023

@author: Mateo-drr
"""
from sklearn_crfsuite import CRF
from nltk.corpus import conll2002
import spacy
#import es_core_news_sm

spacy.cli.download("es_core_news_sm")
nlp = spacy.load("es_core_news_sm")
#nlp = es_core_news_sm.load()

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def word2features(sent, i):
    word = sent[i][0]
    return {'bias': 1.0, 'word.lower()': word.lower()}

def sent2spacy_features(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)
    
    return feats

trn_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]

tst_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testa')]

trn_feats = [sent2features(s) for s in trn_sents]
trn_label = [sent2labels(s) for s in trn_sents]

crf = CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)


try:
    crf.fit(trn_feats, trn_label)
except AttributeError:
    pass

tst_feats = [sent2features(s) for s in tst_sents]
pred = crf.predict(tst_feats)



'''
results = evaluate(tst_sents, hyp)
pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)
'''