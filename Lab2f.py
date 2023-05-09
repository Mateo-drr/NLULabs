# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:48:00 2023

@author: Mateo
"""

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.model_selection import StratifiedKFold
stratified_split = StratifiedKFold(n_splits=5, shuffle=True)

def fitsvm(max_iter, data, labels,v_test, ng_test, test,refs,C):
    model = svm.LinearSVC(max_iter=max_iter, C=C)
    model.fit(data,labels)

    #Predict labels from test samples
    hyps = model.predict(v_test)
    
    
    #cross_valid(model, v_test, refs)
    #Evaluate the model
    report = classification_report(refs, hyps, target_names=ng_test.target_names)
    print(report)
    return report[-200:-1]

def cross_valid(clf, data, target):
    import math
    from sklearn.model_selection import cross_validate
    import warnings
    warnings.filterwarnings("ignore")

    #some are not for this 
    scores = ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'neg_log_loss', 'precision', 'recall', 'jaccard', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']

    for scr in scores:
        #clf = GaussianNB()
        scores = cross_validate(clf, test, target, cv=5, scoring=[scr])
        if not math.isnan(sum(scores['test_'+scr])/len(scores['test_'+scr])):
            print(scr , sum(scores['test_'+scr])/len(scores['test_'+scr]))

def formating(train,test,ng_train):
    sample_data = [ng_train.data[i] for i in train]
    sample_labels = [ng_train.target[i] for i in train]
    test_data = [ng_train.data[i] for i in test]
    refs = [ng_train.target[i] for i in test]
    return sample_data,sample_labels,test_data,refs


#loading data
ng_train = fetch_20newsgroups(subset='all')
print(len(ng_train.data))

comp_res = []

#'''
print('\nCountvectorize results: \n')

for train, test in stratified_split.split(ng_train.data, ng_train.target):
    sample_data, sample_labels, test_data, refs = formating(train, test, ng_train)
    
    #count vectorize
    vectorizer = CountVectorizer(lowercase=True, binary=True, dtype=np.int8)
    vectors_cv = vectorizer.fit_transform(sample_data)
    v_test = vectorizer.transform(test_data)
    
    comp_res.append(fitsvm(1000, vectors_cv, sample_labels, v_test, ng_train, test, refs,0.01))
    
print('\nTF-IDF results: \n')
    
for train, test in stratified_split.split(ng_train.data, ng_train.target):
    sample_data, sample_labels, test_data, refs = formating(train, test, ng_train)
    
    #count vectorize
    vectorizer = TfidfVectorizer(lowercase=True)
    vectors_cv = vectorizer.fit_transform(sample_data)
    v_test = vectorizer.transform(test_data)
    
    comp_res.append(fitsvm(1000, vectors_cv, sample_labels, v_test, ng_train, test, refs, 0.5))
    

    
print('\nResults comparison:\n')
for result in comp_res:
    print(result)