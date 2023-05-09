# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:22:27 2023

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

def fitsvm(max_iter, data, labels,test, ng_test):
    model = svm.LinearSVC(max_iter=max_iter)
    model.fit(data,labels)

    #Predict labels from test samples
    hyps = model.predict(test)
    refs = ng_test.target
    cross_valid(model, test, refs)
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

#loading data
ng_train = fetch_20newsgroups(subset='train')
ng_test = fetch_20newsgroups(subset='test')
print(len(ng_train.data))

# randomly sample a portion of the data (unused since i fixed the memory issues by using int8)
sample_size = 11314
sample_indices = np.random.choice(len(ng_train.data), sample_size, replace=False)
sample_data = [ng_train.data[i] for i in sample_indices]
sample_labels = [ng_train.target[i] for i in sample_indices]

comp_res = []

#'''
print('\nCountvectorize results: \n')

#count vectorize
vectorizer = CountVectorizer(lowercase=True, binary=True)
vectors_cv = vectorizer.fit_transform(sample_data)
#print(len(vectors_cv.get_feature_names_out()))
vectors_cv.toarray()
print(vectors_cv.max(), vectors_cv.min())
vectors_cv = vectors_cv.astype(np.int8)
print(vectors_cv.max(), vectors_cv.min())
test = vectorizer.transform(ng_test.data).toarray().astype(np.int8)
labels = sample_labels

#print(vectors_cv[10], labels[:10])

comp_res.append(fitsvm(2000, vectors_cv, labels, test, ng_test))

'''
model = svm.LinearSVC(max_iter=2000)
model.fit(vectors_cv,labels)

#Predict labels from test samples
hyps = model.predict(X_test)
refs = ng_test.target
    
#Evaluate the model
report = classification_report(refs, hyps, target_names=ng_test.target_names)

print(report)
#'''
vectors_cv = 0

print('\nTF-IDF results: \n')

#TF-IDF
vectorizer = TfidfVectorizer(lowercase=True)
tf_idf = vectorizer.fit_transform(sample_data).toarray()
print(tf_idf.max(), tf_idf.min())
tf_idf = tf_idf.astype(np.float32)
print(tf_idf.max(), tf_idf.min())

test = vectorizer.transform(ng_test.data).toarray()

comp_res.append(fitsvm(1000, tf_idf, labels, test, ng_test))

print('\nTF-IDF cutoff results: \n')
#TF-IDF cutoff
cutoff = [10, 0.8]
print('Cutoff used: tokens that appear in less than', cutoff[0], 'documents, and tokens that appear in more than', cutoff[1]*100, '% of documents')
vectorizer = TfidfVectorizer(lowercase=True, min_df=cutoff[0], max_df=cutoff[1])
tf_idf = vectorizer.fit_transform(sample_data).toarray()
print(tf_idf.max(), tf_idf.min())
tf_idf = tf_idf.astype(np.float32)
print(tf_idf.max(), tf_idf.min())

test = vectorizer.transform(ng_test.data).toarray()

comp_res.append(fitsvm(1000, tf_idf, labels, test, ng_test))


print('\nTF-IDF stopwords results: \n')
#TF-IDF stopwords
NLTK_STOP_WORDS = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(lowercase=True, stop_words=NLTK_STOP_WORDS)
tf_idf = vectorizer.fit_transform(sample_data).toarray()
tf_idf = tf_idf.astype(np.float32)
print(tf_idf.max(), tf_idf.min())
test = vectorizer.transform(ng_test.data).toarray()

comp_res.append(fitsvm(1000, tf_idf, labels, test, ng_test))

print('\nTF-IDF no lowercasing results: \n')
#TF-IDF stopwords
NLTK_STOP_WORDS = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(lowercase=False)
tf_idf = vectorizer.fit_transform(sample_data).toarray()
tf_idf = tf_idf.astype(np.float32)
print(tf_idf.max(), tf_idf.min())
test = vectorizer.transform(ng_test.data).toarray()

comp_res.append(fitsvm(1000, tf_idf, labels, test, ng_test))


print('\nResults comparison:\n')
for result in comp_res:
    print(result)
    
    