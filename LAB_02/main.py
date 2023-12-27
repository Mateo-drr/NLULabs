# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import fitsvm, cross_valid

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
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
    
    #PART 1: Vectorize the data using count vectorize
    print('\nCountvectorize results: ')
    
    #count vectorize
    vectorizer = CountVectorizer(lowercase=True, dtype=np.int16)#, binary=True)
    vectors_cv = vectorizer.fit_transform(sample_data)
    #print(len(vectors_cv.get_feature_names_out()))
    #vectors_cv.toarray()
    
    #print('Double checking if data range isnt affected: \n','\t',vectors_cv.max(), vectors_cv.min())
    #vectors_cv = vectors_cv.astype(np.int8)
    #print('\t',vectors_cv.max(), vectors_cv.min())
    test = vectorizer.transform(ng_test.data)#.toarray().astype(np.int8)
    labels = sample_labels
    
    #print(vectors_cv[10], labels[:10])
    
    comp_res.append(fitsvm(2000, vectors_cv, labels, test, ng_test))
    vectors_cv = 0
    
    print('\n..........................................\n','\nTF-IDF results: \n')
    
    #TF-IDF
    vectorizer = TfidfVectorizer(lowercase=True)
    tf_idf = vectorizer.fit_transform(sample_data).toarray()
    print('Double checking if data range isnt affected: \n','\t',tf_idf.max(), tf_idf.min())
    tf_idf = tf_idf.astype(np.float32)
    print('\t',tf_idf.max(), tf_idf.min())
    
    test = vectorizer.transform(ng_test.data).toarray().astype(np.float32)
    
    comp_res.append(fitsvm(1000, tf_idf, labels, test, ng_test))
    
    print('\n..........................................\n','\nTF-IDF cutoff results: \n')
    #TF-IDF cutoff
    cutoff = [10, 0.8]
    print('Cutoff used: tokens that appear in less than', cutoff[0], 'documents, and tokens that appear in more than', cutoff[1]*100, '% of documents')
    vectorizer = TfidfVectorizer(lowercase=True, min_df=cutoff[0], max_df=cutoff[1])
    tf_idf = vectorizer.fit_transform(sample_data).toarray()
    print('Double checking if data range isnt affected: \n','\t',tf_idf.max(), tf_idf.min())
    tf_idf = tf_idf.astype(np.float32)
    print('\t',tf_idf.max(), tf_idf.min())
    
    test = vectorizer.transform(ng_test.data).toarray()
    
    comp_res.append(fitsvm(1000, tf_idf, labels, test, ng_test))
    
    
    print('\n..........................................\n','\nTF-IDF stopwords results: \n')
    #TF-IDF stopwords
    NLTK_STOP_WORDS = stopwords.words('english')
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=NLTK_STOP_WORDS)
    tf_idf = vectorizer.fit_transform(sample_data).toarray()
    print('Double checking if data range isnt affected: \n','\t',tf_idf.max(), tf_idf.min())
    tf_idf = tf_idf.astype(np.float32)
    print('\t',tf_idf.max(), tf_idf.min())
    test = vectorizer.transform(ng_test.data).toarray()
    
    comp_res.append(fitsvm(1000, tf_idf, labels, test, ng_test))
    
    print('\n..........................................\n','\nTF-IDF no lowercasing results: \n')
    #TF-IDF stopwords
    NLTK_STOP_WORDS = stopwords.words('english')
    vectorizer = TfidfVectorizer(lowercase=False)
    tf_idf = vectorizer.fit_transform(sample_data).toarray()
    print('Double checking if data range isnt affected: \n','\t',tf_idf.max(), tf_idf.min())
    tf_idf = tf_idf.astype(np.float32)
    print('\t',tf_idf.max(), tf_idf.min())
    test = vectorizer.transform(ng_test.data).toarray()
    
    comp_res.append(fitsvm(1000, tf_idf, labels, test, ng_test))
    
    
    print('\nResults comparison:\n')
    for result in comp_res:
        print(result)
    
    
