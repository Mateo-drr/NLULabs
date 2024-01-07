# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import fitsvm, testsvm, vectorizerGetData, printRes

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # -*- coding: utf-8 -*-
    """
    Created on Thu Mar  9 11:22:27 2023
    
    @author: Mateo
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.datasets import fetch_20newsgroups
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    
    #loading data
    ng_train = fetch_20newsgroups(subset='train')
    ng_test = fetch_20newsgroups(subset='test')
    
    sample_data = ng_train.data
    labels = ng_train.target
    comp_res = [] #results list
    cutoff = [10, 0.8]
    variations=['Countvectorize','TF-IDF','TF-IDF cutoff','TF-IDF stopwords','TF-IDF stpw no lowercasing']
    
    #PART 1: Vectorize the data using count vectorize
    print(f'\n..........................................\n{variations[0]} results: ')
    
    #count vectorize
    vectorizer = CountVectorizer(lowercase=True, dtype=np.int16)#, binary=True)
    #get the vectorized data
    vectors_cv,test = vectorizerGetData(vectorizer,sample_data,ng_test)
    #fit the svm
    model = fitsvm(2000, vectors_cv, labels)
    #test and store the results
    comp_res.append(testsvm(model, test, ng_test))
    del vectors_cv
    
    print('\n..........................................\n',f'\n{variations[1]} results: \n')
    
    #TF-IDF
    vectorizer = TfidfVectorizer(lowercase=True, dtype=np.float32)
    #get the vectorized data
    tf_idf,test = vectorizerGetData(vectorizer,sample_data,ng_test)
    #fit the svm
    model = fitsvm(1000, tf_idf, labels)
    #test and store the results
    comp_res.append(testsvm(model, test, ng_test))
    
    print('\n..........................................\n',f'\n{variations[2]} results: \n')
    #TF-IDF cutoff
    print('Cutoff used: tokens that appear in less than', cutoff[0], 'documents, and tokens that appear in more than', cutoff[1]*100, '% of documents')
    vectorizer = TfidfVectorizer(lowercase=True, min_df=cutoff[0], max_df=cutoff[1], dtype=np.float32)
    #get the vectorized data
    tf_idf,test = vectorizerGetData(vectorizer,sample_data,ng_test)
    #fit the svm
    model = fitsvm(1000, tf_idf, labels)
    #test and store the results
    comp_res.append(testsvm(model, test, ng_test))
    
    print('\n..........................................\n',f'\n{variations[3]} results: \n')
    #TF-IDF stopwords
    NLTK_STOP_WORDS = stopwords.words('english')
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=NLTK_STOP_WORDS)
    #get the vectorized data
    tf_idf,test = vectorizerGetData(vectorizer,sample_data,ng_test)
    #fit the svm
    model = fitsvm(1000, tf_idf, labels)
    #test and store the results
    comp_res.append(testsvm(model, test, ng_test))
    
    print('\n..........................................\n',f'\n{variations[4]} results: \n')
    #TF-IDF stopwords no lowercase
    NLTK_STOP_WORDS = stopwords.words('english')
    vectorizer = TfidfVectorizer(lowercase=False,dtype=np.float32)
    #get the vectorized data
    tf_idf,test = vectorizerGetData(vectorizer,sample_data,ng_test)
    #fit the svm
    model = fitsvm(1000, tf_idf, labels)
    #test and store the results
    comp_res.append(testsvm(model, test, ng_test))
    
    #print results
    printRes(comp_res,variations)
