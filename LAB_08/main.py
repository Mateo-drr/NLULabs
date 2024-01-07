# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import nltk
nltk.download('senseval')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet_ic')
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    from nltk.corpus import senseval
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import cross_validate
    import numpy as np
    
    #Load and process data
    instances = senseval.instances('interest.pos')
    data,lbls,vectors,labels,stratified_split = dataproc()
    #classifier
    classifier = MultinomialNB()
    
    sc=[]
    ###########################################################################
    #BOW
    scores = cross_validate(classifier, vectors, labels, cv=stratified_split, scoring=['f1_micro'])

    sc.append(scores)
    
    ###########################################################################
    #COLLOCATIONAL FEATURES N window + POS   

    #process data
    dvectors = getJointData(instances, 3) #n window
    
    scores = cross_validate(classifier, dvectors, labels, cv=stratified_split, scoring=['f1_micro'])

    sc.append(scores)
    
    ###########################################################################
    #BOW + features
    
    dvectors = getJointData(instances, 2)
    #Join the vectors and the dvectors
    uvectors = np.concatenate((vectors.toarray(), dvectors), axis=1)
    
    scores = cross_validate(classifier, uvectors, labels, cv=stratified_split, scoring=['f1_micro'])

    sc.append(scores)
    
    ###########################################################################
    #LESK

    ac,ac2=[],[]
    for train_index, test_index in stratified_split.split(instances, lbls):
        train_data = [instances[i] for i in train_index]
        test_data = [instances[i] for i in test_index]
        
        ac.append(leskEval(test_data, option=1)) #1 for og lesk, otherwise simlesk
        ac2.append(leskEval(test_data, option=0)) #1 for og lesk, otherwise simlesk

    print("\nAverage Accuracy OG LESK:", np.mean(np.array(ac)))
    print("Average Accuracy Sim LESK:", np.mean(np.array(ac2)))
    n = ['BOW','COL Feats N windw + POS','BOW + Feats']
    for i,scores in enumerate(sc):
        print(n[i]+':',sum(scores['test_f1_micro'])/len(scores['test_f1_micro']))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    