# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

#IT MAY BE NECESSARY TO RUN IT 3 TIMES IF STUCK!!

# Import everything from functions.py file
import nltk
import gc

from functions import *
import pandas as pd

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    nltk.download('conll2002')
    
    ###########################################################################
    
    # let's get only word and iob-tag
    trn_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
    tst_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testa')]
    
    #Start multiprocess for faster processing
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    #Get the base, suffix and all features data
    result1 = pool.apply_async(spacyfeats,(trn_sents, tst_sents))
    result2 = pool.apply_async(suffeats,(trn_sents, tst_sents))
    result3 = pool.apply_async(afeats,(trn_sents, tst_sents))
    
    trn_feats, trn_label, tst_feats, tst_label = result1.get()
    trn_feats_s, tst_feats_s = result2.get()
    trn_feats_a, tst_feats_a = result3.get()
    
    # Close the pool
    pool.close()
    pool.join()
    gc.collect()
    
    #Get the data with 1 and 2 of context
    trn_feats_1 = featsWcon1(trn_feats)
    tst_feats_1 = featsWcon1(tst_feats)
    trn_feats_2 = featsWcon2(trn_feats)
    tst_feats_2 = featsWcon2(tst_feats)

    
    ###########################################################################
    
    #Start multiprocessing for training
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    res=[]
    
    #Get the results of each
    res1 = pool.apply_async(runcrf,(trn_feats, trn_label, tst_feats, tst_sents))
    res2 = pool.apply_async(runcrf,(trn_feats_s, trn_label, tst_feats_s, tst_sents))
    res3 = pool.apply_async(runcrf,(trn_feats_a, trn_label, tst_feats_a, tst_sents))
    res4 = pool.apply_async(runcrf,(trn_feats_1, trn_label, tst_feats_1, tst_sents))
    res5 = pool.apply_async(runcrf,(trn_feats_2, trn_label, tst_feats_2, tst_sents))
    
    res.append(res1.get())
    res.append(res2.get())
    res.append(res3.get())
    res.append(res4.get())
    res.append(res5.get())
    
    # Close the pool            
    pool.close()
    pool.join()
    gc.collect()
    
    #Print the results in table format
    name = ['Baseline', 'Suffix', 'All Features', 'Context 1', 'Context 2']
    for i,results in enumerate(res):
        pd_tbl = pd.DataFrame().from_dict(results, orient='index')
        pd_tbl.round(decimals=3)
        print('---------------------------------------')
        print(name[i])
        print(pd_tbl)
        print('---------------------------------------')
    
    