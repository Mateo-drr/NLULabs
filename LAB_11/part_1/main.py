# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    from sklearn.model_selection import StratifiedKFold
    import copy
    from transformers import BertTokenizer, BertModel
    
    #CONFIG VARIABLES
    numsp = 10
    device='cuda'
    res = []
    names = ['BERT', 'SVM', 'SVM+BERT']
    batch_size=90
    lr = 0.00005
    num_epochs=4
    criterion = nn.BCELoss()
    #subjective = 0, objective =1
    window = 0.6 #remove sentences that get a prediction lower than
    torch.backends.cudnn.benchmark = True 
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    
    #DEFINE KFOLD
    kfold = StratifiedKFold(n_splits=numsp, shuffle=True, random_state=1)
    kfmetrics = []
    
    #PREP SUBJECTIVITY DATA
    sents = prepDSSubj()
    
    #load bert
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    
    #RUN KFOLD ON BERT
    bestLoss = 1e5
    for fold, (train_indices, val_indices) in enumerate(kfold.split(sents,[s['lbl'] for s in sents])):
        
        #RESET MODEL
        model = SubjModel(768, 128, 1,bert).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        print(f'Fold {fold + 1}')
        for epoch in range(num_epochs):
            
            train_ds,train_dl,valid_ds,valid_dl = prepDL(sents, train_indices, val_indices, tokenizer, batch_size)
            
            trainLoop(train_dl, train_ds, model, optimizer, criterion, epoch, device)
        
            kfmetrics = validLoop(valid_dl, valid_ds, model, criterion, epoch, device, kfmetrics)

            #Store the best model
            if kfmetrics[-1]['loss'] < bestLoss:
                bestLoss = kfmetrics[-1]['loss']
                bestModel = copy.deepcopy(model)
                
    #GET AVERAGE RESULTS
    res.append(printMetrics(kfmetrics, names[0]))
    
    ###########################################################################
    
    #PREP MOVIES DATASET
    words, txtlbls, fwords, docsents = prepDSMov()

    #RUN SVM CLASSIFICATION
    kfmetrics = runSvm(fwords,txtlbls,numsp)
    
    #GET AVERAGE RESULTS
    res.append(printMetrics(kfmetrics, names[1]))
    
    ###########################################################################
    
    #Remove objective sentences
    fsents=[]
    for i in tqdm(range(len(docsents))): #loop each document by creating a ds for each doc
        dl = prepDL2(docsents, i, tokenizer, batch_size)
        fsents = removeObjSents(dl, bestModel, docsents, device, i, fsents, window)
    
    #Join the sentences in the documents to a single string
    fdocs = [' '.join(doc) for doc in fsents]
    
    #Train svm with filtered sentences
    kfmetrics = runSvm(fdocs, txtlbls,numsp)
    #GET AVERAGE RESULTS
    res.append(printMetrics(kfmetrics, names[2]))
    
    ###########################################################################
    
    #FINAL RESULTS
    for i in range(len(res)):
        print(names[i])
        rounded_values = {key: round(value, 5) for key, value in res[i].items()}
        print(rounded_values)

    
    
    
    
    
    
    
    
    
    
    
    
    
    