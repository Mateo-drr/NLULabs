# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    import torch.nn as nn
    import torch.optim as optim
    from functools import partial
    from torch.utils.data import DataLoader
    import gc
    
    
    #spacy.cli.download("en_core_web_lg")
    
    #PARAMS
    device="cuda" if torch.cuda.is_available() else "cpu"
    path='D:/Universidades/Trento/2S/NLU/dataset/'
    bpath='D:/Universidades/Trento/2S/NLU/LAB_09/part_1/bin/' 
    hid_size = 128+256
    emb_size = 256
    clip = 5 # Clip the gradient
    n_epochs = 50
    patience = 7
    runs=5
    save=True
    torch.backends.cudnn.benchmark = True 
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    #LOAD THE DATA
    train_raw = read_file(path+"ptb.train.txt")
    dev_raw = read_file(path+"ptb.valid.txt")
    test_raw = read_file(path+"ptb.test.txt")
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    #MAKE THE DATASET
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    #COMPUTE VOCAB
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    
    #DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    #LOSSES
    vocab_len = len(lang.word2id)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    #store results here
    pptot=[]
    
    ###########################################################################
    #RNN w SGD
    lr = 0.3
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model.apply(init_weights)
        
        #OPTIMIZER
        optimizer = optim.SGD(model.parameters(), lr=lr)
            
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device,
                 patience)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
        
        restemp.append(res)
        if save and res <= max(restemp):
            topModel = copy.deepcopy(best_model)
            
        torch.cuda.empty_cache()
        
    #save model
    if save:
        torch.save(topModel.state_dict(), bpath+'RNN.pth')
        
    pptot.append(restemp)
    gc.collect()
    
    ###########################################################################
    #LSTM w SGD
    lr = 0.8
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"] ,
                        emb_dropout=0, out_dropout=0).to(device)    
        model.apply(init_weights)
        
        #OPTIMIZER
        optimizer = optim.SGD(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
            
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device,
                 patience)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
        
        restemp.append(res)
        if save and res <= max(restemp):
            topModel = copy.deepcopy(best_model)
            
        torch.cuda.empty_cache()
        
    #save model
    if save:
        torch.save(topModel.state_dict(), bpath+'LSTM.pth')
        
    pptot.append(restemp)
    gc.collect()
    
    ###########################################################################
    #LSTM w SGD + DROPOUT
    lr = 0.85
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                        emb_dropout=0.2, out_dropout=0.2).to(device)    
        model.apply(init_weights)
        
        #OPTIMIZER
        optimizer = optim.SGD(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
            
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device,
                 patience)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
        
        restemp.append(res)
        if save and res <= max(restemp):
            topModel = copy.deepcopy(best_model)
        
        torch.cuda.empty_cache()
        
    #save model
    if save:
        torch.save(topModel.state_dict(), bpath+'LSTMd.pth')
        
    pptot.append(restemp)
    gc.collect()
    
    ###########################################################################
    #LSTM w AdamW + DROPOUT
    lr = 0.001
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                        emb_dropout=0.2, out_dropout=0.2).to(device)    
        model.apply(init_weights)
        
        #OPTIMIZER
        optimizer = optim.AdamW(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
            
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device,
                 patience)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
        
        restemp.append(res)
        if save and res <= max(restemp):
            topModel = copy.deepcopy(best_model)
            
        torch.cuda.empty_cache()
            
    #save model
    if save:
        torch.save(topModel.state_dict(), bpath+'LSTMdAW.pth')
        
    pptot.append(restemp)
    gc.collect()
    
    ###########################################################################

    printRes(pptot)
    
    