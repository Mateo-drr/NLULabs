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
    device = 'cuda'
    path='D:/Universidades/Trento/2S/NLU/dataset/'
    batch_size=32
    hid_size = 256
    emb_size = 256
    clip = 5 # Clip the gradient
    n_epochs = 500
    patience = 100
    dout=0.2
    runs=5
    torch.backends.cudnn.benchmark = True 
    
    #LOAD THE DATA
    train_raw = read_file(path+"ptb.train.txt")
    dev_raw = read_file(path+"ptb.valid.txt")
    test_raw = read_file(path+"ptb.test.txt")
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    #DATASET
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    #VOCAB
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    
    #DATALOADER
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    

    #LOSSES
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    vocab_len = len(lang.word2id)
    
    #Store results here
    pptot=[]
    
    ###########################################################################
    print('\nWeight Tying + normal dropout')
    lr = 0.001
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                        emb_dropout=dout, out_dropout=dout).to(device)    
        model.apply(init_weights)
        
        #OPTIMIZER
        optimizer = optim.AdamW(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
            
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device, patience=patience)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
        
    
        restemp.append(res)
    #save model
    torch.save(best_model, 'WTND.pth')
    pptot.append(restemp)
    gc.collect()
    torch.cuda.empty_cache()
 
    ###########################################################################
    
    ###########################################################################
    print('\nWeight Tying + var dropout')
    lr = 0.001
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                        emb_dropout=dout, out_dropout=dout, vardrop=True, dropprob=dout).to(device)    
        model.apply(init_weights)
        
        #OPTIMIZER 
        optimizer = optim.AdamW(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
            
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device,patience=patience)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
        
        restemp.append(res)
    #save model
    torch.save(best_model, 'WTVD.pth')
    pptot.append(restemp)
    gc.collect()
    torch.cuda.empty_cache()

    ###########################################################################
    print('\nWeight Tying + var dropout + sgd')
    lr = 0.9
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                        emb_dropout=dout, out_dropout=dout, vardrop=True, dropprob=dout).to(device)    
        model.apply(init_weights)
        
        #OPTIMIZER
        optimizer =  optim.SGD(model.parameters(), lr=lr)
            
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device,patience=patience)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
    
        restemp.append(res)
    #save model
    torch.save(best_model, 'WTVDsgd.pth')
    pptot.append(restemp)
    gc.collect()
    torch.cuda.empty_cache()


    ###########################################################################
    print('\nWeight Tying + var dropout + nt-asgd init asgd')
    lr = 0.9
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                        emb_dropout=dout, out_dropout=dout, vardrop=True, dropprob=dout).to(device)    
        model.apply(init_weights)
        
        #OPTIMIZER
        optimizer =  optim.ASGD(model.parameters(), lr=lr, t0=0, weight_decay=1.2e-6)
        config = {
                'n': 5,  # Non-monotone interval
                'L': len(train_loader),  # Logging interval
                't': 0,
                'lr':lr,
                'k':0,
                'T':0,
                'logs':[]
                }
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device,ntasgd=True,patience=patience,config=config)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
    
        restemp.append(res)
    #save model
    torch.save(best_model, 'WTVDnt1.pth')
    pptot.append(restemp)
    gc.collect()
    torch.cuda.empty_cache()

    ###########################################################################
    
    print('\nWeight Tying + var dropout + nt-asgd init sgd')
    lr = 0.9
    restemp=[]
    for r in range(0,runs):
        print('Run', r+1)
        #INIT MODEL
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                        emb_dropout=dout, out_dropout=dout, vardrop=True, dropprob=dout).to(device)    
        model.apply(init_weights)
        
        #OPTIMIZER
        optimizer =  optim.SGD(model.parameters(), lr=lr)#, t0=0, weight_decay=0)
        config = {
                'n': 5,  # Non-monotone interval
                'L': len(train_loader),  # Logging interval
                't': 0,
                'lr':lr,
                'k':0,
                'T':0,
                'logs':[]
                }
        #TRAIN AND TEST MODEL
        best_model = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
                 model, clip, dev_loader, train_loader, test_loader, lang, device,ntasgd=True,patience=patience,config=config)
        res = testModel(best_model,device, test_loader, criterion_eval, lang)
    
        restemp.append(res)
    #save model
    torch.save(best_model, 'WTVDnt1.pth')
    pptot.append(restemp)
    gc.collect()
    torch.cuda.empty_cache()
    
    ###########################################################################
    
    printRes(pptot)
