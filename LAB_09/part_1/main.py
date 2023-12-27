# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from functools import partial
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import gc
    
    
    #spacy.cli.download("en_core_web_lg")
    
    device = 'cuda'
    path='D:/Universidades/Trento/2S/NLU/dataset/'
    
    train_raw = read_file(path+"ptb.train.txt")
    dev_raw = read_file(path+"ptb.valid.txt")
    test_raw = read_file(path+"ptb.test.txt")
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # Vocab is computed only on training set 
    # However you can compute it for dev and test just for statistics about OOV 
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    
    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    # Experiment also with a smaller or bigger model by changing hid and emb sizes 
    # A large model tends to overfit
    hid_size = 256
    emb_size = 256
     
    clip = 5 # Clip the gradient
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    n_epochs = 100
    patience = 3
    vocab_len = len(lang.word2id)
    
    pptot=[]
    ###########################################################################
    #RNN w SGD
    lr = 0.05
    
    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
        
    res = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
             model, clip, dev_loader, train_loader, test_loader, lang, device)
    
    pptot.append(res)
    gc.collect()
    torch.cuda.empty_cache()
    ###########################################################################
    #LSTM w SGD
    lr = 0.05
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"] ,
                    emb_dropout=0, out_dropout=0).to(device)    
    model.apply(init_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
        
    res = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
             model, clip, dev_loader, train_loader, test_loader, lang, device)
    
    pptot.append(res)
    gc.collect()
    torch.cuda.empty_cache()
    ###########################################################################
    #LSTM w SGD + DROPOUT
    lr = 0.05
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                    emb_dropout=0.1, out_dropout=0.1).to(device)    
    model.apply(init_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
        
    res = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
             model, clip, dev_loader, train_loader, test_loader, lang, device)
    
    pptot.append(res)
    gc.collect()
    torch.cuda.empty_cache()
    ###########################################################################
    #LSTM w SGD + DROPOUT
    lr = 0.0001
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                    emb_dropout=0.1, out_dropout=0.1).to(device)    
    model.apply(init_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
        
    res = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
             model, clip, dev_loader, train_loader, test_loader, lang, device)
    
    pptot.append(res)
    gc.collect()
    torch.cuda.empty_cache()
    ###########################################################################
    lbl=['RNN', 'LSTM', 'LSTM+DO', 'LSTM+DO+AW']
    print('\n')
    for i,name in enumerate(lbl):
        print(name, pptot[i])
    
    
    