# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *

from torch.utils.data import DataLoader
import torch.optim as optim

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    
    path='D:/Universidades/Trento/2S/NLU/dataset/ATIS/'
    device='cuda'
    batch_size=64
    
    train_raw, dev_raw, test_raw = preproc(path)
    
    words, intents, slots = getLang(train_raw, dev_raw, test_raw)
    lang = Lang(words, intents, slots, cutoff=0)
    
    train_ds = CustomDataset(train_raw, lang, tokenizer, 128)
    test_ds = CustomDataset(test_raw, lang, tokenizer, 128)
    dev_ds = CustomDataset(dev_raw, lang, tokenizer, 128)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn,shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, collate_fn=collate_fn)
    
    lr = 0.0002 # learning rate
    clip = 5 # Clip the gradient    
    n_epochs=20 
    patience=3
    hid_size=768
    out_int=len(intents)
    out_slot=len(lang.slot2id)

    model = BertCstm(bert, hid_size, out_slot, out_int)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    
    slot_f1s, intent_acc = [], []
    best_f1=0
    for epoch in range(1,n_epochs+1):
        model.train()
        patience, best_f1 = train_loop(train_dl, train_ds, device, optimizer,
                                       model,criterion_intents, criterion_slots,
                                       epoch, n_epochs,lang,tokenizer,dev_dl,
                                       patience,best_f1)
        
        if patience <=0:
            print('Patience limit reached!')
            break

    model.eval()
    slot_f1s, intent_acc = evalModel(model, test_dl, criterion_slots, criterion_intents, lang, slot_f1s, intent_acc, tokenizer)
    
    printRes(slot_f1s, intent_acc, 'Bert')