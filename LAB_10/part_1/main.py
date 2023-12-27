# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    path='D:/Universidades/Trento/2S/NLU/dataset/ATIS/'
    device='cuda'
    
    train_raw, dev_raw, test_raw = preproc(path)
    words, intents, slots = getLang(train_raw, dev_raw, test_raw)
    lang = Lang(words, intents, slots, cutoff=0)
    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
                        
    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    hid_size = 200
    emb_size = 300
    
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient
    
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    runs = 5
    n_epochs = 200
    patience = 3    
    
    ###########################################################################
    #Original Model
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                         vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)
    
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        
        model = trainModel(model, n_epochs, train_loader, optimizer, criterion_slots,
                           criterion_intents, sampled_epochs, losses_train,
                           dev_loader, lang, losses_dev, best_f1, patience)
    
        slot_f1s, intent_acc = evalModel(model, test_loader, criterion_slots, 
                                         criterion_intents, lang, slot_f1s,
                                         intent_acc)
        
    printRes(slot_f1s, intent_acc, 'Original')
    ###########################################################################
    #Bidirectional Model
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = ModelIASbd(hid_size, out_slot, out_int, emb_size, 
                         vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)
    
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        
        model = trainModel(model, n_epochs, train_loader, optimizer, criterion_slots,
                           criterion_intents, sampled_epochs, losses_train,
                           dev_loader, lang, losses_dev, best_f1, patience)
    
        slot_f1s, intent_acc = evalModel(model, test_loader, criterion_slots, 
                                         criterion_intents, lang, slot_f1s,
                                         intent_acc)
        
    printRes(slot_f1s, intent_acc, 'Bidirectional')
    ###########################################################################
    #Bidirectional Model + Dout
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = ModelIASbd(hid_size, out_slot, out_int, emb_size, 
                         vocab_len, pad_index=PAD_TOKEN, dout=0.1).to(device)
        model.apply(init_weights)
    
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        
        model = trainModel(model, n_epochs, train_loader, optimizer, criterion_slots,
                           criterion_intents, sampled_epochs, losses_train,
                           dev_loader, lang, losses_dev, best_f1, patience)
    
        slot_f1s, intent_acc = evalModel(model, test_loader, criterion_slots, 
                                         criterion_intents, lang, slot_f1s,
                                         intent_acc)
        
    printRes(slot_f1s, intent_acc, 'Bidirectional & Dropout')
    
    