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
    
    #PARAMS
    path='D:/Universidades/Trento/2S/NLU/dataset/ATIS/'
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True  
    runs=5
    batch_size=128
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient    
    n_epochs=50
    hid_size=768
    pat = 7
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    #get the data
    train_raw, dev_raw, test_raw = preproc(path) #longest utter in tokens is 52, in words its 46. 
    
    #Process it
    words, intents, slots = getLang(train_raw, dev_raw, test_raw)
    lang = Lang(words, intents, slots, cutoff=0)
    
    #create the dataset
    train_ds = CustomDataset(train_raw, lang, tokenizer)
    test_ds = CustomDataset(test_raw, lang, tokenizer)
    dev_ds = CustomDataset(dev_raw, lang, tokenizer)
    
    #dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn,shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=64, collate_fn=collate_fn)
    
    #output sizes
    out_int=len(intents)
    out_slot=len(lang.slot2id)
    
    #store the results in these
    slot_met, intent_met = [[],[],[]], [[],[],[]]
    
    #Train and test the model n times
    for x in range(0, runs):
        print('Run',x+1)
        
        #model and train params
        model = BertCstm(bert, hid_size, out_slot, out_int)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        best_f1=0
        patience=pat

        for epoch in range(1,n_epochs+1):
            #TRAINING
            model.train()
            patience, best_f1 = train_loop(train_dl, train_ds, device, optimizer,
                                           model,criterion_intents, criterion_slots,
                                           epoch, n_epochs,lang,tokenizer,dev_dl,
                                           patience,best_f1,pat) #also includes validation
            
            if patience <=0:
                print('Patience limit reached!')
                break
    
        #TESTING
        #load best model for testing
        model = BertCstm(bert, hid_size, out_slot, out_int)
        model.to(device)
        model.load_state_dict(torch.load('best.pth', map_location=device))
        model.eval()
        slot_met, intent_met = evalModel(model, test_dl, criterion_slots, criterion_intents, lang, slot_met, intent_met, tokenizer) 
    
    #RESULTS
    printRes(slot_met, intent_met, 'Bert')