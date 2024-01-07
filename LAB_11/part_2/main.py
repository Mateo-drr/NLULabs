# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
#from functions import *
from torch.utils.data import DataLoader
import torch
#from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from torch import nn
import torch.optim as optim
from utils import loadData, CustomDataset, idx2pol, preproc
from model import Bert2d
from functions import *
import gc

if __name__ == "__main__":
    
    #PARAMS
    runs = 3
    device='cuda'
    res = []
    names = ['BERT', 'SVM', 'SVM+BERT']
    batch_size=64
    lr = 1e-4
    num_epochs=75
    hsize=128
    outPol=10
    outAsp=10
    dout=0.4
    clip=1
    
    # critAsp = nn.CrossEntropyLoss(ignore_index=0)
    # critPol = nn.CrossEntropyLoss(ignore_index=0, weight=torch.tensor([1,0.1,1,1,1]).to(device))
    critAsp = nn.MSELoss()
    critPol = nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True 
    #torch.set_num_threads(8)
    #torch.set_num_interop_threads(8)
    
    #DATA LOADING AND PRE-PROCESSING
    classes = []
    path = 'D:/Universidades/Trento/2S/NLU/dataset/laptop14_train.txt' #766 asp #max num jasp 9
    temp,classes = loadData(path,classes)
    path = 'D:/Universidades/Trento/2S/NLU/dataset/laptop14_dev.txt' #max num jasp 5
    vsents,classes = loadData(path,classes)
    path = 'D:/Universidades/Trento/2S/NLU/dataset/laptop14_test.txt' #max num jasp 5
    test_sents,classes = loadData(path,classes)
    #get all the aspects and clean training data
    tsents,aspects = preproc(temp,classes)

    #LOAD ROBERTA
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#("bert-base-uncased")
    # bert = BertModel.from_pretrained('bert-base-uncased').to(device)    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')#("bert-base-uncased")
    bert = RobertaModel.from_pretrained('roberta-base').to(device)    
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-large')#("bert-base-uncased")
    #bert = RobertaModel.from_pretrained('roberta-large').to(device)   
    
    #DS and DL
    # using 128 size since biggest tokenized sentence in training data is 85 tokens
    train_ds = CustomDataset(tsents, tokenizer, 128, aspects)
    valid_ds = CustomDataset(vsents, tokenizer, 128, aspects)
    test_ds = CustomDataset(test_sents, tokenizer, 128, aspects)
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    valid_dl = DataLoader(valid_ds,batch_size=512,shuffle=False)
    test_dl = DataLoader(test_ds,batch_size=512,shuffle=False)
    
    #Store results
    finalModel=None
    met = []
    amet = []
    pmet = []
    bestTL=1e6
    
    #RUN MODEL
    for r in range(0,runs):
        #BEST RESULTS IN EACH RUN
        bestVL=1e6
        bestModel = None
        
        print('\nRun', r + 1)
        
        #MODEL
        #model = aspExt(bert,hsize, outsize,dout=dout)
        #model = LstmMod(hsize, outsize, outsize, 128, n_layer=layers)
        #model = modCSTM(hsize, outsize, bert)
        #model = BertCstm(bert, hsize, len(aspects)+1, 5,dout=dout)
        #model = Bert10(bert, hsize, 10, 10,dout=dout)
        model = Bert2d(bert, hsize, outPol, outAsp,dout=dout)
        model.to(device)
        
        #OPTIM
        optimizer = optim.AdamW(model.parameters(),lr=lr)
        
        #TRAIN AND EVAL
        for epoch in range(0,num_epochs):
            print('E:',epoch+1)
            
            trainLoop(model,train_ds,train_dl,device,optimizer,critAsp,critPol,clip)

            bestVL,bestModel = evalLoop(model,valid_ds,valid_dl,device,critAsp,critPol,bestModel,bestVL)
            
            gc.collect()
            torch.cuda.empty_cache()
            
        #TESTING
        predictions,aspm,polm,finalModel = testLoop(bestModel,test_ds,test_dl,device,critAsp,critPol,tokenizer,idx2pol,bestTL,finalModel)
        
        allm = metricsAll(predictions['allP'],predictions['allT'])
        met.append(allm)
        amet.append(aspm)
        pmet.append(polm)
        
    #Metrics of all runs
    finalEval(met,amet,pmet)    
        
    #save best model of all runs
    torch.save(finalModel.state_dict(),'best.pth')