# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
#from functions import *
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer, BertModel
#from transformers import RobertaTokenizer, RobertaModel
from torch import nn
import torch.optim as optim
from utils import *
from model import BertCstm
from functions import *
import gc
#from gensim.models import KeyedVectors

if __name__ == "__main__":
    
    #PARAMS
    runs = 5
    device="cuda" if torch.cuda.is_available() else "cpu"
    res = []
    batch_size=64
    lr = 1e-4
    num_epochs=100
    hsize=256
    outPol=10
    outAsp=10
    dout=0.5
    clip=5
    patience=10
    pOG=copy.deepcopy(patience)
    
    critAsp = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
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

    #LOAD bert
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#("bert-base-uncased")
    bert = BertModel.from_pretrained('bert-base-uncased').to(device)       
    
    #W2V for augmentation
    #model_path = "D:/Universidades/Trento/2S/NLU/LAB_11/part_2/GoogleNews-vectors-negative300.bin.gz"
    #w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)
    
    #DS and DL
    train_ds = CustomDataset(tsents, tokenizer,train=True)#,w2v=w2v,train=True)
    
    valid_ds = CustomDataset(vsents, tokenizer)
    test_ds = CustomDataset(test_sents, tokenizer)
    train_dl = DataLoader(train_ds,batch_size=batch_size,collate_fn=collate_fn,shuffle=True)
    valid_dl = DataLoader(valid_ds,batch_size=512,collate_fn=collate_fn,shuffle=False)
    test_dl = DataLoader(test_ds,batch_size=512,collate_fn=collate_fn,shuffle=False)
    
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
        patience=copy.deepcopy(pOG)
        
        print('\nRun', r + 1)
        
        #MODEL
        model = BertCstm(bert, hsize, len(redux)+1,dout=dout)
        model.to(device)
        
        #OPTIM
        optimizer = optim.AdamW(model.parameters(),lr=lr)
        
        #TRAIN AND EVAL
        for epoch in range(0,num_epochs):
            print('E:',epoch+1)
            
            trainLoop(model,train_ds,train_dl,device,optimizer,critAsp,clip)

            bestVL,vl,bestModel = evalLoop(model,valid_ds,valid_dl,device,critAsp,bestModel,bestVL)
            
            if vl > bestVL:
                patience-=1
            else:
                patience=copy.deepcopy(pOG)
                
            if patience<=0:
                print('Patience reached!')
                break
            
            gc.collect()
            torch.cuda.empty_cache()
            
        #TESTING
        aspm,polm,allm,finalModel = testLoop(bestModel,test_ds,test_dl,device,critAsp,tokenizer,idx2pol,bestTL,finalModel)
        
        #allm = metricsAll(predictions['allP'],predictions['allT'])
        met.append(allm)
        amet.append(aspm)
        pmet.append(polm)
        
    #Metrics of all runs
    #finalEval(met,amet,pmet)    
    finalEval(amet,'Aspects')
    finalEval(pmet,'Polarity')
    finalEval(met,'Joint')
        
    #save best model of all runs
    torch.save(finalModel.state_dict(),'best.pth')