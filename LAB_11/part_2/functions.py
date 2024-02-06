from tqdm import tqdm
import torch
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import chain
import numpy as np
from utils import pol2idx,idx2pol

def trainLoop(model,train_ds,train_dl,device,optimizer,critAsp,clip):
    """
    Training loop for the model.

    Args:
        model: Model to be trained.
        train_ds (Dataset): Training dataset.
        train_dl (DataLoader): Training data loader.
        device: Device on which to perform training.
        optimizer: Model optimizer.
        critAsp: Aspect loss criterion.
        clip (float): Gradient clipping value.

    Returns:
        None
    """
    lossA = 0
    lossP = 0
    tloss = 0
    model.train()
    for _,data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
        utt = data['utterances'].to(device)
        attm = data['text_attention_mask']
        asp = data['y_lbl'].to(device)

        optimizer.zero_grad()
        
        predS = model(utt,attm)
        
        lossS = critAsp(predS, asp)
        
        # Backpropagation and optimization
        loss = lossS
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
        
        tloss += loss.item()
        
    print('Train',
          'L',round(tloss/len(train_ds),6),
          )
    
def uncoll(pred,sample):
    
    pred = torch.argmax(pred, dim=1)
    ref_asps=[]
    hyp_asps=[]
    for id_seq, seq in enumerate(pred):
        
        #Get the lenght of the asps w cls pad
        length = sample['asp_len'].tolist()[id_seq]                
        #remove the batch padding
        predasp = seq[:length]
        #remove the cls padding
        predasp = predasp[1:-1]
        #turn into list
        predasp = predasp.tolist()
        goalasp = sample['y_lbl'][id_seq].tolist()[:length][1:-1]
        
        ref_asps.append(goalasp)
        hyp_asps.append(predasp)
        
    
    return ref_asps,hyp_asps
    
def evalLoop(model,valid_ds,valid_dl,device,critAsp,bestModel,bestVL):
    """
    Validation loop for the model.

    Args:
        model: Model to be evaluated.
        valid_ds (Dataset): Validation dataset.
        valid_dl (DataLoader): Validation data loader.
        device: Device on which to perform evaluation.
        critAsp: Aspect loss criterion.
        bestModel: Best model achieved so far.
        bestVL (float): Best validation loss achieved so far.

    Returns:
        tuple: Tuple containing the best validation loss and the best model.
    """
    lossA = 0
    tloss = 0
    predA=[]
    targA=[]
    model.eval()
    with torch.no_grad():
        for _,data in tqdm(enumerate(valid_dl), total=int(len(valid_ds)/valid_dl.batch_size)):
            lbl = data['y_lbl'].to(device)
            text = data['utterances'].to(device)
            att = data['text_attention_mask'].to(device)
    
            asp = model(text,att)
            
            loss1 = critAsp(asp,lbl)
            lossA += loss1.item()
    
            loss= loss1
            
            tloss +=loss.item()
            
            lbl,asp = uncoll(asp, data)
            
            predA.append(asp)

            targA.append(lbl)

        vl = tloss/len(valid_ds)      

        print('Valid',
              'L',round(vl,6),
              )
        _,_,joint = metrics(predA, targA)
        
        vl = vl/joint['F1-score'] #make the patience work with f1 in mind too
        
        if vl < bestVL:
            bestVL = copy.deepcopy(vl)
            bestModel = copy.deepcopy(model).cpu()
        print('Best score:',bestVL)
        
        print('------------------------------------------------------')
        
    return bestVL,vl,bestModel


def testLoop(bestModel,test_ds,test_dl,device,critAsp,tokenizer,idx2pol,bestTL,finalModel):
    """
    Testing loop for the model.

    Args:
        bestModel: Best model achieved during training.
        test_ds (Dataset): Test dataset.
        test_dl (DataLoader): Test data loader.
        device: Device on which to perform testing.
        critAsp: Aspect loss criterion.
        tokenizer: Tokenizer used for encoding aspects.
        idx2pol (dict): Dictionary mapping polarity indices to strings.
        bestTL (float): Best test loss achieved so far.
        finalModel: best model between runs.

    Returns:
        tuple: Tuple containing predictions, aspect metrics, polarity metrics, and the final model.
    """
    lossA = 0
    tloss = 0
    predA=[]
    targA=[]
    model = bestModel
    model.to(device)
    model.eval()
    with torch.no_grad():
        for _,data in tqdm(enumerate(test_dl), total=int(len(test_ds)/test_dl.batch_size)):
            lbl = data['y_lbl'].to(device)
            text = data['utterances'].to(device)
            att = data['text_attention_mask'].to(device)
    
            asp = model(text,att)
            
            loss1 = critAsp(asp,lbl)
            lossA += loss1.item()
    
            loss= loss1
            
            tloss +=loss.item()
            
            lbl,asp = uncoll(asp, data)
            
            predA.append(asp)
            targA.append(lbl)
    
    print('\nTest',
          'L',round(tloss/len(test_ds),6),
          )
    aspm,polm,allm = metrics(predA,targA)
    
    if tloss/len(test_ds) < bestTL:
        bestTL = tloss/len(test_ds)
        finalModel = copy.deepcopy(model).cpu()
    
    return aspm,polm,allm,finalModel

def metrics(predA, targA):
    """
    Calculate precision, recall, and F1-score for aspect and polarity predictions.

    Args:
        predA (torch.Tensor): Predicted aspect values.
        predP (torch.Tensor): Predicted polarity values.
        targA (torch.Tensor): Target aspect values (ground truth).
        targP (torch.Tensor): Target polarity values (ground truth).

    Returns:
        tuple: Tuple containing dictionaries of aspect and polarity metrics.
    """
    
    #Flatten and round outputs
    pred = list(chain.from_iterable(chain.from_iterable(predA)))
    targ = list(chain.from_iterable(chain.from_iterable(targA)))
    
    #Make aspect predition binary ie only the position of the aspects as 1s
    predA = [0 if x == 1 else 1 for x in pred]
    targA = [0 if x == 1 else 1 for x in targ]
    #For polarity only evaluate the predicted polarity of the aspects, ie remove all predictions of non aspects
    predP = []
    targP = []
    for i in range(len(targ)):
        #find the aspect positions and get their pred pol
        if targ[i] != 1:
            predP.append(pred[i])
            targP.append(targ[i])
    
    #Joint simpy use the direct outputs of the model

    aspect_metrics = {
        'Precision': round(precision_score(targA, predA, average='macro',zero_division=0), 6),
        'Recall': round(recall_score(targA, predA, average='macro',zero_division=0), 6),
        'F1-score': round(f1_score(targA, predA, average='macro',zero_division=0), 6),
    }

    polarity_metrics = {
        'Precision': round(precision_score(targP, predP, average='macro',zero_division=0), 6),
        'Recall': round(recall_score(targP, predP, average='macro',zero_division=0), 6),
        'F1-score': round(f1_score(targP, predP, average='macro',zero_division=0), 6),
    }
    
    joint_metrics = {
        'Precision': round(precision_score(targ, pred, average='macro',zero_division=0), 6),
        'Recall': round(recall_score(targ, pred, average='macro',zero_division=0), 6),
        'F1-score': round(f1_score(targ, pred, average='macro',zero_division=0), 6),
    }

    print('Aspects:',
          'Precision:', aspect_metrics['Precision'],
          'Recall:', aspect_metrics['Recall'],
          'F1-score:', aspect_metrics['F1-score']
          )

    print('Polarity:',
          'Precision:', polarity_metrics['Precision'],
          'Recall:', polarity_metrics['Recall'],
          'F1-score:', polarity_metrics['F1-score']
          )
    
    print('Joint:',
          'Precision:', joint_metrics['Precision'],
          'Recall:', joint_metrics['Recall'],
          'F1-score:', joint_metrics['F1-score']
          )

    return aspect_metrics, polarity_metrics, joint_metrics
    

def finalEval(metrics_list,name):
    metrics_array = np.array([[d['Precision'], d['Recall'], d['F1-score']] for d in metrics_list])

    average_metrics = np.mean(metrics_array, axis=0)
    std_metrics = np.std(metrics_array, axis=0)

    print(f"\n{name} Metrics:")
    print(f"Precision: {average_metrics[0]:.4f} +- {std_metrics[0]:.4f}")
    print(f"Recall: {average_metrics[1]:.4f} +- {std_metrics[1]:.4f}")
    print(f"F1-score: {average_metrics[2]:.4f} +- {std_metrics[2]:.4f}")
