from tqdm import tqdm
import torch
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import chain
import numpy as np

def metricsAll(pred, targ):
    """
    Calculate precision, recall, and F1-score for aspect and polarity predictions combined.

    Args:
        pred (list): List of predicted values.
        targ (list): List of target (ground truth) values.

    Returns:
        dict: Dictionary containing precision, recall, and F1-score.
    """
    predf = list(chain.from_iterable(list(chain.from_iterable(pred))))
    targf = list(chain.from_iterable(list(chain.from_iterable(targ))))

    # Calculate precision, recall, and F1-score
    precision = precision_score(targf, predf, average='macro',zero_division=0)
    recall = recall_score(targf, predf, average='macro',zero_division=0)
    f1 = f1_score(targf, predf, average='macro',zero_division=0)

    allmetrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
    }

    print('\nJoint Metrics:',
          'Precision:', round(precision, 6),
          'Recall:', round(recall, 6),
          'F1-score:', round(f1, 6)
          )
    return allmetrics

def extractAspPol(test_ds, aspect_positions, polarity, idx2pol, tokenizer):
    """
    Formatting of the model output data and the ground truth values.

    Args:
        test_ds (Dataset): Test dataset.
        aspect_positions (torch.Tensor): Tensor containing predicted aspect positions.
        polarity (torch.Tensor): Tensor containing predicted polarity values.
        idx2pol (dict): Dictionary mapping polarity indices to strings.
        tokenizer: Tokenizer used for encoding aspects.

    Returns:
        dict: Dictionary containing aspect and polarity for evaluation.
    """
    
    aspect_positions = torch.round(aspect_positions).to(torch.int).cpu()
    polarity = torch.round(polarity).to(torch.int).cpu()
    
    predAspectsS = [[] for _ in range(len(test_ds))]
    predAspectsT = [[] for _ in range(len(test_ds))]
    targAspectsS = [[] for _ in range(len(test_ds))]
    targAspectsT = [[] for _ in range(len(test_ds))]
    
    predPolarityS = [[] for _ in range(len(test_ds))]
    predPolarityT = [[] for _ in range(len(test_ds))]
    targPolarityS = [[] for _ in range(len(test_ds))]
    targPolarityT = [[] for _ in range(len(test_ds))]
    
    allPred = [[] for _ in range(len(test_ds))]
    allTarg = [[] for _ in range(len(test_ds))]
    
    for i in range(len(test_ds)):
        for j in range(0,aspect_positions[i].shape[0]):
            
            #PREDICTIONS
            start, end = aspect_positions[i][j]
            # Exclude positions where both start and end are zero (padding)
            # and if the prediction is illogical
            if (start != 0 or end != 0) and start <= end:
                
                # Extract the aspect from the sentence based on the predicted positions
                aspect = test_ds[i]['s'].split(" ")[start-1:end]
                aspect = ' '.join(aspect)
                predAspectsS[i].append(aspect)
                # Get the tokenized version
                aspect = tokenizer.encode(aspect)
                predAspectsT[i].append(aspect)
                
                #Get combined data to evaluate, just take it if an aspect was predicted
                allPred[i].append([start.item(), end.item(), polarity[i][j].item()])
                
            #Get the polarity in string
            polar = polarity[i][j]
            predPolarityS[i].append(idx2pol[polar.item()])
            
            #Get the polarity in idx
            predPolarityT[i].append(polar)
                
                
            #GROUND TRUTH
            start,end= test_ds[i]['asp'][j] #get the correct positions
            if start != 0 or end != 0:    
                
                # Extract the aspect from the sentence based on the positions
                aspect = test_ds[i]['s'].split(" ")[start-1:end]
                aspect = ' '.join(aspect)
                targAspectsS[i].append(aspect)
                
                # Get the tokenized version
                aspect = tokenizer.encode(aspect)
                targAspectsT[i].append(aspect)
                
                #Get combined data to evaluate, just take it if an aspect was predicted
                allTarg[i].append([start.item(), end.item(), polarity[i][j].item()])
                
            #Get the polarity in string
            polar = test_ds[i]['pol'][j]
            targPolarityS[i].append(idx2pol[polar.item()])
            
            #Get the polarity in idx
            targPolarityT[i].append(polar)
            
            
        
        #Remove duplicates
        clean = []
        for p in allPred[i]:
            if p not in clean:
                clean.append(p)
        
        allPred[i] = clean
        
        # Pad predictions and targets to allow comparison
        max_len = max(len(allPred[i]), len(allTarg[i]))
        allPred[i] = allPred[i] + [[0, 0, 0]] * (max_len - len(allPred[i]))
        allTarg[i] = allTarg[i] + [[0, 0, 0]] * (max_len - len(allTarg[i]))

    return {'aspP':predAspectsS,
            'aspPt':predAspectsT,
            'aspT':targAspectsS,
            'aspTt':targAspectsT,
            'polP':predPolarityS,
            'polPt':predPolarityT,
            'polT':targPolarityS,
            'polTt':targPolarityT,
            'allP':allPred,
            'allT':allTarg
            }

def trainLoop(model,train_ds,train_dl,device,optimizer,critAsp,critPol,clip):
    """
    Training loop for the model.

    Args:
        model: Model to be trained.
        train_ds (Dataset): Training dataset.
        train_dl (DataLoader): Training data loader.
        device: Device on which to perform training.
        optimizer: Model optimizer.
        critAsp: Aspect loss criterion.
        critPol: Polarity loss criterion.
        clip (float): Gradient clipping value.

    Returns:
        None
    """
    lossA = 0
    lossP = 0
    tloss = 0
    model.train()
    for _,data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
        lbl = data['pol'].to(device)
        lblA = data['asp'].to(device)#['asp_enc'].to(device)#.to(torch.float32)
        text = data['text_input_ids'].to(device)
        att = data['text_attention_mask'].to(device)
        
        optimizer.zero_grad()
        asp,pol = model(text,att)
        
        loss1 = critAsp(asp,lblA)
        lossA += loss1.item()
        
        loss2 = critPol(pol,lbl)
        lossP += loss2.item()
        
        loss= loss1+loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
        
        tloss +=loss.item()
        
    print('Train',
          'L',round(tloss/len(train_ds),6),
          'Asp-L',round(lossA/len(train_ds),6),
          'Pol-L',round(lossP/len(train_ds),6)
          )
    
def evalLoop(model,valid_ds,valid_dl,device,critAsp,critPol,bestModel,bestVL):
    """
    Validation loop for the model.

    Args:
        model: Model to be evaluated.
        valid_ds (Dataset): Validation dataset.
        valid_dl (DataLoader): Validation data loader.
        device: Device on which to perform evaluation.
        critAsp: Aspect loss criterion.
        critPol: Polarity loss criterion.
        bestModel: Best model achieved so far.
        bestVL (float): Best validation loss achieved so far.

    Returns:
        tuple: Tuple containing the best validation loss and the best model.
    """
    lossA = 0
    lossP = 0
    tloss = 0
    predP=[]
    predA=[]
    targP=[]
    targA=[]
    model.eval()
    with torch.no_grad():
        for _,data in tqdm(enumerate(valid_dl), total=int(len(valid_ds)/valid_dl.batch_size)):
            lbl = data['pol'].to(device)
            lblA = data['asp'].to(device)#['asp_enc'].to(device)#.to(torch.float32)
            text = data['text_input_ids'].to(device)
            att = data['text_attention_mask'].to(device)
    
            asp,pol = model(text,att)
            
            loss1 = critAsp(asp,lblA)
            lossA += loss1.item()
            
            loss2 = critPol(pol,lbl)
            lossP += loss2.item()
            
            loss= loss1+loss2
            
            tloss +=loss.item()
            
            #asp,pol = torch.argmax(asp,dim=1),torch.argmax(pol,dim=1)
            predA.append(asp)
            predP.append(pol)
            targA.append(lblA)
            targP.append(lbl)
            

        predA,predP,targA,targP=torch.cat(predA,dim=0).cpu(),torch.cat(predP,dim=0).cpu(),torch.cat(targA,dim=0).cpu(),torch.cat(targP,dim=0).cpu() 

        print('Valid',
              'L',round(tloss/len(valid_ds),6),
              'Asp-L',round(lossA/len(valid_ds),6),
              'Pol-L',round(lossP/len(valid_ds),6),
              #'Asp-F1',round(f1_score(targA,predA,average='macro'),6),
              #'Pol-F1',round(f1_score(targP,predP,average='macro'),6)
              )
        _ = metrics(predA, predP, targA, targP)
        
        
        if tloss/len(valid_ds) < bestVL:
            bestVL = tloss/len(valid_ds)
            bestModel = copy.deepcopy(model).cpu()
        print('BestL:',bestVL)
        
        print('------------------------------------------------------')
        
    return bestVL,bestModel


def testLoop(bestModel,test_ds,test_dl,device,critAsp,critPol,tokenizer,idx2pol,bestTL,finalModel):
    """
    Testing loop for the model.

    Args:
        bestModel: Best model achieved during training.
        test_ds (Dataset): Test dataset.
        test_dl (DataLoader): Test data loader.
        device: Device on which to perform testing.
        critAsp: Aspect loss criterion.
        critPol: Polarity loss criterion.
        tokenizer: Tokenizer used for encoding aspects.
        idx2pol (dict): Dictionary mapping polarity indices to strings.
        bestTL (float): Best test loss achieved so far.
        finalModel: best model between runs.

    Returns:
        tuple: Tuple containing predictions, aspect metrics, polarity metrics, and the final model.
    """
    lossA = 0
    lossP = 0
    tloss = 0
    predP=[]
    predA=[]
    targP=[]
    targA=[]
    model = bestModel
    model.to(device)
    model.eval()
    with torch.no_grad():
        for _,data in tqdm(enumerate(test_dl), total=int(len(test_ds)/test_dl.batch_size)):
            lbl = data['pol'].to(device)
            lblA = data['asp'].to(device)#['asp_enc'].to(device)#.to(torch.float32)
            text = data['text_input_ids'].to(device)
            att = data['text_attention_mask'].to(device)
    
            asp,pol = model(text,att)
            
            loss1 = critAsp(asp,lblA)
            lossA += loss1.item()
            
            loss2 = critPol(pol,lbl)
            lossP += loss2.item()
            
            loss= loss1+loss2
            
            tloss +=loss.item()
            
            #asp,pol = torch.argmax(asp,dim=1),torch.argmax(pol,dim=1)
            predA.append(asp)
            predP.append(pol)
            targA.append(lblA)
            targP.append(lbl)
    
    predA,predP,targA,targP=torch.cat(predA,dim=0).cpu(),torch.cat(predP,dim=0).cpu(),torch.cat(targA,dim=0).cpu(),torch.cat(targP,dim=0).cpu() 
    
    predictions = extractAspPol(test_ds, predA, predP, idx2pol, tokenizer)
    
    print('\nTest',
          'L',round(tloss/len(test_ds),6),
          'Asp-L',round(lossA/len(test_ds),6),
          'Pol-L',round(lossP/len(test_ds),6),
          #'Asp-F1',round(f1_score(targA,predA,average='macro'),6),
          #'Pol-F1',round(f1_score(targP,predP,average='macro'),6)
          )
    print('\nIndividual Metrics: ')
    aspm,polm = metrics(predA, predP, targA, targP)
    
    if tloss/len(test_ds) < bestTL:
        bestTL = tloss/len(test_ds)
        finalModel = copy.deepcopy(model).cpu()
    
    return predictions, aspm,polm,finalModel

def metrics(predA, predP, targA, targP):
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
    predA, predP, targA, targP = predA.flatten(), predP.flatten(), targA.flatten(), targP.flatten()
    predA, predP = torch.round(predA).to(torch.int), torch.round(predP).to(torch.int)

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

    return aspect_metrics, polarity_metrics
    

def finalEval(joint,asp,pol):
    """
    Evaluate and print average and standard deviation of precision, recall, and F1-score.

    Args:
        joint (list): List of dictionaries containing metrics for the first set of results.
        asp (list): List of dictionaries containing metrics for the second set of results.
        pol (list): List of dictionaries containing metrics for the third set of results.

    Returns:
        None
    """
    
    # Extract precision, recall, and F1 scores from the lists of dictionaries
    metrics_lists = {'Precision': [], 'Recall': [], 'F1-score': []}

    for results_list in [joint, asp, pol]:
        for metric in metrics_lists.keys():
            metrics_lists[metric].append([item[metric] for item in results_list])

    # Calculate average and standard deviation for each metric
    metrics_avg_std = {}

    for metric, values_list in metrics_lists.items():
        avg_values = np.mean(values_list, axis=0)
        std_values = np.std(values_list, axis=0)

        metrics_avg_std[metric] = {
            'average': avg_values,
            'std_dev': std_values
        }

    # Print the results
    print('\nMetric\t\tJoint\t\tAspects\t\tPolarity')
    print('-' * 40)

    for metric, values in metrics_avg_std.items():
        print(f'{metric.capitalize()}:\t{values["average"][0]:.6f} +- {values["std_dev"][0]:.6f}\t'
              f'{values["average"][1]:.6f} +- {values["std_dev"][1]:.6f}\t'
              f'{values["average"][2]:.6f} +- {values["std_dev"][2]:.6f}')
