# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
from conllm import evaluate
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import copy
                    
def train_loop(train_dl,train_ds,device,optimizer,model,criterion_intents,
               criterion_slots,epoch,n_epochs,lang,tokenizer,dev_dl,patience,
               best_f1,ogP):
    """
    Training loop for the model.

    Args:
    - train_dl: Training data loader.
    - train_ds: Training dataset.
    - device: Device on which to train the model.
    - optimizer: Model optimizer.
    - model: The model to be trained.
    - criterion_intents: Criterion for intent classification.
    - criterion_slots: Criterion for slot classification.
    - epoch: Current epoch number.
    - n_epochs: Total number of epochs.
    - lang: Vocabulary and label mappings (output of Lang() class).
    - tokenizer: Tokenizer for text data.
    - dev_dl: Development data loader for evaluation.
    - patience: Patience parameter for early stopping.
    - best_f1: Best F1 score achieved so far.

    Returns:
    updated patience and best F1 score.
    """
    total_loss=0
    for _, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
        
        utt = data['utterances'].to(device)
        attm = data['text_attention_mask']#.to(device)
        slot = data['y_slots'].to(device)
        intent = data['intents'].to(device)
    
        optimizer.zero_grad()
    
        predS, predI = model(utt,attm)
        
        lossS = criterion_slots(predS, slot)
        lossI = criterion_intents(predI, intent)

        # Backpropagation and optimization
        loss = lossS + lossI
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    results, report_intent, loss_array = eval_loop(dev_dl, criterion_slots,
                                                   criterion_intents, model,
                                                   lang, tokenizer)
    
    f1 = results['total']['f']
    #PATIENCE CODE
    if True:#epoch%2==0:
        
        if f1/total_loss > best_f1 and f1 > 0.90: #just to avoid entering here in the first epochs
            best_f1 = f1/total_loss
            torch.save(model.state_dict(), 'best.pth')
            patience=copy.deepcopy(ogP)
            print('Best E',epoch,patience)
        else:
            patience-=1
            
    print(f'Epoch [{epoch}/{n_epochs}], Loss: {total_loss / len(train_dl)}, F1: {f1}')
            
    return patience,best_f1

def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    """
    Evaluation loop for the model.

    Args:
    - data: Evaluation data.
    - criterion_slots: Criterion for slot classification.
    - criterion_intents: Criterion for intent classification.
    - model: The model to be evaluated.
    - lang: Vocabulary and label mappings (output of Lang() class).
    - tokenizer: Tokenizer for text data.

    Returns:
    Tuple containing evaluation results, intent classification report, and loss array.
    """
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): 
        for sample in data:
            attm = sample['text_attention_mask']
            slots, intents = model(sample['utterances'], attm)
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                
                #Get the lenght of the slots w cls pad
                length = sample['slots_len'].tolist()[id_seq]                
                #remove the batch padding
                predSlot = seq[:length]
                #remove the cls padding
                predSlot = predSlot[1:-1]
                #turn into list
                predSlot = predSlot.tolist()
                goalSlot = sample['slots'][id_seq].tolist()[1:-1]
                
                #Get the slots in text
                txtpredSlot = [lang.id2slot[elem] for elem in predSlot]
                txtgoalSlot = [lang.id2slot[elem] for elem in goalSlot]
                #Get the utter in text
                txtutter = tokenizer.decode(sample['utterance'][id_seq][1:-1]).split(" ")
                #txtgoalUtter = sample['utt'][id_seq].split(" ")
                
                #Format targets as a list of tupples
                ref_slots.append([(txtutter[id_el], elem) for id_el, elem in enumerate(txtgoalSlot)])
                #Format predic as a list of tupples
                hyp_slots.append([(txtutter[id_el], elem) for id_el, elem in enumerate(txtpredSlot)])
                
                # #utt_ids = sample['utterance'][id_seq][:length].tolist()
                # gt_ids = sample['y_slots'][id_seq].tolist()
                # if sample['pad'][id_seq] != 0: #handle padding
                #     gt_ids = gt_ids[:-sample['pad'][id_seq]]
                #     seq = seq[:len(gt_ids)]
                # gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                # utterance = sample['utt'][id_seq].split() #[tokenizer.decode(elem) for elem in utt_ids][1:-1] #remove cls and sep #[lang.id2word[elem] for elem in utt_ids]
                # to_decode = seq[:length].tolist()
                # ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                # tmp_seq = []
                # for id_el, elem in enumerate(to_decode):
                #     if lang.id2slot[elem] == 'O':
                #         sl = 'O-O'
                #     else:
                #         sl = lang.id2slot[elem]
                #     tmp_seq.append((utterance[id_el], sl))
                # hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array



def evalModel(model, test_loader, criterion_slots, criterion_intents,lang,
              sltm,intm,tokenizer):
    """
    Evaluate the model on a test set.

    Args:
    - model: The model to be evaluated.
    - test_loader: Test data loader.
    - criterion_slots: Criterion for slot classification.
    - criterion_intents: Criterion for intent classification.
    - lang: Vocabulary and label mappings (output of Lang() class).
    - sltm: List to store slot-related metrics.
    - intm: List to store intent-related metrics.
    - tokenizer: Tokenizer for text data.

    Returns:
    Tuple containing updated slot-related metrics and intent-related metrics.
    """
    res_slt, res_int, _  = eval_loop(test_loader, criterion_slots, 
                                             criterion_intents, model, lang,tokenizer)

    sltm[0].append(res_slt['total']['f'])
    sltm[1].append(res_slt['total']['p'])
    sltm[2].append(res_slt['total']['r'])
    intm[0].append(res_int['weighted avg']['f1-score'])
    intm[1].append(res_int['weighted avg']['precision'])
    intm[2].append(res_int['weighted avg']['recall'])

    return sltm, intm

def printRes(sltm, intm, name):
    """
    Print evaluation results.

    Args:
    - sltm: Slot-related metrics.
    - intm: Intent-related metrics.
    - name: Name of the evaluation set.

    Returns:
    None
    """
    print('\n', name)
    n = ['F1','Precision','Recall']
    for i in range(0,3):
        smet = np.array(sltm[i])
        imet = np.array(intm[i])
        print(n[i]+':', 'S:', round(smet.mean(),4),'+-', round(smet.std(),4),
              'I:', round(imet.mean(),4),'+-', round(imet.std(),4))