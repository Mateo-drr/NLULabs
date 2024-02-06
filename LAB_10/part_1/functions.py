# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
from conllm import evaluate
from sklearn.metrics import classification_report
import numpy as np
import torch.optim as optim
import copy

PAD_TOKEN = 0

def reset(model,lr):
    """
    Reset the training parameters.

    Args:
    - model: PyTorch model.
    - lr: Learning rate.

    Returns:
    - optimizer: Optimizer with reset parameters.
    - criterion_slots: CrossEntropyLoss for slot classification.
    - criterion_intents: CrossEntropyLoss for intent classification.
    - losses_train: List to store training losses.
    - losses_dev: List to store development losses.
    - sampled_epochs: List to store sampled epochs.
    - best_f1: to store best F1 score.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    return optimizer,criterion_slots,criterion_intents,losses_train,losses_dev,sampled_epochs,best_f1

def init_weights(mat):
    """
    Initialize weights and biases of the specified module.

    Args:
    - mat: PyTorch module.
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
                    
def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    """
    Training loop for the custom model.

    Args:
    - data: Training data loader.
    - optimizer: Model optimizer.
    - criterion_slots: Criterion for slot classification.
    - criterion_intents: Criterion for intent classification.
    - model: The model to be trained.

    Returns:
    - List of training losses.
    """
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Evaluation loop for the custom model.

    Args:
    - data: Evaluation data loader.
    - criterion_slots: Criterion for slot classification.
    - criterion_intents: Criterion for intent classification.
    - model: The model for evaluation.

    Returns:
    - Evaluation results, classification report, and list of losses.
    """
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    # hypraw=[[],[]]
    # refraw=[[],[]]
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
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
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

            # hypraw[0].append(torch.argmax(intents, dim=1).to('cpu'))
            # hypraw[1].append(output_slots.to('cpu'))
            # refraw[0].append(sample['intents'].to('cpu'))
            # refraw[1].append(sample['y_slots'].to('cpu'))
            
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        print('RERUN CODE!!')
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array#, (hypraw,refraw)

def trainModel(model,n_epochs,train_loader,optimizer,criterion_slots,criterion_intents,
             sampled_epochs,losses_train, dev_loader, lang, losses_dev, best_f1,patience):
    """
    Train the custom model.

    Args:
    - model: The custom model.
    - n_epochs: Total number of training epochs.
    - train_loader: Training data loader.
    - optimizer: Model optimizer.
    - criterion_slots: Criterion for slot classification.
    - criterion_intents: Criterion for intent classification.
    - sampled_epochs: List to store sampled epochs for evaluation.
    - losses_train: List to store training losses.
    - dev_loader: Development data loader for evaluation.
    - lang: Vocabulary and label mappings (output of Lang() class).
    - losses_dev: List to store development losses.
    - best_f1: Best F1 score achieved so far.
    - patience: Patience parameter for early stopping.

    Returns:
    - Trained model.
    """
    pOG=copy.deepcopy(patience)
    for x in range(1,n_epochs):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                          criterion_intents, model)
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                          criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev['total']['f']

            if f1 > best_f1:
                best_f1 = f1
                patience=copy.deepcopy(pOG)
            else:
                patience -= 1
            if patience <= 0: # Early stoping with patient
                print('Patience reached E:', x)
                break # Not nice but it keeps the code clean
                
    return model

def evalModel(model, test_loader, criterion_slots, criterion_intents,lang,
              sltm,intm):
    """
    Evaluate the custom model on the test set.

    Args:
    - model: The trained custom model.
    - test_loader: Test data loader.
    - criterion_slots: Criterion for slot classification.
    - criterion_intents: Criterion for intent classification.
    - lang: Language-related information.
    - sltm: List to store slot metrics.
    - intm: List to store intent metrics.

    Returns:
    - Updated slot and intent metric lists.
    """
    res_slt, res_int, _  = eval_loop(test_loader, criterion_slots, 
                                             criterion_intents, model, lang)
    
    #runMetrics(refraw[0],hypraw[0],refraw[1],hypraw[1])
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
        print(n[i]+':', 'S:', round(smet.mean(),4),'+-', round(smet.std(),3),
              'I:', round(imet.mean(),4),'+-', round(imet.std(),3))
