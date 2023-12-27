# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
from conllm import evaluate
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

def init_weights(mat):
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
                    
def train_loop(train_dl,train_ds,device,optimizer,model,criterion_intents,
               criterion_slots,epoch,n_epochs,lang,tokenizer,dev_dl,patience,
               best_f1):
    total_loss=0
    for _, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
        
        utt = data['utterances'].to(device)
        slot = data['y_slots'].to(device)
        intent = data['intents'].to(device)
    
        optimizer.zero_grad()
    
        predS, predI = model(utt)
        
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
    if epoch%2==0:
        
        if f1 > best_f1:
            best_f1 = f1
        else:
            patience-=1
            
    print(f'Epoch [{epoch}/{n_epochs}], Loss: {total_loss / len(train_dl)}, F1: {f1}')
            
    return patience,best_f1

def evalModel(model, test_loader, criterion_slots, criterion_intents,lang,
              slot_f1s,intent_acc, tokenizer):
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                             criterion_intents, model, lang, tokenizer)
    intent_acc.append(intent_test['accuracy'])
    slot_f1s.append(results_test['total']['f'])
    return slot_f1s, intent_acc

def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'])
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
                length = len(sample['utt'][id_seq].split())#sample['slots_len'].tolist()[id_seq]
                #utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                if sample['pad'][id_seq] != 0: #handle padding
                    gt_ids = gt_ids[:-sample['pad'][id_seq]]
                    seq = seq[:len(gt_ids)]
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = sample['utt'][id_seq].split() #[tokenizer.decode(elem) for elem in utt_ids][1:-1] #remove cls and sep #[lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    if lang.id2slot[elem] == 'O':
                        sl = 'O-O'
                    else:
                        sl = lang.id2slot[elem]
                    tmp_seq.append((utterance[id_el], sl))
                hyp_slots.append(tmp_seq)
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


def printRes(slot_f1s, intent_acc, name):
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('\n', name)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))