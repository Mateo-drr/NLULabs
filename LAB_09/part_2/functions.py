# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import math

from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import spacy
import torch.optim as optim
nlp = spacy.load('en_core_web_lg')

def train_loop(data, optimizer, criterion, model, clip=5):
    """
    Trains the provided model on the given data using the specified optimizer
    and criterion.

    Parameters:
    - data (iterable): dataloader
    - optimizer (torch.optim.Optimizer): The optimizer to use for training.
    - criterion (torch.nn.Module): The loss criterion for training.
    - model (nn.Module): The PyTorch model to be trained.
    - clip (float): Gradient clipping threshold (default is 5).

    Returns:
    - float: Average training loss per token.
    """
    model.train()
    loss_array = []
    number_of_tokens = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return sum(loss_array)/sum(number_of_tokens)

def train_loopNT(data, optimizer, criterion, model, config, dev_loader, eval_criterion, clip=5):
    """
    Trains the model with NT-ASGD.

    Parameters:
    - data (iterable): Iterable containing training samples.
    - optimizer (torch.optim.Optimizer): The optimizer to use for training.
    - criterion (torch.nn.Module): The loss criterion for training.
    - model (nn.Module): The PyTorch model to be trained.
    - config (dict): Dictionary containing various optimzer data.
    - dev_loader (iterable): Iterable containing development/validation samples.
    - eval_criterion (torch.nn.Module): The evaluation loss criterion.
    - clip (float): Gradient clipping threshold (default is 5).

    Returns:
    - float: Average training loss per token.
    """
    model.train()
    loss_array = []
    number_of_tokens = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    #NT-ASGD
    #Simply change to ASGD when the NT condition is met
    _,val_loss = eval_loop(dev_loader, eval_criterion, model)
    if 't0' not in optimizer.param_groups[0] and (len(config['logs'])>config['n'] and val_loss > min(config['logs'][:-config['n']])):
            print('\nNT condition met, switching to ASGD', optimizer)
            optimizer = torch.optim.ASGD(model.parameters(), lr=config['lr'], t0=0, lambd=0., weight_decay=1.2e-6)
            print(optimizer)
    config['logs'].append(val_loss)
        
    return sum(loss_array)/sum(number_of_tokens),optimizer


def eval_loop(data, eval_criterion, model):
    """
    Evaluates the model on the eval data.

    Parameters:
    - data (iterable): Iterable containing evaluation samples.
    - eval_criterion (torch.nn.Module): The evaluation loss criterion.
    - model (nn.Module): The PyTorch model to be evaluated.

    Returns:
    - tuple: A tuple containing perplexity and average evaluation loss per token.
    """
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    """
    Initializes the weights of the specified PyTorch module.

    Parameters:
    - mat (nn.Module): PyTorch module to initialize weights.
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
                    
                    
def analogy_spacy(w1, w2, w3):
    """
    Performs word analogy using spaCy word vectors.

    Parameters:
    - w1, w2, w3 (str): Input words for analogy.
    """
    v1 = nlp.vocab[w1].vector
    v2 = nlp.vocab[w2].vector
    v3 = nlp.vocab[w3].vector
    
    # relation vector
    rv = v3 + v2 - v1
   
    # n=1 & sorted by default
    ms = nlp.vocab.vectors.most_similar(np.asarray([rv]), n=10)
    
    # getting words & scores
    for i, key in enumerate(ms[0][0]):
        print(nlp.vocab.strings[key], ms[2][0][i])
        
def analogy_our_model(w1, w2, w3, model, lang):
    """
    Performs word analogy using the provided PyTorch model.

    Parameters:
    - w1, w2, w3 (str): Input words for analogy.
    - model (nn.Module): The PyTorch model.
    - lang: Language model containing word-to-id and id-to-word mappings.
    """
    model.eval().to('cpu')
    tmp_w1 = torch.LongTensor([lang.word2id[w1] if w1 in lang.word2id else lang.word2id['<unk>']]) 
    tmp_w2 = torch.LongTensor([lang.word2id[w2] if w2 in lang.word2id else lang.word2id['<unk>']])
    tmp_w3 = torch.LongTensor([lang.word2id[w3] if w3 in lang.word2id else lang.word2id['<unk>']])
    
    v1 = model.get_word_embedding(tmp_w1)
    v2 = model.get_word_embedding(tmp_w2)
    v3 = model.get_word_embedding(tmp_w3)
    # relation vector
    rv = v3 + v2 - v1

    # n=1 & sorted by default
    ms = model.get_most_similar(rv)

    
    # getting words & scores
    for i, key in enumerate(ms[0]):
        print(lang.id2word[key], ms[1][i])
        
        
def runModel(n_epochs, optimizer, criterion_train, criterion_eval, model, clip,
             dev_loader,train_loader,test_loader,lang, device, ntasgd=False,
             patience=10,config=None):
    """
    Train and eval the PyTorch model.

    Parameters:
    - n_epochs (int): Number of training epochs.
    - optimizer (torch.optim.Optimizer): The optimizer for training.
    - criterion_train (torch.nn.Module): The training loss criterion.
    - criterion_eval (torch.nn.Module): The evaluation loss criterion.
    - model (nn.Module): The PyTorch model to be trained and evaluated.
    - clip (float): Gradient clipping threshold.
    - dev_loader (iterable): Iterable containing development/validation samples.
    - train_loader (iterable): Iterable containing training samples.
    - test_loader (iterable): Iterable containing test samples.
    - lang: Language model containing word-to-id and id-to-word mappings.
    - device (str): Device to which the model should be moved.
    - ntasgd (bool): Whether to use NT-ASGD optimization (default is False).
    - patience (int): Patience for early stopping (default is 10).
    - config (dict): Dictionary containing ntasgd configurations.

    Returns:
    - nn.Module: The best model based on the evaluation results.
    """
    pOg= copy.deepcopy(patience)
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    for epoch in pbar:
        if not ntasgd:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        else:
            loss,optimizer = train_loopNT(train_loader, optimizer, criterion_train, model, config, dev_loader, criterion_eval)
            
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f, Best: %f" % (ppl_dev, best_ppl))
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = pOg
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                print('Patience Limit Reached!')
                break # Not nice but it keeps the code clean
            
    return best_model

def testModel(best_model,device, test_loader, criterion_eval, lang):
    """
    Tests the provided model on the test data and prints the results.

    Parameters:
    - best_model (nn.Module): The best model to be tested.
    - device (str): Device to which the model should be moved.
    - test_loader (iterable): Iterable containing test samples.
    - criterion_eval (torch.nn.Module): The evaluation loss criterion.
    - lang: Language model containing word-to-id and id-to-word mappings.

    Returns:
    - float: Test perplexity.
    """
    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    analogy_our_model('man', 'woman', 'u.s.', best_model, lang)
    
    print(analogy_spacy('man', 'woman', 'king'))
    return final_ppl

def printRes(pptot):
    """
    Prints the results based on the provided perplexities.

    Parameters:
    - pptot (list): List of perplexities for different models and configurations.
    """
    lbl=['WT+DO+AdW', 'WT+VDO+Adw', 'WT+VDO+SGD', 'WT+VDO+NTASGD']
    print('\n')
    for i,name in enumerate(lbl):
        print(name, np.array(pptot[i]).mean(), '+-', np.array(pptot[i]).std())