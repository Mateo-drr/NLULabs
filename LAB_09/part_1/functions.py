# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import math

from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import spacy
nlp = spacy.load('en_core_web_lg')

def train_loop(data, optimizer, criterion, model, clip=5):
    """
    Train the given model on the provided data using the specified optimizer and criterion.

    Parameters:
    - data (iterable): dataloader
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - criterion (torch.nn.Module): Loss criterion for computing the training loss.
    - model (torch.nn.Module): The model to be trained.
    - clip (float): Gradient clipping threshold.

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


def eval_loop(data, eval_criterion, model):
    """
    Evaluate the given model on the provided data using the specified evaluation criterion.

    Parameters:
    - data (iterable): Iterable containing evaluation samples.
    - eval_criterion (torch.nn.Module): Evaluation loss criterion.
    - model (torch.nn.Module): The model to be evaluated.

    Returns:
    - tuple: Perplexity and average evaluation loss per token.
    """
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
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
    Initialize the weights of the given PyTorch module using specific strategies.

    Parameters:
    - mat (torch.nn.Module): The module whose weights need to be initialized.
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
    Perform word analogy using pre-trained spaCy vectors.

    Parameters:
    - w1 (str): First word.
    - w2 (str): Second word.
    - w3 (str): Third word.

    Prints:
    - Word similarities based on the word analogy.
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
    Perform word analogy using the provided model and language.
 
    Parameters:
    - w1 (str): First word.
    - w2 (str): Second word.
    - w3 (str): Third word.
    - model (torch.nn.Module): The model for word embeddings.
    - lang: Language object containing word-to-index and index-to-word mappings.
 
    Prints:
    - Word similarities based on the word analogy.
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
             dev_loader,train_loader,test_loader,lang, device, patience):
    """
    Train the model, evaluate on the development set, and return the best model based on perplexity.

    Parameters:
    - n_epochs (int): Number of training epochs.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - criterion_train (torch.nn.Module): Training loss criterion.
    - criterion_eval (torch.nn.Module): Evaluation loss criterion.
    - model (torch.nn.Module): The model to be trained.
    - clip (float): Gradient clipping threshold.
    - dev_loader (iterable): Development set data loader.
    - train_loader (iterable): Training set data loader.
    - test_loader (iterable): Test set data loader.
    - lang: Language object containing word-to-index and index-to-word mappings.
    - device (str): Device to which the model should be moved.

    Returns:
    - torch.nn.Module: Best model based on perplexity.
    """
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pat = copy.deepcopy(patience)
    pbar = tqdm(range(1,n_epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                pat = patience
            else:
                pat -= 1
                
            if pat <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
                              
    return best_model

def testModel(best_model,device, test_loader, criterion_eval, lang):
    """
    Evaluate the best model on the test set and perform word analogy.

    Parameters:
    - best_model (torch.nn.Module): Best model based on perplexity.
    - device (str): Device to which the model should be moved.
    - test_loader (iterable): Test set data loader.
    - criterion_eval (torch.nn.Module): Evaluation loss criterion.
    - lang: Language object containing word-to-index and index-to-word mappings.

    Prints:
    - Test perplexity and word analogy results.
    """
    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    analogy_our_model('man', 'woman', 'u.s.', best_model, lang)
    
    print(analogy_spacy('man', 'woman', 'king'))
    return final_ppl

def printRes(pptot):
    """
    Print the results (perplexity) for different model configurations.

    Parameters:
    - pptot (list of lists): List of perplexity values for different model configurations.
    """
    lbl=['RNN', 'LSTM', 'LSTM+DO', 'LSTM+DO+AW']
    print('\n')
    for i,name in enumerate(lbl):
        print(name, np.array(pptot[i]).mean(), '+-', np.array(pptot[i]).std())