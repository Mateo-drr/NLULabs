# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from functools import partial
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import gc
    
    
    #spacy.cli.download("en_core_web_lg")
    
    device = 'cuda'
    path='D:/Universidades/Trento/2S/NLU/dataset/'
    
    train_raw = read_file(path+"ptb.train.txt")
    dev_raw = read_file(path+"ptb.valid.txt")
    test_raw = read_file(path+"ptb.test.txt")
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # Vocab is computed only on training set 
    # However you can compute it for dev and test just for statistics about OOV 
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    
    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    # Experiment also with a smaller or bigger model by changing hid and emb sizes 
    # A large model tends to overfit
    hid_size = 256
    emb_size = 256
     
    clip = 5 # Clip the gradient
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    n_epochs = 100
    patience = 10
    vocab_len = len(lang.word2id)
    
    pptot=[]
    ###########################################################################
    #Weight Tying + normal dropout

    lr = 0.001
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                    emb_dropout=0.1, out_dropout=0.1, vardrop=False).to(device)    
    model.apply(init_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
        
    res = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
             model, clip, dev_loader, train_loader, test_loader, lang, device)
    
    pptot.append(res)
    gc.collect()
    torch.cuda.empty_cache()
 
    ###########################################################################
    '''
    class LM_LSTM(nn.Module):
        def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                     emb_dropout=0.1, n_layers=1, vardrop=False):
            super(LM_LSTM, self).__init__()
            # Token ids to vectors
            self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
            # Pytorch's LSTM layer
            self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
            self.pad_token = pad_index
            # Linear layer to project the hidden layer to our output space
            self.output = nn.Linear(hidden_size, output_size)
            
            # Dropout
            self.drop1 = emb_dropout#nn.Dropout(emb_dropout)
            self.drop2 = out_dropout#nn.Dropout(out_dropout)
            
            # Weight tying
            self.output.weight = self.embedding.weight
            
            # Variational Dropout
            self.vardrop = vardrop
            self.dmask1=None
            self.dmask2=None
            self.x = nn.Dropout(emb_dropout)

        def forward(self, input_sequence):
            #print(input_sequence.size())
            emb = self.embedding(input_sequence)
            
            if self.vardrop: #and self.dmask1==None:
                self.dmask1 = torch.bernoulli(torch.full(emb.size(), 1 - self.drop1, device=device))
            
            if self.vardrop:
                emb = emb * self.dmask1
                
            emb = self.x(emb)
            
            lstm_out, _ = self.lstm(emb)  

            output = self.output(lstm_out).permute(0, 2, 1)
            return output

        def get_word_embedding(self, token):
            return self.embedding(token).squeeze(0).detach().cpu().numpy()

        def get_most_similar(self, vector, top_k=10):
            embs = self.embedding.weight.detach().cpu().numpy()
            # Our function that we used before
            scores = []
            for i, x in enumerate(embs):
                if i != self.pad_token:
                    scores.append(cosine_similarity(x, vector))
            # Take ids of the most similar tokens
            scores = np.asarray(scores)
            indexes = np.argsort(scores)[::-1][:top_k]
            top_scores = scores[indexes]
            return (indexes, top_scores)
    '''
    ###########################################################################
    
    #WT + variational dropout
    lr = 0.001
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                    emb_dropout=0.1, out_dropout=0.1, vardrop=True).to(device)    
    model.apply(init_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)#optim.SGD(model.parameters(), lr=lr)
        
    res = runModel(n_epochs, optimizer, criterion_train, criterion_eval,
             model, clip, dev_loader, train_loader, test_loader, lang, device)
    
    pptot.append(res)
    gc.collect()
    torch.cuda.empty_cache()

    ###########################################################################
    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    for epoch in pbar:
        
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
            
            #Get gradients
            gradient = [param.grad for param in model.parameters()]
                
        
            optimizer.step() # Update the weights
            
        loss = sum(loss_array)/sum(number_of_tokens)
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            
            #While stopping crit not met
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

        model.newvd=True #unlock mask for new epoch
                     
    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    analogy_our_model('man', 'woman', 'u.s.', model, lang)
    
    print(analogy_spacy('man', 'woman', 'king'))