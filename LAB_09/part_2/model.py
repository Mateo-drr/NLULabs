import numpy as np
from numpy.linalg import norm
import torch.nn as nn
import torch
from torch.nn.utils.rnn import PackedSequence    

def cosine_similarity(v, w):
    return np.dot(v,w)/(norm(v)*norm(w))  

###########################################################################
#LSTM MODEL

class LM_LSTM(nn.Module):
    """
    LSTM Language Model class.

    Parameters:
    - emb_size (int): Size of the word embeddings.
    - hidden_size (int): Size of the hidden layer in LSTM.
    - output_size (int): Size of the output vocabulary.
    - pad_index (int): Index of the padding token (default is 0).
    - out_dropout (float): Dropout probability for the output layer (default is 0.1).
    - emb_dropout (float): Dropout probability for the embedding layer (default is 0.1).
    - n_layers (int): Number of layers in the LSTM (default is 1).
    - vardrop (bool): Whether to use variational dropout (default is False).
    - dropprob (float): Dropout probability for variational dropout (default is 0).

    Attributes:
    - embedding (nn.Embedding): Token ids to vectors.
    - lstm (nn.LSTM): PyTorch's LSTM layer.
    - output (nn.Linear): Linear layer to project the hidden layer to the output space.
    - drop1 (nn.Dropout): Dropout layer for the embedding.
    - drop2 (nn.Dropout): Dropout layer for the output.
    - pad_token (int): Index of the padding token.
    - vardrop (bool): Flag indicating the use of variational dropout.
    - dropprob (float): Dropout probability for variational dropout.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, vardrop=False,dropprob=0):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=pad_index) 
        self.emblin = nn.Linear(hidden_size, emb_size) #used to allow weight tying with hidden_size != emb_size
        # Pytorch's LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        
        # Dropout
        self.drop1 = nn.Dropout(emb_dropout)
        self.drop2 = nn.Dropout(out_dropout)
        
        # Weight tying
        self.output.weight = self.embedding.weight
        
        # Variational Dropout control
        self.vardrop = vardrop
        self.dropprob = dropprob
        
    def checkPacked(x):
        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)
        return x,max_batch_size
        
    def forward(self, input_sequence):
        
        emb = self.embedding(input_sequence)
        
        if self.vardrop and self.training:    #dont apply dropout during testing
            x=emb
            emb = self.emblin(emb)
            #sample a dropout mask
            dropout_mask = x.new_empty(x.shape[0], 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropprob)
            #since the expected input was x (without dropout) now we need to scale it
            #Ex: dropprob is 0.4 then we'd be giving the output of value x*0.6 so we need to divide by 0.6 so it always gives x
            #https://stackoverflow.com/questions/57193633/how-inverting-the-dropout-compensates-the-effect-of-dropout-and-keeps-expected-v
            x = x * dropout_mask / (1 - self.dropprob)
            lstm_out,_=self.lstm(emb)
            #apply same dropout after
            lstm_out = lstm_out* dropout_mask / (1 - self.dropprob)
        else:
            emb = self.emblin(emb)
            emb = self.drop1(emb)
            lstm_out,_=self.lstm(emb)  
            lstm_out = self.drop2(lstm_out)
        
        x = self.output(lstm_out).permute(0, 2, 1)
        return x

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
