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
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, vardrop=False):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        self.lstmvd = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, dropout=out_dropout)
        
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        
        # Dropout
        self.drop1 = nn.Dropout(emb_dropout)
        self.drop2 = nn.Dropout(out_dropout)
        
        # Weight tying
        self.output.weight = self.embedding.weight
        
        # Variational Dropout
        self.vardrop = vardrop
        
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
        emb = self.drop1(emb)
        
        if self.vardrop:    
            lstm_out, _ = self.lstmvd(emb)
            lstm_out = self.drop2(lstm_out) #apply a single mask to whole output, not between lstm layers
        else:
            lstm_out,_=self.lstm(emb) #lstm with internal dropout 
        
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
