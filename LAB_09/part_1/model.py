import numpy as np
from numpy.linalg import norm
import torch.nn as nn

 ###########################################################################
 #RNN MODEL
 
def cosine_similarity(v, w):
    return np.dot(v,w)/(norm(v)*norm(w))    
 
class LM_RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) language model.

    Parameters:
    - emb_size (int): Dimensionality of word embeddings.
    - hidden_size (int): Number of features in the hidden state of the RNN.
    - output_size (int): Size of the vocabulary.
    - pad_index (int): Index of the padding token in the vocabulary.
    - out_dropout (float): Dropout applied to the output layer.
    - emb_dropout (float): Dropout applied to the word embeddings.
    - n_layers (int): Number of recurrent layers.

    Methods:
    - forward(input_sequence): Forward pass of the model.
    - get_word_embedding(token): Get the word embedding for a given token.
    - get_most_similar(vector, top_k=10): Get the indices and scores of the most similar tokens to a given vector.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()
    
    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens 
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]  
        top_scores = scores[indexes]
        return (indexes, top_scores)    
    
###########################################################################
#LSTM MODEL
class LM_LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) language model.

    Parameters:
    - emb_size (int): Dimensionality of word embeddings.
    - hidden_size (int): Number of features in the hidden state of the LSTM.
    - output_size (int): Size of the vocabulary.
    - pad_index (int): Index of the padding token in the vocabulary.
    - out_dropout (float): Dropout applied to the output layer.
    - emb_dropout (float): Dropout applied to the word embeddings.
    - n_layers (int): Number of recurrent layers.

    Methods:
    - forward(input_sequence): Forward pass of the model.
    - get_word_embedding(token): Get the word embedding for a given token.
    - get_most_similar(vector, top_k=10): Get the indices and scores of the most similar tokens to a given vector.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        
        #Dropout
        self.drop1 = nn.Dropout(emb_dropout)
        self.drop2 = nn.Dropout(out_dropout)
        

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.drop1(emb)
        lstm_out, _ = self.lstm(emb)  # Use LSTM here
        lstm_out = self.drop2(lstm_out)
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