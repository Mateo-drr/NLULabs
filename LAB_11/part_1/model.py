import torch.nn as nn
import torch.nn.functional as F

class SubjModel(nn.Module):
    """
    PyTorch model for subjectivity classification using BERT.

    Parameters:
    - input_size (int): Size of the input features.
    - hidden_size (int): Size of the hidden layer.
    - output_size (int): Size of the output layer.
    - bert (transformers.BertModel): BERT model 
    """
    def __init__(self, input_size, hidden_size, output_size, bert):
        super(SubjModel, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        bx = self.bert(x)
        x = bx.last_hidden_state[:,0,:]
        
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out