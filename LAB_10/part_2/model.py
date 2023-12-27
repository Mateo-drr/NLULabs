from transformers import BertTokenizer, BertModel
import torch.nn as nn

device='cuda'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)

class BertCstm(nn.Module):

    def __init__(self,bert, hid_size, out_slot, out_int, dout=0):
        super(BertCstm, self).__init__()
        
        self.bert = bert
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(dout)
        
    def forward(self, utterance):

        bx = self.bert(utterance)
        
        slots = self.slot_out(bx.last_hidden_state).permute(0,2,1)#bx.pooler_output)
        # Compute intent logits
        intent = self.intent_out(bx.last_hidden_state[:,0,:])
        
        
        return slots, intent