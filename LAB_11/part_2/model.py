from torch import nn
import torch.nn.functional as F
# import torch

class BertCstm(nn.Module):

    def __init__(self,bert, hid_size, out_asp, dout=0.3):
        super(BertCstm, self).__init__()
        
        self.bert = bert
        self.asp_out = nn.Linear(hid_size, out_asp)
        self.asp_hid = nn.Linear(768,hid_size)
        self.asp_hid2 = nn.Linear(hid_size,hid_size)
        self.dropout = nn.Dropout(dout)
        self.d2 = nn.Dropout(dout)
        self.bn0 = nn.BatchNorm1d(768)
        self.bn = nn.BatchNorm1d(hid_size)
        self.bn2 = nn.BatchNorm1d(hid_size)
        self.m = nn.Mish(inplace=True)
        
    def forward(self, utterance,attm):

        bx = self.bert(utterance,attention_mask=attm)

        bx = self.bn0(bx.last_hidden_state.permute(0,2,1)).permute(0,2,1)
        bx = self.m(bx)

        bx = self.dropout(bx)                
            
        asps = self.asp_hid(bx).permute(0,2,1)
        asps = self.bn(asps).permute(0,2,1)
        asps = self.m(asps)
        asps = self.d2(asps)
        asps = self.asp_hid2(asps).permute(0,2,1)
        asps = self.bn2(asps).permute(0,2,1)
        asps = self.m(asps)
        asps = self.d2(asps)
        
        asps = self.asp_out(asps).permute(0,2,1)#bx.pooler_output)        
        
        return asps
