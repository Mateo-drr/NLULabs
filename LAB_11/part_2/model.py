from torch import nn
import torch.nn.functional as F
# import torch

class Bert2d(nn.Module):
    """
    Custom neural network model using BERT embeddings for aspect and polarity prediction.

    Args:
        bert: Pre-trained BERT/RoBERTa model.
        hid_size (int): Hidden size for linear layers.
        outPol (int): Number of output units for polarity predictions.
        outAsp (int): Number of output units for aspect predictions.
        dout (float): Dropout rate.
        maxb (int): Maximum tokens length.

    Attributes:
        bert: Pre-trained BERT/RoBERTa model.
        dout (nn.Dropout): Dropout layer.
        linear (nn.Sequential): Sequential linear layers for aspect prediction.
        linear2 (nn.Sequential): Additional sequential linear layers for aspect prediction.
        conv (nn.Sequential): Convolutional layer for aspect prediction.
        linearP (nn.Sequential): Sequential linear layers for polarity prediction.
        bn (nn.BatchNorm1d): Batch normalization layer.
    """

    def __init__(self,bert, hid_size, outPol, outAsp, dout=0.3, maxb=128):
        super(Bert2d, self).__init__()
        
        self.bert = bert
        #self.bert2 = bert
        
        self.dout = nn.Dropout(dout)
        
        self.linear = nn.Sequential(nn.Linear(768,hid_size),
                                    nn.Mish(inplace=True),
                                    nn.Linear(hid_size, 4)
                                    )
        self.linear2 = nn.Sequential(nn.Linear(maxb, outAsp*2),
                                     #nn.Mish(inplace=True),
                                     #nn.Linear(32, 20)
                                     )
        
        self.conv = nn.Sequential(#nn.Conv2d(1, 16, 3,1,1),
                                  #nn.Mish(inplace=True),
                                  #nn.BatchNorm2d(1),
                                  nn.Conv2d(1, 1, 3,2,1)
                                  )
        
        self.linearP = nn.Sequential(nn.Linear(768, outPol),
                                     #nn.Mish(inplace=True),
                                     #nn.Linear(128, out_slot)
                                     )
        
        self.bn = nn.BatchNorm1d(maxb)
        
        # self.optimus = nn.TransformerEncoder(nn.TransformerEncoderLayer(768,nhead=2,
        #                                                               activation='gelu',
        #                                                               batch_first=True),
        #                                    num_layers=6)
        
    def forward(self, utterance,attm):

        bx = self.bert(utterance,attention_mask=attm).last_hidden_state
        bx = self.bn(bx)
        bx = self.dout(bx) 

        # bx = self.optimus(bx)

        asp = self.linear(bx).permute(0,2,1)
        asp = self.linear2(asp).permute(0,2,1)
        
        #asp = self.dout(asp) 
        
        asp = self.conv(asp.unsqueeze(1)).squeeze(1)
        
        #pol = self.net(bx)
        #bx = self.bert2(utterance,attention_mask=attm)
        pol = self.linearP(bx[:,0,:])#.last_hidden_state[:,0,:])      
            
        return F.relu(asp),pol.clip(0,3)
##############################################################################

##############################################################################
#SERIES OF THE OTHER MODELS TESTED:
    
# class modCSTM(nn.Module):
#     def __init__(self, hidden_size, output_size, bert):
#         super(modCSTM, self).__init__()
#         self.bert = bert
#         self.fc1 = nn.Linear(768, hidden_size)
#         self.m = nn.Mish(inplace=True)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x,mask):
#         bx = self.bert(x,attention_mask=mask)
#         x = bx.last_hidden_state
        
#         out = self.fc1(x)
#         out = self.dropout(out)
#         out = self.m(out)
#         out = self.fc2(out).permute(0,2,1)
        
#         return out


# class aspExt(nn.Module):

#     def __init__(self,bert, hid_size, out_size, dout=0.3):
#         super(aspExt, self).__init__()
        
#         self.bert = bert
        
#         self.lin = nn.Linear(768, hid_size)
#         self.extra = nn.Sequential(
#                                    nn.Mish(inplace=True),
#                                    nn.Dropout(dout)
#                                    )
#         self.lin2 = nn.Linear(hid_size, out_size)
#         self.prime = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid_size,nhead=2,
#                                                                       activation='gelu',
#                                                                       batch_first=True),
#                                            num_layers=6)
#         self.optimus = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid_size,nhead=2,
#                                                                       activation='gelu',
#                                                                       batch_first=True),
#                                            num_layers=6)
#         self.bbb = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid_size,nhead=2,
#                                                                       activation='gelu',
#                                                                       batch_first=True),
#                                            num_layers=6)
        
#         self.bn = nn.BatchNorm1d(hid_size)
#         self.bn2 = nn.BatchNorm1d(hid_size)
#         self.bn3 = nn.BatchNorm1d(hid_size)
        
#         self.lin3 = nn.Linear(out_size, hid_size)
        
#         base = 128
#         self.rrdb = RRDBLinear(hid_size,base)
#         self.rrdb2 = RRDBLinear(hid_size,base*2)
#         self.rrdb3 = RRDBLinear(hid_size,base*4)
#         self.rrdb4 = RRDBLinear(hid_size,base*4)
        
#         self.act = nn.Mish(inplace=True)

        
#     def forward(self, utterance,attm):

#         bx = self.bert(utterance,attention_mask=attm).last_hidden_state

#         # aspects = self.lin(bx[:,0,:])
#         # aspectsj = self.bn(self.extra(aspects))
        
#         # aspects = self.prime(aspectsj.unsqueeze(1)).squeeze(1)s
#         # aspects = self.bn2(self.extra(aspects))
        
#         # aspectsj2 = self.lin3(aspects + aspectsj)
#         # aspects = self.bn3(self.extra(aspects))
        
#         # aspects = self.lin2(aspects + aspectsj + aspectsj2) 
        
#         aspects = self.lin(bx[:,0,:])
#         aspects = self.bn(self.extra(aspects))
        
#         aspects = self.optimus(aspects.unsqueeze(1)).squeeze(1)
        
#         aspects = self.rrdb(aspects)
#         aspects = self.rrdb2(aspects)
        
#         #aspects = aspects + self.lin3(utterance.to(torch.float32))
        
#         aspects = self.rrdb3(aspects)
#         aspects = self.rrdb4(aspects)
        
#         #aspects = self.bn(self.extra(aspects))
#         aspects = self.prime(aspects.unsqueeze(1)).squeeze(1)
        
#         aspects = self.lin2(aspects) 

        
#         return aspects
    
    
# class LstmMod(nn.Module):

#     def __init__(self, hid_size, out_slot, out_int, emb_size, n_layer=1, pad_index=0, dout=0):
#         super(LstmMod, self).__init__()
        
#         self.embedding = nn.Linear(128, emb_size)
#         self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    
#         self.slot_out = nn.Linear(2 * hid_size, out_slot)
#         self.intent_out = nn.Linear(hid_size, out_int)
#         self.dropout = nn.Dropout(dout)
        
#     def forward(self, utterance,mask):
#         utt_emb = self.embedding(utterance.to(torch.float32)) #[b,128]
#         utt_emb = self.dropout(utt_emb)
        
#         utt_encoded, (last_hidden, cell) = self.utt_encoder(utt_emb.unsqueeze(1)) #[b,1,512]
#         last_hidden = last_hidden[-1, :, :] #[b,256]
        
#         slots = self.slot_out(utt_encoded)
#         #intent = self.intent_out(last_hidden)
#         return slots



# class ResidualDenseBlock_5CLinear(nn.Module):
#     def __init__(self, nf=64, gc=32, bias=True):
#         super(ResidualDenseBlock_5CLinear, self).__init__()
#         self.fc1 = nn.Linear(nf, gc, bias=bias)
#         self.fc2 = nn.Linear(nf + gc, gc, bias=bias)
#         self.fc3 = nn.Linear(nf + 2 * gc, gc, bias=bias)
#         self.fc4 = nn.Linear(nf + 3 * gc, gc, bias=bias)
#         self.fc5 = nn.Linear(nf + 4 * gc, nf, bias=bias)
#         self.lrelu = nn.Mish(inplace=True)

#     def forward(self, x):
#         x1 = self.lrelu(self.fc1(x))
#         x2 = self.lrelu(self.fc2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.fc3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.fc4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.fc5(torch.cat((x, x1, x2, x3, x4), 1))
#         return x5 * 0.2 + x

# class RRDBLinear(nn.Module):
#     '''Residual in Residual Dense Block with Linear Layers'''
    
#     def __init__(self, nf, gc=32):
#         super(RRDBLinear, self).__init__()
#         self.RDB1 = ResidualDenseBlock_5CLinear(nf, gc)
#         self.RDB2 = ResidualDenseBlock_5CLinear(nf, gc)
#         self.RDB3 = ResidualDenseBlock_5CLinear(nf, gc)
    
#     def forward(self, x):
#         out = self.RDB1(x)
#         out = self.RDB2(out)
#         out = self.RDB3(out)
#         return out * 0.2 + x
    
    

# class BertCstm(nn.Module):

#     def __init__(self,bert, hid_size, out_slot, out_int, dout=0.3):
#         super(BertCstm, self).__init__()
        
#         self.bert = bert
        
#         self.slot_out = nn.Linear(hid_size, out_slot)
#         self.slt_hid = nn.Linear(768,hid_size)
#         self.slt_hid2 = nn.Linear(hid_size,hid_size)
        
#         self.intent_out = nn.Linear(hid_size, out_int)
#         self.int_hid = nn.Linear(768,hid_size)
#         self.int_hid2 = nn.Linear(hid_size,hid_size)
        
#         self.dout = nn.Dropout(dout)
#         self.bn = nn.BatchNorm1d(hid_size)
#         self.bn2 = nn.BatchNorm1d(hid_size)
#         self.m = nn.Mish(inplace=True)
        
#         self.optimus = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid_size,nhead=2,
#                                                                       activation='gelu',
#                                                                       batch_first=True),
#                                            num_layers=6)
        
#     def forward(self, utterance,attm):

#         bx = self.bert(utterance,attention_mask=attm)

#         bx = self.dout(bx.last_hidden_state)                
            
#         slots = self.slt_hid(bx).permute(0,2,1)
#         slots = self.bn(slots).permute(0,2,1)
#         slots = self.m(slots)
#         slots = self.dout(slots)
#         slots = self.optimus(slots).permute(0,2,1)
#         slots = self.bn2(slots).permute(0,2,1)
#         slots = self.m(slots)
#         slots = self.dout(slots)
        
#         slots = self.slot_out(slots).permute(0,2,1)

#         intent = self.int_hid(bx).permute(0,2,1)
#         intent = self.bn(intent).permute(0,2,1)
#         intent = self.m(intent)
#         intent = self.dout(intent)
        
#         intent = self.intent_out(intent).permute(0,2,1)

#         return slots, intent
    
    
# class Bert10(nn.Module):

#     def __init__(self,bert, hid_size, out_slot, out_int, dout=0.3):
#         super(Bert10, self).__init__()
        
#         self.bert = bert
        
#         self.dout = nn.Dropout(dout)
        
#         self.linear = nn.Linear(hid_size, out_slot)
#         self.linear2 = nn.Linear(hid_size, out_slot)
        
#         self.net = nn.Sequential(nn.Linear(768,hid_size),
#                                  #nn.BatchNorm1d(hid_size),
#                                  nn.Mish(inplace=True),
#                                  #nn.Dropout(dout)
#                                  # RRDBLinear(hid_size,hid_size*2),
#                                  #nn.Linear(hid_size,hid_size),
#                                  # nn.Mish(inplace=True),
#                                  )
#         self.net2 = nn.Sequential(nn.Linear(768,hid_size),#RRDBLinear(768,hid_size),
#                                  #nn.BatchNorm1d(hid_size),
#                                  nn.Mish(inplace=True),
#                                  #nn.Dropout(dout)
#                                  )
        
#     def forward(self, utterance,attm):

#         bx = self.bert(utterance,attention_mask=attm)
#         bx = self.dout(bx.last_hidden_state[:,0,:]) 

#         asp = self.net(bx)
#         asp = self.linear(asp)
#         #asp = self.optimus(asp)
        
#         # pol = self.net(bx)
#         # pol = self.linear2(pol)      
#         #pol = self.bbb(pol)       
            
#         return F.relu(asp)#,pol.clip(0,4)

    