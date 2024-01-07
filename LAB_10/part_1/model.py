
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False)    
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance) 
        utt_emb = utt_emb.permute(1,0,2) 
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: seq_len, batch size, calsses 
        slots = slots.permute(1,2,0) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    
class ModelIASbd(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dout=0):
        super(ModelIASbd, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True)    
        self.slot_out = nn.Linear(2*hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(dout)
        
    def forward(self, utterance, seq_lengths):

        utt_emb = self.embedding(utterance)
        #dout
        utt_emb = self.dropout(utt_emb)
        utt_emb = utt_emb.permute(1,0,2)
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())
        
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        # Compute slot logits
        #print(utt_encoded.size(), last_hidden.size())
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: seq_len, batch size, calsses 
        slots = slots.permute(1,2,0) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent