
# Add functions or classes used for data loading and preprocessing
import torch
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset
import copy
import torch.nn.functional as F

PAD_TOKEN = 0
device='cuda'

    
    
def load_data(path):
    """
    Load data from a JSON file and return it as a list of dictionaries.

    Args:
    - path: Path to the JSON file.

    Returns:
    List of dictionaries representing the loaded data.
    """
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def preproc(path):
    """
    Preprocess the data by splitting it into training, development, and test sets.

    Args:
    - path: Path to the data.

    Returns:
    Tuple containing training, development, and test sets.
    """
    tmp_train_raw = load_data(path+'train.json')
    test_raw = load_data(path+'test.json')
    # Firt we get the 10% of dataset, then we compute the percentage of these examples 
    # on the training set which is around 11% 
    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10)/(len(tmp_train_raw)),2)
    
    
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)
    
    Y = []
    X = []
    mini_Train = []
    
    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occure once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=Y)
    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev
    
    return train_raw, dev_raw, test_raw

class Lang():
    """
    Language class to handle vocabulary and label mappings.

    Methods:
    - __init__: Initialize the language class.
    - w2id: Map words to indices.
    - lab2id: Map labels to indices.
    """
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

def getLang(train_raw, dev_raw, test_raw):
    """
    Get language-related information from the dataset.

    Args:
    - train_raw: Training dataset.
    - dev_raw: Development dataset.
    - test_raw: Test dataset.

    Returns:
    Tuple containing words, intents, and slots.
    """
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    return words, intents, slots

class CustomDataset(Dataset):
    """
    Custom dataset class for PyTorch DataLoader.

    Methods:
    - __init__: Initialize the dataset.
    - __len__: Get the length of the dataset.
    - __getitem__: Get an item from the dataset.
    - mapping_lab: Map labels to indices.
    - mapping_seq: Map sequences to indices.
    """

    def __init__(self, data, lang, tokenizer, max_len_bert):
        #WHAT DO WE PUT HERE?
        #copy the data
        self.data = copy.deepcopy(data)
        #load the given tokenizer
        self.tokenizer = tokenizer
        #load the maximum length of the input
        self.max_len_bert = max_len_bert
        
        self.intents = []
        self.slots = []
        #get separate lists
        for x in self.data:
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])
        
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)


    def __getitem__(self, idx):
        #TAKE ONE ITEM FROM THE DATASET
        utt = self.data[idx]['utterance']
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        
        encoding_text = self.tokenizer.encode_plus(
            utt,
            max_length=self.max_len_bert,
            add_special_tokens=True,
            #padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
            truncation=True
        )
        
        pad=encoding_text['input_ids'].flatten().size()[0]- slots.size()[0]
        if pad != 0:    
            slots = F.pad(slots, (0, pad), value=0)
            
        
        #decode =  self.tokenizer.convert_ids_to_tokens(encoding_text['input_ids'].flatten())
        #print(decode)
        return {
            'utt': utt,
            #'decoded': decode,
            'utterance': encoding_text['input_ids'].flatten(),
            'text_attention_mask': encoding_text['attention_mask'].flatten(),
            'slots': slots,
            'intent': intent,
            'pad':pad
        }

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
def collate_fn(data):
    """
    Collate function for the DataLoader.

    Args:
    - data: List of items from the dataset.

    Returns:
    Dictionary containing the collated data.
    """
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e., 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    text_attention_mask, _ = merge(new_item["text_attention_mask"])
    
    src_utt = src_utt.to(device)  # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    text_attention_mask = text_attention_mask.to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["text_attention_mask"] = text_attention_mask
    
    return new_item