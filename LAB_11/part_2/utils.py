# Add functions or classes used for data loading and preprocessing

from torch.utils.data import Dataset
import copy
import torch
import torch.nn.functional as F
import augment as aug
import random
import re


#To simplify the predictions
redux = ['POS','NEU','NEG','O']
device="cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN=0
#mapping
pol2idx = {'POS': 4, 'NEU': 3, 'NEG': 2, 'O': 1}
idx2pol = {4: 'POS', 3: 'NEU', 2: 'NEG', 1: 'O'}
prob = 0.8


def preproc(temp,classes):
    """
    Preprocesses the data removing excess samples with no aspects.

    Args:
        temp (list): List of sentences with aspect information.
        classes (list): List of aspect classes.

    Returns:
        tuple: Tuple containing the preprocessed sentences and the list of aspect classes.
    """

    #data is unbalanced, it contains tot 2724 samples but 1434 with no aspects = 52%
    #so considering there are 4 classes for polarity i removed samples to reach 25%
    #of data with no aspects
    i=0
    tsents=[]
    for s in temp:
        if s['asp'] == []:
            i+=1
        if i > int(1434*1): #33% #685: #25%
            pass
        else:
            tsents.append(s)
    aspects = list(set(classes))
    #aspects.append('unk')
    #aspects.append('O')
    return tsents,aspects

def loadData(file_path,classes):  
    """
    Load data from a file and process it into a list of sentences with aspect and polarity information.

    Args:
        file_path (str): Path to the file containing the data.
        classes (list): List of aspects.

    Returns:
        tuple: Tuple containing the list of sentences with aspect information and the list of aspect classes.
    """

    # Read the data from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    # Split the data into lines
    lines = data.split('\n')

    sents = []
    temp = []
    tlbl = []
    asp=[]
    jasp=[]
    jlbl=[]
    lastA = False
    pos=1 #position stored as postition+1 to avoid padding overlap
    loc=[]
    for line in lines:
        if line == '':
            
            if loc != [] and len(loc[-1]) == 1:
                loc[-1] = [loc[-1][0],loc[-1][0]]
            
            sents.append({'s':' '.join(temp),
                          'lbl':tlbl,
                          'asp': asp,
                          #'slist': temp,
                          'jasp': jasp,
                          'jlbl': jlbl,
                          'posA': loc})
            temp = []
            tlbl = []
            asp=[]
            jasp=[]
            jlbl=[]
            loc=[]
            pos=1
            lastA=False
        else:
            x = line.split(" ")
            temp.append(x[0].replace('PUNCT', '.'))
            
            #term is an aspect
            if x[1] != 'O':
                idx = pol2idx[x[1].split('-')[1]]
                tlbl.append(idx) #append only polarity not B I E or S
                
                word = x[0].replace('PUNCT', '.') #aspect term
                asp.append(word)
                
                if lastA: #check if previous term was also an aspect
                    jasp[-1] = ' '.join([jasp[-1], word]) #store it as single aspect term
                    classes[-1] = ' '.join([classes[-1], word])
                    loc[-1].append(pos)
                    
                else:
                    jasp.append(word) #save word
                    classes.append(word)
                    jlbl.append(idx)
                    #store aspect position
                    loc.append([pos])
                    
                lastA=True 

            #term is not an aspect
            else:
                idx = pol2idx[x[1]]
                tlbl.append(idx)
                
                #case were the aspect is a single term
                if lastA==True and len(loc[-1]) == 1:
                    loc[-1]= [loc[-1][0],loc[-1][0]]
                #case the aspect is longer than 2 words
                elif lastA==True and len(loc[-1]) > 2:
                    loc[-1] = [loc[-1][0],loc[-1][-1]]
                
                lastA=False
            
            pos+=1
                
    return sents[:-1],classes #last sent is always empty

class CustomDataset(Dataset):
    """
    Custom dataset class for PyTorch DataLoader.

    Methods:
    - __init__: Initialize the dataset.
    - __len__: Get the length of the dataset.
    - __getitem__: Get an item from the dataset.
    """

    def __init__(self, data, tokenizer,w2v=None,train=False):
        #WHAT DO WE PUT HERE?
        #copy the data
        self.data = copy.deepcopy(data)
        #load the given tokenizer
        self.tokenizer = tokenizer
        self.w2v = w2v
        self.train = train

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)


    def __getitem__(self, idx):
        #TAKE ONE ITEM FROM THE DATASET
        uttOG = self.data[idx]['s']
        lbl = self.data[idx]['lbl']
        lbl = torch.Tensor(lbl)
        
        #AUGMET THE DATA BY CHANGING WORDS TO SYNONYMS
        #print(uttOG)
        # if random.random() < prob and self.train:
        #     nons = aug.rmvStopwords(uttOG)
        #     syn = aug.findSynonyms(nons,self.w2v)
        #     utt = aug.replaceWords(syn, uttOG)
        # else:
        utt=uttOG
        
        #FIX TOKENIZATION SPLIT ERRORS
        #######################################################################
        #letter.letter -> letterletter
        utt = re.sub(r"([a-zA-Z])\.([a-zA-Z])", r"\1\2", utt)
        #fix decimals
        utt = re.sub(r'\d+\.\d+', lambda x: x.group(0).split('.')[0], utt)
            #utt = utt.replace("10.8", "10")
            #utt = utt.replace("10.9", "10")
        #fix symbols
        utt = re.sub(r'--', '^', utt)
        utt = re.sub(r'[-_\"*+/]', '', utt)
        utt = utt.replace('(','')
        utt = utt.replace(')','')
        #fix fractions
        utt = re.sub(r'(\d+)/(\d+)', 'frac', utt)
            # utt = utt.replace("--", "^")
            # utt = utt.replace("-","")
            # utt = utt.replace("_","")
            # utt = utt.replace('"','')
            # utt = utt.replace('*','')
            # utt = utt.replace('+','')
            # utt = utt.replace('1/2','half')
            # utt = utt.replace('/','')
        #fix numbers
        utt = re.sub(r'(\d+,\d+)', lambda x: x.group(0).replace(',', ''), utt)
        #utt = utt.replace('1,700','1700')
        #utt = utt.replace('1,500','1500')
        #utt = utt.replace('1,200','1200')
        #utt = utt.replace('2,400','2400')
        #utt = utt.replace('2,300','2300')
        #a symbol on its own got removed
        utt = utt.replace('  ', " - ") 
        

        txt = utt.split(' ')
        
        for i in range(0,len(txt)):
            if txt[i].endswith('.') and len(txt[i])>1:
                txt[i] = txt[i][:-1]
            if "'" in txt[i]:
                txt[i] = txt[i].replace("'","")
        #######################################################################
        
        #AUGMENTATIONS
        if random.random() < prob and self.train and self.data[idx]['asp'] != []:
            txt,lbl = aug.switch(txt,lbl)
        
        utt = ' '.join(txt)
        
        joint = self.tokenizer.encode(utt)
        tkns = self.tokenizer.convert_ids_to_tokens(joint)
        
        #TOKENIZATION FIX
        split = []
        for i,t in enumerate(tkns):
            if not t.startswith('##') and not t.startswith("'"):     
                split.append(t)
                
        split = self.tokenizer.convert_tokens_to_ids(split)

        split = torch.tensor(split)
        lbl = torch.cat([torch.tensor([PAD_TOKEN]),lbl,torch.tensor([PAD_TOKEN])]) #pad slot
        
        #just in case code fails
        if len(split) != len(lbl):
            print('ERROR', split, lbl, utt, uttOG)#, syn)
            utt=uttOG #w2v caused a missalignment so use original sentence
        #else:
         #   break

        return {
            'utt': uttOG,
            'utterance': split, #encoding_text['input_ids'].flatten(),
            'text_attention_mask': torch.ones_like(split),#encoding_text['attention_mask'].flatten(),
            'lbl': lbl,
        }
    
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
    y_lbl, y_lengths = merge(new_item["lbl"])
    text_attention_mask, _ = merge(new_item["text_attention_mask"])
    
    src_utt = src_utt.to(device)  # We load the Tensor on our selected device
    y_lbl = y_lbl.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    text_attention_mask = text_attention_mask.to(device)
    
    new_item["utterances"] = src_utt
    new_item["y_lbl"] = y_lbl
    new_item["asp_len"] = y_lengths
    new_item["text_attention_mask"] = text_attention_mask
    
    return new_item

