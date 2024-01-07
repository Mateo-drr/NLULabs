# Add functions or classes used for data loading and preprocessing

from torch.utils.data import Dataset
import copy
import torch
import torch.nn.functional as F

#To simplify the predictions
redux = ['POS','NEU','NEG','O']
#mapping
pol2idx = {'POS': 3, 'NEU': 2, 'NEG': 1, 'O': 0}#{'POS': 4, 'NEU': 3, 'NEG': 2, 'O': 1}
idx2pol = {3: 'POS', 2: 'NEU', 1: 'NEG', 0: 'O'}#{4: 'POS', 3: 'NEU', 2: 'NEG', 1: 'O'}

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
        if i > 899: #33% #685: #25%
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
                          #'lbl':tlbl,
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
    Custom dataset giving target [10,2] and [10]

    Args:
        data (list): List of sentences with aspect information.
        tokenizer: Tokenizer for encoding sentences.
        max_len_bert (int): Maximum length for input sequences.
        aspects (list): List of aspect classes.
        pad (int): Padding size for aspect positions.

    Attributes:
        data (list): List of sentences with aspect information.
        tokenizer: Tokenizer for encoding sentences.
        max_len_bert (int): Maximum length for input sequences.
        enc (dict): Dictionary mapping aspect classes to indices.
        pad (int): Padding size for aspect positions.
    """
    def __init__(self, data, tokenizer, max_len_bert, aspects, pad=10):
        self.data = copy.deepcopy(data)
        self.tokenizer = tokenizer
        self.max_len_bert = max_len_bert
        self.enc = {class_name: number + 1 for number, class_name in enumerate(aspects)} #+1 for padding
        self.pad = pad

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['s']
        lbl = torch.Tensor(self.data[idx]['jlbl'])
        # asp = self.data[idx]['jasp']
        asp = torch.tensor(self.data[idx]['posA'])
        
        #tokenize sentence
        encoding_text = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len_bert,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
            truncation=True
        )
        
        if len(asp) == 0:
            posA = torch.zeros([self.pad,2])
        elif asp.shape[0] != self.pad:
            posA = F.pad(asp, (0,0,0,self.pad-asp.shape[0])) 
            
        if posA.shape[0] != 10 and posA.shape[1] != 2:
            print('a')
        
        #encode aspects/ Used for one of the tested models
        # encAsp=[]
        # for i in range(len(asp)):
        #     encAsp.append(self.enc[asp[i]]) 
        # #pad to standard size
        # if len(encAsp) < self.pad:
        #     encAsp = F.pad(torch.tensor(encAsp), (0, self.pad-len(encAsp)), value=0)
        
        #pad polarity
        if len(lbl) < self.pad:
            lbl = F.pad(lbl, (0, self.pad-len(lbl)), value=0)
        
        return {
            's':self.data[idx]['s'],
            'text_input_ids': encoding_text['input_ids'].flatten(),
            'text_attention_mask': encoding_text['attention_mask'].flatten(),
            'pol':lbl,
            # 'asp':encAsp
            'asp':posA
        }


#DATASET USED FOR OTHER TESTED MODELS
# class CustomDataset(Dataset):

#     def __init__(self, data, tokenizer, max_len_bert, aspects):
#         self.data = copy.deepcopy(data)
#         self.tokenizer = tokenizer
#         self.max_len_bert = max_len_bert
#         self.enc = {class_name: number + 1 for number, class_name in enumerate(aspects)} #+1 for padding

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text = self.data[idx]['s']
#         lbl = torch.Tensor(self.data[idx]['lbl'])
#         asp = self.data[idx]['asp']
        
#         #tokenize sentence
#         encoding_text = self.tokenizer.encode_plus(
#             text,
#             max_length=self.max_len_bert,
#             add_special_tokens=True,
#             padding='max_length',
#             return_attention_mask=True,
#             return_token_type_ids=False,
#             return_tensors='pt',
#             truncation=True
#         )
        
#         #tokenize aspects
#         if asp == []:
#             enc_asp={}
#             enc_asp['input_ids'] = torch.zeros([128])
#             enc_asp['attention_mask'] = torch.zeros([128])
#         else:
#             enc_asp = self.tokenizer.encode_plus(
#                 asp,
#                 max_length=self.max_len_bert,
#                 add_special_tokens=True,
#                 padding='max_length',
#                 return_attention_mask=True,
#                 return_token_type_ids=False,
#                 return_tensors='pt',
#                 truncation=True
#             )
        
#         #pad polarity lbls
#         pad=encoding_text['input_ids'].flatten().size()[0]- lbl.size()[0]
#         if pad != 0:    
#             lbl = F.pad(lbl, (0, pad), value=0)
            
#         #aspect labels
#         encAsp = []
#         for i,pol in enumerate(self.data[idx]['lbl']):
#             if pol != 1:
#                 #print(self.data[idx])
#                 encAsp.append(self.enc[self.data[idx]['slist'][i]]) 
#             else:
#                 encAsp.append(self.enc['O']) 
#         if pad != 0:    
#             encAsp = F.pad(torch.tensor(encAsp), (0, pad), value=0)
        
#         return {
#             'text_input_ids': encoding_text['input_ids'].flatten(),
#             'text_attention_mask': encoding_text['attention_mask'].flatten(),
#             'pol': lbl.to(torch.long),
#             'asp_ids': enc_asp['input_ids'].flatten(),
#             'asp_att': enc_asp['attention_mask'].flatten(),
#             'asp_pos': encoding_text['input_ids'].flatten(),
#             'asp_enc': encAsp.to(torch.long) #assign class id in sentence position
#         }
