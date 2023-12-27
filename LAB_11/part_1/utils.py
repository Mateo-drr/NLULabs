from torch.utils.data import Dataset, DataLoader
import copy
import torch
#import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#from torch.utils.data import Dataset, DataLoader
from nltk.corpus import movie_reviews
from nltk.corpus import subjectivity

stpw = set(stopwords.words('english'))

class CustomDataset(Dataset):

    def __init__(self, data, tokenizer, max_len_bert):
        self.data = copy.deepcopy(data)
        self.tokenizer = tokenizer
        self.max_len_bert = max_len_bert

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['s']
        
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
        
        #decode =  self.tokenizer.convert_ids_to_tokens(encoding_text['input_ids'].flatten())
        return {
            #'text': text,
            #'decoded': decode,
            'text_input_ids': encoding_text['input_ids'].flatten(),
            'text_attention_mask': encoding_text['attention_mask'].flatten(),
            #'author': torch.Tensor([int(self.data['author_list'].index(self.data['author'][idx]))]),
            'lbl': torch.Tensor([self.data[idx]['lbl']])
        }

class CustomDataset2(Dataset):
    def __init__(self, data, tokenizer, max_len_bert):
        self.data = copy.deepcopy(data)
        self.tokenizer = tokenizer
        self.max_len_bert = max_len_bert

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        
        tkns = self.tokenizer.tokenize(text)
        
        #Split the sentence if exceeds max lenght
        if len(tkns) > self.max_len_bert-2:
            chunks = [tkns[i:i + self.max_len_bert-2] for i in range(0, len(tkns), self.max_len_bert-2)]
        
            tkchunks=[]
            for c in chunks:
                encoding_text = self.tokenizer.encode_plus(
                    c,
                    max_length=self.max_len_bert,
                    add_special_tokens=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors='pt',
                    truncation=True
                )
                tkchunks.append([encoding_text])
        else:
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
            tkchunks = [encoding_text]
            
        return tkchunks
    
def remStopwSubj(osents,ssents):
    sents=[] #1 = obj, 0 = sub
    for i in range(0,len(osents)):
        temp=[]
        for word in osents[i]:
            if word not in stpw:
               temp.append(word) 
        sents.append({'s':temp, 'lbl':1})
        temp=[]
        for word in ssents[i]:
            if word not in stpw:
               temp.append(word) 
        sents.append({'s':temp, 'lbl':0})
    return sents

def remStopwMov(words):
    fwords=[]
    for i in range(0,len(words)):
        ws = word_tokenize(words[i])
        fws = []
        for w in ws:
            if w.lower() not in stpw:# and word_counts[w] > cutoff:
                fws.append(w)
        fwords.append(' '.join(fws))
    return fwords

def prepDL(sents, train_indices, val_indices, tokenizer, batch_size):
    train_ds = [sents[i] for i in train_indices]
    train_ds = CustomDataset(train_ds, tokenizer, 128)
    valid_ds = [sents[i] for i in val_indices]
    valid_ds = CustomDataset(valid_ds, tokenizer, 128)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_ds,train_dl,valid_ds,valid_dl

def prepDL2(docsents,i,tokenizer, batch_size):
    ds = CustomDataset2(docsents[i], tokenizer, 128)
    dl = DataLoader(ds,batch_size=1,shuffle=False,pin_memory=True)
    return dl

def prepDSMov():
    #Prepare movies data
    documents = [(movie_reviews.words(file_id),category) for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id)]
    #max len sentence 15097 characters -> Average Metrics:
        #{'accuracy': 0.849, 'precision': 0.841790955775083, 'recall': 0.8630000000000001, 'f1': 0.850814182943461}
    #max len sentence 11274 characters this is only removing stopwords -> Average Metrics: 
        #{'accuracy': 0.8484999999999999, 'precision': 0.8429076702435795, 'recall': 0.859, 'f1': 0.8498115071913777}
    #max len sentence 10831 characters this is removing f<cutoff and stopwords -> Average Metrics:
        #{'accuracy': 0.845, 'precision': 0.840587532247875, 'recall': 0.8549999999999999, 'f1': 0.8466590716216527}
    
    #Train model without removing objective sentences
    words = [' '.join(words) for words, _ in documents]
    txtlbls = [category for _, category in documents]
    
    fwords = remStopwMov(words)
    
    docsents=[]
    for doc in fwords:
        docsents.append(sent_tokenize(doc))
    
    return words, txtlbls, fwords, docsents

def prepDSSubj():
    osents = [s for s in subjectivity.sents(categories='obj')] #max len 120
    ssents = [s for s in subjectivity.sents(categories='subj')] #max len 55

    sents = remStopwSubj(osents, ssents)
    return sents