# Add functions or classes used for data loading and preprocessing
import torch.utils.data as data
import torch
device="cuda" if torch.cuda.is_available() else "cpu"

#CUSTOM DATASET
###########################################################################
class PennTreeBank (data.Dataset):
    """
    Custom PyTorch dataset for the Penn TreeBank corpus.

    Parameters:
    - corpus (list): List of sentences from the Penn TreeBank corpus.
    - lang: Language object containing word-to-index and index-to-word mappings.

    Methods:
    - __init__(corpus, lang): Initializes the dataset.
    - __len__(): Returns the number of samples in the dataset.
    - __getitem__(idx): Returns the source and target tensors for the specified index.
    - mapping_seq(data, lang): Maps sequences to numerical representations.
    """
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)


    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    # Auxiliary methods
    
    def mapping_seq(self, data, lang): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res
    
###############################################################################
def read_file(path, eos_token="<eos>"):
    """
    Read a file and append the end-of-sentence token to each line.

    Parameters:
    - path (str): Path to the file.
    - eos_token (str): End-of-sentence token.

    Returns:
    - list: List of lines from the file with the end-of-sentence token appended.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    """
    Build a vocabulary from the given corpus.

    Parameters:
    - corpus (list): List of sentences.
    - special_tokens (list): List of special tokens.

    Returns:
    - dict: Vocabulary with word-to-index mappings.
    """
    output = {}
    i = 0 
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

class Lang():
    """
    Language class with word-to-index and index-to-word mappings.

    Parameters:
    - corpus (list): List of sentences.
    - special_tokens (list): List of special tokens.

    Methods:
    - __init__(corpus, special_tokens): Initializes the language object.
    - get_vocab(corpus, special_tokens): Builds the vocabulary.
    """
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
        
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
        
###############################################################################
def collate_fn(data, pad_token):
    """
    Collate function for creating batches.

    Parameters:
    - data (list): List of samples.
    - pad_token (int): Padding token index.

    Returns:
    - dict: Dictionary containing 'source', 'target', and 'number_tokens'.
    """
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item