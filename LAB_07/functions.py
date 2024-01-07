# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import multiprocessing
import spacy
from spacy.tokenizer import Tokenizer
import es_core_news_sm
nlp = es_core_news_sm.load()
nlp.tokenizer = Tokenizer(nlp.vocab)
import copy

from nltk.corpus import conll2002
from sklearn_crfsuite import CRF
from conllm import evaluate

def sent2spacy_features(sent):
    """
    Convert a sentence to a list of spacy features.

    Args:
        sent (list): A list of tokens in a sentence.

    Returns:
        list: A list of dictionaries containing spacy features for each token in the sentence.
    """
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)

    return feats

def sent2spacy_features_s(sent):
    """
    Convert a sentence to a list of spacy features with additional suffix information.

    Args:
        sent (list): A list of tokens in a sentence.

    Returns:
        list: A list of dictionaries containing spacy features with suffix information for each token in the sentence.
    """
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_,
            'suffix': token.suffix_
        }
        feats.append(token_feats)

    return feats

def allfeats(sent, i):
    """
    Extract features for a given token in the context of the entire sentence.

    Args:
        sent (list): A list of tokens in a sentence.
        i (int): Index of the target token in the sentence.

    Returns:
        dict: A dictionary containing features for the target token in the given context.
    """
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features

def featsWcon1(trn_feats):
    """
    Add contextual information to each token in the dataset with one-token context.

    Args:
        trn_feats (list): A list of sentences, where each sentence is a list of token features.

    Returns:
        list: A modified list of sentences with contextual information added to each token.
    """
    #Join context to each token
    f = copy.deepcopy(trn_feats)#[[] for _ in range(len(trn_feats))]
    for i in range(len(trn_feats)): #each sentence
        for j in range(len(trn_feats[i])): #each token
        
            if j != 0 and j != len(trn_feats[i])-1:
                f[i][j] = trn_feats[i][j-1],trn_feats[i][j],trn_feats[i][j+1]
                
        if len(trn_feats[i]) > 1:
                f[i][0] = trn_feats[i][0], trn_feats[i][1]
                f[i][-1] = trn_feats[i][-2],trn_feats[i][-1]
    
    #Combine tokens into one dict
    ss = copy.deepcopy(f)
    for i in range(len(f)):
        for j in range(len(f[i])):
            
            if len(f[i][j]) > 2 and len(f[i]) > 1:
                #print(i,j)
                ss[i][j] = {
                            'bias-c': f[i][j][0]['bias'],
                            'word.lower()-c': f[i][j][0]['word.lower()'],
                            'pos-c': f[i][j][0]['pos'],
                            'lemma-c': f[i][j][0]['lemma'],
                            
                           'bias': f[i][j][1]['bias'],
                           'word.lower()': f[i][j][1]['word.lower()'],
                           'pos': f[i][j][1]['pos'],
                           'lemma': f[i][j][1]['lemma'],
                           
                           'bias+c': f[i][j][2]['bias'],
                           'word.lower()+c': f[i][j][2]['word.lower()'],
                           'pos+c': f[i][j][2]['pos'],
                           'lemma+c': f[i][j][2]['lemma']
                           }
        
        #For beggining and end tokens
        if len(f[i])>1: #avoid sentences with a single word
            ss[i][0] = {
                       'bias': f[i][0][0]['bias'],
                       'word.lower()': f[i][0][0]['word.lower()'],
                       'pos': f[i][0][0]['pos'],
                       'lemma': f[i][0][0]['lemma'],
                       
                       'bias+c': f[i][0][1]['bias'],
                       'word.lower()+c': f[i][0][1]['word.lower()'],
                       'pos+c': f[i][0][1]['pos'],
                       'lemma+c': f[i][0][1]['lemma']
                       }
            ss[i][-1] = {
                        'bias-c': f[i][-1][0]['bias'],
                        'word.lower()-c': f[i][-1][0]['word.lower()'],
                        'pos-c': f[i][-1][0]['pos'],
                        'lemma-c': f[i][-1][0]['lemma'],
                        
                       'bias': f[i][-1][1]['bias'],
                       'word.lower()': f[i][-1][1]['word.lower()'],
                       'pos': f[i][-1][1]['pos'],
                       'lemma': f[i][-1][1]['lemma'],
                      
                       }        
    return ss

def featsWcon2(trn_feats):
    """
    Add contextual information to each token in the dataset with two-token context.

    Args:
        trn_feats (list): A list of sentences, where each sentence is a list of token features.

    Returns:
        list: A modified list of sentences with contextual information added to each token.
    """
    #Join context to each token
    f = copy.deepcopy(trn_feats)#[[] for _ in range(len(trn_feats))]
    for i in range(len(trn_feats)): #each sentence
        for j in range(len(trn_feats[i])): #each token
        
            if j != 0 and j!=1 and j != len(trn_feats[i])-1 and j != len(trn_feats[i])-2: #w0,w1,w2,w3,w4 avoid w0,w1,w3,w4
                f[i][j] = trn_feats[i][j-2],trn_feats[i][j-1],trn_feats[i][j],trn_feats[i][j+1],trn_feats[i][j+2]
                
        if len(trn_feats[i]) > 2:  # w0, w1, w2
            f[i][0] = trn_feats[i][0], trn_feats[i][1], trn_feats[i][2]
            f[i][-1] = trn_feats[i][-3], trn_feats[i][-2], trn_feats[i][-1]
        elif len(trn_feats[i]) > 1:  # w0, w1
            f[i][0] = trn_feats[i][0], trn_feats[i][1]
            f[i][-1] = trn_feats[i][-2], trn_feats[i][-1]
            
        if len(trn_feats[i]) > 3: #w0 w1 w2 w3
            f[i][1] = trn_feats[i][0], trn_feats[i][1], trn_feats[i][2], trn_feats[i][3] #w1 -> w0 (w1) w2 w3  
            f[i][-2] = trn_feats[i][-4],trn_feats[i][-3],trn_feats[i][-2],trn_feats[i][-1] #w2 -> w0 w1 (w2) w3 
    
    #Combine tokens into one dict
    ss = copy.deepcopy(f)
    for i in range(len(f)): #sentce
        for j in range(len(f[i])): # w+context
            
            if len(f[i][j]) > 4 and len(f[i]) > 1: #w+context len > 2 and check for 1w sentences
                #print(i,j)
                ss[i][j] = {
                            'bias-2c': f[i][j][0]['bias'],
                            'word.lower()-2c': f[i][j][0]['word.lower()'],
                            'pos-2c': f[i][j][0]['pos'],
                            'lemma-2c': f[i][j][0]['lemma'],
                    
                            'bias-c': f[i][j][1]['bias'],
                            'word.lower()-c': f[i][j][1]['word.lower()'],
                            'pos-c': f[i][j][1]['pos'],
                            'lemma-c': f[i][j][1]['lemma'],
                            
                           'bias': f[i][j][2]['bias'],
                           'word.lower()': f[i][j][2]['word.lower()'],
                           'pos': f[i][j][2]['pos'],
                           'lemma': f[i][j][2]['lemma'],
                           
                           'bias+c': f[i][j][3]['bias'],
                           'word.lower()+c': f[i][j][3]['word.lower()'],
                           'pos+c': f[i][j][3]['pos'],
                           'lemma+c': f[i][j][3]['lemma'],
                           
                           'bias+2c': f[i][j][4]['bias'],
                           'word.lower()+2c': f[i][j][4]['word.lower()'],
                           'pos+2c': f[i][j][4]['pos'],
                           'lemma+2c': f[i][j][4]['lemma']
                           }
        
        #For beggining and end tokens
        if len(f[i])>1: #avoid sentences with a single word
            #First and last words
            if len(f[i])>2:
                ss[i][0] = {
                           'bias': f[i][0][0]['bias'],
                           'word.lower()': f[i][0][0]['word.lower()'],
                           'pos': f[i][0][0]['pos'],
                           'lemma': f[i][0][0]['lemma'],
                           
                           'bias+c': f[i][0][1]['bias'],
                           'word.lower()+c': f[i][0][1]['word.lower()'],
                           'pos+c': f[i][0][1]['pos'],
                           'lemma+c': f[i][0][1]['lemma'],
                           
                           'bias+2c': f[i][0][2]['bias'],
                           'word.lower()+2c': f[i][0][2]['word.lower()'],
                           'pos+2c': f[i][0][2]['pos'],
                           'lemma+2c': f[i][0][2]['lemma']
                           }
                ss[i][-1] = {
                            'bias-2c': f[i][-1][0]['bias'],
                            'word.lower()-2c': f[i][-1][0]['word.lower()'],
                            'pos-2c': f[i][-1][0]['pos'],
                            'lemma-2c': f[i][-1][0]['lemma'],
                            
                           'bias-c': f[i][-1][1]['bias'],
                           'word.lower()-c': f[i][-1][1]['word.lower()'],
                           'pos-c': f[i][-1][1]['pos'],
                           'lemma-c': f[i][-1][1]['lemma'],
                           
                           'bias': f[i][-1][2]['bias'],
                           'word.lower()': f[i][-1][2]['word.lower()'],
                           'pos': f[i][-1][2]['pos'],
                           'lemma': f[i][-1][2]['lemma']
                           }
            else:
                ss[i][0] = {
                           'bias': f[i][0][0]['bias'],
                           'word.lower()': f[i][0][0]['word.lower()'],
                           'pos': f[i][0][0]['pos'],
                           'lemma': f[i][0][0]['lemma'],
                           
                           'bias+c': f[i][0][1]['bias'],
                           'word.lower()+c': f[i][0][1]['word.lower()'],
                           'pos+c': f[i][0][1]['pos'],
                           'lemma+c': f[i][0][1]['lemma']
                           }
                ss[i][-1] = {
                            'bias-c': f[i][-1][0]['bias'],
                            'word.lower()-c': f[i][-1][0]['word.lower()'],
                            'pos-c': f[i][-1][0]['pos'],
                            'lemma-c': f[i][-1][0]['lemma'],
                            
                           'bias': f[i][-1][1]['bias'],
                           'word.lower()': f[i][-1][1]['word.lower()'],
                           'pos': f[i][-1][1]['pos'],
                           'lemma': f[i][-1][1]['lemma'],
                           }
            if len(f[i])>3:
                #second and penultimate words
                ss[i][1] = {
                            'bias-c': f[i][1][0]['bias'],
                            'word.lower()-c': f[i][1][0]['word.lower()'],
                            'pos-c': f[i][1][0]['pos'],
                            'lemma-c': f[i][1][0]['lemma'],
                    
                           'bias': f[i][1][1]['bias'],
                           'word.lower()': f[i][1][1]['word.lower()'],
                           'pos': f[i][1][1]['pos'],
                           'lemma': f[i][1][1]['lemma'],
                           
                           'bias+c': f[i][1][2]['bias'],
                           'word.lower()+c': f[i][1][2]['word.lower()'],
                           'pos+c': f[i][1][2]['pos'],
                           'lemma+c': f[i][1][2]['lemma'],
                           
                           'bias+2c': f[i][1][3]['bias'],
                           'word.lower()+2c': f[i][1][3]['word.lower()'],
                           'pos+2c': f[i][1][3]['pos'],
                           'lemma+2c': f[i][1][3]['lemma']
                           }
                ss[i][-2] = {
                            'bias-2c': f[i][-2][0]['bias'],
                            'word.lower()-2c': f[i][-2][0]['word.lower()'],
                            'pos-2c': f[i][-2][0]['pos'],
                            'lemma-2c': f[i][-2][0]['lemma'],
                            
                           'bias-c': f[i][-2][1]['bias'],
                           'word.lower()-c': f[i][-2][1]['word.lower()'],
                           'pos-c': f[i][-2][1]['pos'],
                           'lemma-c': f[i][-2][1]['lemma'],
                           
                           'bias': f[i][-2][2]['bias'],
                           'word.lower()': f[i][-2][2]['word.lower()'],
                           'pos': f[i][-2][2]['pos'],
                           'lemma': f[i][-2][2]['lemma'],
                           
                           'bias+c': f[i][-2][3]['bias'],
                           'word.lower()+c': f[i][-2][3]['word.lower()'],
                           'pos+c': f[i][-2][3]['pos'],
                           'lemma+c': f[i][-2][3]['lemma']
                           } 
    return ss

def sent2features(sent):
    """
    Convert a sentence to a list of features for each token.

    Args:
        sent (list): A list of tokens in a sentence.

    Returns:
        list: A list of dictionaries containing features for each token in the sentence.
    """
    return [allfeats(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """
    Extract labels from a sentence.

    Args:
        sent (list): A list of tokens in a sentence.

    Returns:
        list: A list of labels corresponding to each token in the sentence.
    """
    return [label for token, label in sent]

def sent2tokens(sent):
    """
    Extract tokens from a sentence.

    Args:
        sent (list): A list of tokens in a sentence.

    Returns:
        list: A list of tokens in the sentence.
    """
    return [token for token, label in sent]

def word2features(sent, i):
    """
    Extract features for a given token.

    Args:
        sent (list): A list of tokens in a sentence.
        i (int): Index of the target token in the sentence.

    Returns:
        dict: A dictionary containing features for the target token.
    """
    word = sent[i][0]
    return {'bias': 1.0, 'word.lower()': word.lower()}

def spacyfeats(trn_sents, tst_sents):
    """
    Extract spacy features from the training and testing datasets.

    Args:
        trn_sents (list): A list of training sentences.
        tst_sents (list): A list of testing sentences.

    Returns:
        tuple: A tuple containing training and testing spacy features and labels.
    """
    #format data to be used for training
    trn_feats = [sent2spacy_features(s) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]
    tst_feats = [sent2spacy_features(s) for s in tst_sents]
    tst_label = [sent2labels(s) for s in tst_sents]
    return trn_feats, trn_label, tst_feats, tst_label

def suffeats(trn_sents, tst_sents):
    """
    Extract features with suffix information from the training and testing datasets.

    Args:
        trn_sents (list): A list of training sentences.
        tst_sents (list): A list of testing sentences.

    Returns:
        tuple: A tuple containing training and testing features with suffix information.
    """
    trn_feats_s = [sent2spacy_features_s(s) for s in trn_sents]
    tst_feats_s = [sent2spacy_features_s(s) for s in tst_sents]
    return trn_feats_s, tst_feats_s
    
def afeats(trn_sents, tst_sents):
    """
    Extract all features from the training and testing datasets.

    Args:
        trn_sents (list): A list of training sentences.
        tst_sents (list): A list of testing sentences.

    Returns:
        tuple: A tuple containing training and testing all features.
    """
    trn_feats_a = [sent2features(s) for s in trn_sents]
    tst_feats_a = [sent2features(s) for s in tst_sents]
    return trn_feats_a, tst_feats_a

def runcrf(trn_feats, trn_label, tst_feats, tst_sents):
    """
    Run a CRF model on the provided datasets and evaluate the performance.

    Args:
        trn_feats: Features for training.
        trn_label: Labels for training.
        tst_feats: Features for testing.
        tst_sents: Testing sentences.

    Returns:
        dict: Evaluation metrics for the CRF model on the testing dataset.
    """
    
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    # workaround for scikit-learn 1.0
    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    # Predict the labels of the input dataset
    pred = crf.predict(tst_feats)
    # Convert the predicted labels to the correct format
    hyp = [[(tst_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]
    
    return evaluate(tst_sents, hyp)