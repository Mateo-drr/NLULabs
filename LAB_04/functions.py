# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import nltk
from nltk.corpus import treebank
import spacy
from spacy.tokenizer import Tokenizer
from nltk.tag import NgramTagger, untag
from nltk.metrics import accuracy
import spacy.cli
from itertools import chain

#test split indes
idx=3131

def res(nk, sp):
    for i in range(0,len(nk)):
        print('--------------------', 'Ngrams lvl', i+1)
        for clvl in range(0,len(nk[i])):
            print('Cutoff lvl', clvl)
            print('NLTK', nk[i][clvl], '| SPACY', sp[i][clvl])
            
    print('--------------------')
    
    
def loadData():
    nltk.download('treebank')
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    # We overwrite the spacy tokenizer with a custom one, that split by whitespace only
    nlp.tokenizer = Tokenizer(nlp.vocab) # Tokenize by whitespace
    #Sentences
    sents = treebank.tagged_sents()
    usents = treebank.sents()
    return sents, usents, nlp

def spacyTag(usents, nlp, mapping_spacy_to_NLTK):
    #Spacy tagger
    # Tag the train set with SpaCy and convert to NLTK format
    spacy_test_sents = []
    for sent in usents:
        spacy_sent = []
        for word in nlp(' '.join(sent)):
            spacy_tag = mapping_spacy_to_NLTK.get(word.pos_, 'XX')
            spacy_sent.append((word.text, spacy_tag))
        spacy_test_sents.append(spacy_sent)
    return spacy_test_sents
    
def testTag(sents, usents, spacy_test_sents):
    nk=[[],[],[]]
    sp=[[],[],[]]
    
    #loop for different ngrams and cutoffs
    for n in range(1,4):
        for cutoff in range(0,3):
            if True:#for bkf in range(0,2):

                #NLTK
                ngram_tagger = NgramTagger(n, train=sents[:idx], cutoff=cutoff)
                tsents = ngram_tagger.tag_sents(usents) #tag the words
                ngram_accuracy = accuracy(list(chain.from_iterable(sents[idx:])), list(chain.from_iterable(tsents)))
                nk[n-1].append(ngram_accuracy)
                    
                #SPACY
                spacy_tagger = NgramTagger(n,train=spacy_test_sents[:idx], cutoff=cutoff)
                spacy_tsents = spacy_tagger.tag_sents(usents)
                ngram_accuracy = accuracy(list(chain.from_iterable(spacy_test_sents[idx:])),list(chain.from_iterable(spacy_tsents)))
                sp[n-1].append(ngram_accuracy)
                
    return nk, sp

def spacyAcc(spacy_sents):
    
    #test_tags = get_mapping_tags_to_nltk(nlp, start_index=train_index)
    tags = treebank.tagged_sents(tagset='universal')[idx:]
    tags = list(chain.from_iterable(tags))
    accuracy_spacy = accuracy(list(chain.from_iterable(spacy_sents[idx:])), tags)
    return accuracy_spacy