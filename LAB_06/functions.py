# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.parse.dependencygraph import DependencyGraph
from spacy.tokenizer import Tokenizer
import spacy 
from nltk.corpus import dependency_treebank
import spacy_stanza
import nltk

def ini():
    """
    Initialize resources and download necessary data.

    Returns:
    stanza (stanza.Pipeline): Stanza pipeline for tokenization and dependency parsing.
    spcy (spacy.Language): Spacy language model for tokenization and dependency parsing.
    sents (list): List of sentences from the dependency_treebank dataset.
    psents (list): List of parsed sentences from the dependency_treebank dataset.
    """
    nltk.download('dependency_treebank')
    
    stanza = spacy_stanza.load_pipeline("en", verbose=False, tokenize_pretokenized=True)
    config = {"ext_names": {"conll_pd": "pandas"},
              "conversion_maps": {"deprel": {"nsubj": "subj", "root":"ROOT"}}}
    stanza.add_pipe("conll_formatter", config=config, last=True)
    
    spcy = spacy.load("en_core_web_sm")
    config = {"ext_names": {"conll_pd": "pandas"},
              "conversion_maps": {"deprel": {"nsubj": "subj"}}}
    spcy.add_pipe("conll_formatter", config=config, last=True)
    spcy.tokenizer = Tokenizer(spcy.vocab) 
    
    sents = dependency_treebank.sents()[-100:]
    psents = dependency_treebank.parsed_sents()[-100:]
    
    return stanza, spcy, sents, psents

def depGraphs(stanza, spcy, sents, jsents,stz,spy, dpz,dpy):
    """
    Generate DependencyGraph objects for sentences using Stanza and Spacy.

    Parameters:
    stanza (stanza.Pipeline): Stanza pipeline for tokenization and dependency parsing.
    spcy (spacy.Language): Spacy language model for tokenization and dependency parsing.
    sents (list): List of sentences from the dependency_treebank dataset.
    jsents (list): List of joined sentences.
    stz (list): List to store Stanza outputs.
    spy (list): List to store Spacy outputs.
    dpz (list): List to store DependencyGraph objects for Stanza.
    dpy (list): List to store DependencyGraph objects for Spacy.

    Returns:
    dpy (list): Updated list of DependencyGraph objects for Spacy.
    dpz (list): Updated list of DependencyGraph objects for Stanza.
    """
    
    lock = False #used for printing
    tmpy = []    
    tmpz = []
    for i,item in enumerate(sents):
        #Join the words of a sentence into a single string
        jsents.append(" ".join(item))
        stz.append(stanza(jsents[i]))
        spy.append(spcy(jsents[i]))
        
        #get the dataframe
        stz[i] = stz[i]._.pandas#["DEPREL"].replace({"root": "ROOT"})
        spy[i] = spy[i]._.pandas
        
        if not lock:
            print('Spacy Dep Tags:  ', ' '.join(spy[i]['DEPREL'].tolist()))
            print('Stanza Dep Tags:', ' '+ ' '.join(stz[i]['DEPREL'].tolist()))     
            #lock = True         
        
        #Replace tag root with ROOT
        stz[i]['DEPREL'] = stz[i]['DEPREL'].replace({"root": "ROOT"})
        
        if not lock:
            print('Spacy Dep Tags:  ', ' '.join(spy[i]['DEPREL'].tolist()))
            print('Stanza Dep Tags:', ' '+ ' '.join(stz[i]['DEPREL'].tolist()))     
            lock = True         
        
        #Take columns of interest and format into string
        tmpz.append( stz[i][["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False))
        tmpy.append(spy[i][["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False))
        
        #Create dependency graphs
        dpz.append(DependencyGraph(tmpz[i]))
        dpy.append(DependencyGraph(tmpy[i]))
        
        if i== 99:
            break
    return dpy,dpz