# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import nltk
from nltk import Nonterminal
from nltk.corpus import treebank
from nltk.parse.generate import generate
from pcfg import PCFG

rules = ['S -> NP VP | VP VP',
        'NP -> VP NP | DT N | PRON TO DT N | RB PRON | PRON N | PRON',
        'VP -> DT V | V PP | VP NP | NP VP | PRON V TO V | V V | V RB V DT N',
        'PP -> P DT N',
        'DT -> "There" | "a" | "the"',
        'N -> "plane" | "tree" | "wild" | "things" | "weather"',
        'V -> "is" | "stuck" | "want" | "return" | "can" | "lay" | "control"',
        'P -> "in"',
        'PRON -> "I" | "it" | "its"',
        'TO -> "to"',
        'RB -> "so" | "not"'
        ]

weighted_rules = [
    #S1
    'S -> NP VP [0.9]',
    
    'NP -> DT V DT N [0.35]',
    'VP -> V PP [0.35]',
    'PP -> P DT N [0.5]',
    
    #S2
    'S -> NP S-NOM [0.1]' ,
    
    'NP -> PRON VP PP [0.35]',
    'VP -> V TO V PRON [0.3]',
    'PP -> TO DT N [0.5]',
    
    #'VP -> RB S-NOM [0.25]' ,
    'S-NOM -> IN PRON MD V PRP N [1.0]',
    
    #S3
    #'S -> NP VP',

    'NP -> PRON [0.3]',
    'VP -> MD RB V DT N [0.35]',
    
    #VOCAB
    'DT -> "there" [0.2]',
    'DT -> "a" [0.4]',
    #'DT -> "a" [0.5]', repeated
    'DT -> "the"[0.4]',
    #'DT -> "the"' repeated
    
    'N -> "plane" [0.2]',#'N -> "rabbit" [0.25]',
    'N -> "tree" [0.2]',
    'N -> "wild"[0.2]',
    'N -> "things"[0.2]',#'N -> "eggs"[0.25]',
    'N -> "weather" [0.2]',
    
    'V -> "is" [0.17]',
    'V -> "stuck" [0.16]',
    'V -> "want" [0.17]',
    'V -> "return" [0.16]',
    'V -> "lay" [0.17]',
    'V -> "control" [0.17]',
    
    'P -> "in" [1.0]',
    
    'PRON -> "I" [0.5]',# 0.5 wot prp$
    'PRON -> "it" [0.5]',
    
    'TO -> "to" [1.0]',

    'RB -> "not" [1.0]',
    
    'IN -> "so" [1.0]', #Subordinating Conjunction
    
    'PRP -> "its" [1.0]',
    
    'MD -> "can" [1.0]'
    
    ]

def getGrammar():
    """
    Get a PCFG from the Treebank dataset.
 
    Returns:
    grammar (nltk.grammar.PCFG): Probabilistic Context-Free Grammar induced from the Treebank dataset.
    """
    productions = []
    for item in treebank.fileids():
        for tree in treebank.parsed_sents(item):
            productions += tree.productions()
    S = Nonterminal('S')
    grammar = nltk.induce_pcfg(S, productions)
    return grammar

def cstmGrammar():
    """
    Define a custom PCFG.

    Returns:
    cstm_grammar (nltk.grammar.PCFG): Custom Probabilistic Context-Free Grammar.
    """
    cstm_grammar = nltk.PCFG.fromstring(weighted_rules)
    return cstm_grammar

def printTrees(grammar,test_sents):
    """
    Parse and print parse trees for given sentences using the Viterbi parser.

    Parameters:
    grammar (nltk.grammar.PCFG): Probabilistic Context-Free Grammar.
    test_sents (list): List of sentences to parse and print.
    """
    for sent in test_sents:
        parser = nltk.ViterbiParser(grammar)
        for tree in parser.parse(sent.split()):
            #print(tree)
            print(tree.pretty_print())