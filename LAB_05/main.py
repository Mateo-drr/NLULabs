# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import nltk
from nltk import Nonterminal
from nltk.corpus import treebank
from nltk.parse.generate import generate
from pcfg import PCFG

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    print(treebank)

    productions = []
    # let's keep it small
    for item in treebank.fileids():
        for tree in treebank.parsed_sents(item):
            productions += tree.productions()
    S = Nonterminal('S')
    grammar = nltk.induce_pcfg(S, productions)
    
    # test setenteces
    test_sents = [
        "there is a plane stuck in a tree", #"There is a rabbit stuck in a tree", 
        "I want to return it to the wild so it can lay its things",#"I want to return it to the wild so it can lay its eggs",
        "I can not control the weather",
        ]
    
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
    
    # weighted_rules = [
    #     #S1
    #     'S -> NP VP [0.67]', #affected by third sentence
        
    #     'NP -> VP NP [0.2]', #rounding
    #     'NP -> DT N [0.16]',
        
    #     'VP -> DT V [0.16]',
    #     'VP -> V PP [0.14]',
        
    #     'PP -> P DT N [1.0]',
        
    #     'DT -> "There" [0.25]',
    #     'DT -> "a" [0.5]',
        
    #     'N -> "plane" [0.2]',#'N -> "rabbit" [0.25]',
    #     'N -> "tree" [0.2]',
        
    #     'V -> "is" [0.125]',
    #     'V -> "stuck" [0.125]',
        
    #     'P -> "in" [1.0]',
        
    #     #S2
    #     'S -> VP VP [0.33]',
        
    #     'VP -> VP NP [0.14]', #x2 NP
    #     'VP -> NP VP [0.14]',
    #     'VP -> PRON V TO V  [0.14]',
    #     'VP -> V V [0.14]',
        
    #     'NP -> PRON TO DT N [0.16]',
    #     'NP -> RB PRON [0.16]', #RB adverb
    #     'NP -> PRP N [0.16]', #pos pron
        
    #     'PRON -> "I" [0.5]',# 0.5 wot prp$
    #     'PRON -> "it" [0.5]',
        
    #     'V -> "want" [0.125]',
    #     'V -> "return" [0.125]',
    #     'V -> "can" [0.25]', #affected by third sentence
    #     'V -> "lay" [0.125]',
        
    #     'TO -> "to" [1.0]',
        
    #     'DT -> "the"[0.25]',
        
    #     'N -> "wild"[0.2]',
    #     'N -> "things"[0.2]',#'N -> "eggs"[0.25]',
        
    #     'RB -> "so" [0.5]',
        
    #     'PRP -> "its" [1.0]',
        
    #     #S3
    #     #'S -> NP VP' repeated so prob at beggining increased
        
    #     'NP -> PRON [0.16]',
        
    #     'VP -> V RB V DT N [0.14]',
        
    #     #'V -> "can"' repeated
    #     'V -> "control" [0.125]',
        
    #     'RB -> "not" [0.5]',
        
    #     #'DT -> "the"' repeated
        
    #     'N -> "weather" [0.2]'
    # ]
    
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
    
    
    # #CFG parser
    # crules = nltk.CFG.fromstring(rules)
    # parser = nltk.ChartParser(crules)
    
    # prod = []
    # for sent in test_sents:
    #     for tree in parser.parse(sent.split()):
    #         #print(tree.pretty_print())
    #         prod += tree.productions()
    
    # #Get an autogenerated PCFG
    # S = Nonterminal('S')
    # prod_pcfg = nltk.induce_pcfg(S, prod)
    # prod_pcfg.productions()
    
    # print('------------------------------------------------------')
    # print('Generating 10 sentences:')
    # for sent in generate(prod_pcfg, start = S ,depth=5, n=3):
    #     print(sent)

    # #Formatting the rules
    # gen_gram =[]
    # for rule in prod_pcfg.productions():
    #     gen_gram.append(str(rule))
    
    # print('------------------------------------------------------')
    # print('Generating 10 sentences using PCFG.generate():')
    # gram = PCFG.fromstring(gen_gram)
    # for sent in gram.generate(3):
    #     print(sent)
    
    
    #Viterbi Parser
    toy_grammar = nltk.PCFG.fromstring(weighted_rules)
    
    for sent in test_sents:
        parser = nltk.ViterbiParser(grammar)
        for tree in parser.parse(sent.split()):
            print(tree)
            print(tree.pretty_print())
    
        parser = nltk.ViterbiParser(toy_grammar)
        for tree in parser.parse(sent.split()):
            print('a', tree)
            print(tree.pretty_print())
    
    
    #Generate sentences from manual PCFG
    toy_grammar = nltk.PCFG.fromstring(weighted_rules)
    for sent in generate(toy_grammar, depth=8, n=3):
        print(sent)

    from pcfg import PCFG
    toy_grammar = PCFG.fromstring(weighted_rules)
    for sent in toy_grammar.generate(10):
        print(sent)
