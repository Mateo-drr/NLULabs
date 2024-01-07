Lab Exercise: Comparative Evaluation of NLTK Tagger and Spacy Tagger
Train and evaluate NgramTagger

experiment with different tagger parameters
some of them have cut-off
Evaluate spacy POS-tags on the same test set

create mapping from spacy to NLTK POS-tags
SPACY list https://universaldependencies.org/u/pos/index.html
NLTK list https://github.com/slavpetrov/universal-pos-tags
convert output to the required format (see format above)
flatten into a list
evaluate using accuracy from nltk.metrics
link

OUTPUTS:

-------------------- Ngrams lvl 1
Cutoff lvl 0
NLTK 0.8608213982733669 | SPACY 0.8519387194969809
Cutoff lvl 1
NLTK 0.829282898348221 | SPACY 0.8190528469484505
Cutoff lvl 2
NLTK 0.7988422575976846 | SPACY 0.7894106492339937
-------------------- Ngrams lvl 2
Cutoff lvl 0
NLTK 0.1132791057437996 | SPACY 0.13109436598632665
Cutoff lvl 1
NLTK 0.07645092070462597 | SPACY 0.08139128699036878
Cutoff lvl 2
NLTK 0.06442437247367633 | SPACY 0.06721892309995509
-------------------- Ngrams lvl 3
Cutoff lvl 0
NLTK 0.06736863116922003 | SPACY 0.07615150456609611
Cutoff lvl 1
NLTK 0.049902689754977796 | SPACY 0.05334597534807126
Cutoff lvl 2
NLTK 0.042666799740506016 | SPACY 0.04481261539997006
--------------------
NLTK best ACC: 0.8608213982733669 vs SPACY ACC: 0.8623184789660163