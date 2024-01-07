Lab Exercise
The exsecise is to experiment with a CRF model on NER task. You have to train and test a CRF with using different features on the conll2002 corpus (the same that we used here in the lab).

The features that you have to experiment with are:

Baseline using the fetures in sent2spacy_features
Train the model and print results on the test set
Add the "suffix" feature
Train the model and print results on the test set
Add all the features used in the tutorial on CoNLL dataset
Train the model and print results on the test set
Increase the feature window (number of previous and next token) to:
[-1, +1]
Train the model and print results on the test set
[-2, +2]
Train the model and print results on the test set
The format of the results has to be the table that we used so far, that is:

results = evaluate(tst_sents, hyp)
pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)

OUTPUT:
[nltk_data] Downloading package conll2002 to C:\Users\Mateo-
[nltk_data]     drr\AppData\Roaming\nltk_data...
[nltk_data]   Package conll2002 is already up-to-date!
---------------------------------------
Baseline
              p         r         f     s
LOC    0.724480  0.742132  0.733200   985
ORG    0.607407  0.723529  0.660403  1700
MISC   0.624183  0.429213  0.508655   445
PER    0.819588  0.650573  0.725365  1222
total  0.683759  0.677160  0.680443  4352
---------------------------------------
---------------------------------------
Suffix
              p         r         f     s
MISC   0.623003  0.438202  0.514512   445
LOC    0.657689  0.768528  0.708801   985
PER    0.810395  0.689034  0.744803  1222
ORG    0.692308  0.682941  0.687593  1700
total  0.706938  0.678998  0.692686  4352
---------------------------------------
---------------------------------------
All Features
         p    r    f     s
PER    1.0  1.0  1.0  1222
MISC   1.0  1.0  1.0   445
ORG    1.0  1.0  1.0  1700
LOC    1.0  1.0  1.0   985
total  1.0  1.0  1.0  4352
---------------------------------------
---------------------------------------
Context 1
              p         r         f     s
LOC    0.653650  0.781726  0.711974   985
PER    0.827735  0.786416  0.806546  1222
ORG    0.770129  0.737059  0.753231  1700
MISC   0.548851  0.429213  0.481715   445
total  0.735976  0.729550  0.732749  4352
---------------------------------------
---------------------------------------
Context 2
              p         r         f     s
ORG    0.741803  0.745294  0.743545  1700
PER    0.829464  0.760229  0.793339  1222
MISC   0.554517  0.400000  0.464752   445
LOC    0.662371  0.782741  0.717543   985
total  0.729191  0.722656  0.725909  4352
---------------------------------------