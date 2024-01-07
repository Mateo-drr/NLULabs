Lab Exercise: Text Classification
Using Newsgroup dataset from scikit-learn train and evaluate Linear SVM (LinearSVC) model
Experiment with different vectorization methods and parameters, experiment_id in parentheses (e.g. CounVector, CutOff, etc.):
binary of Count Vectorization (CountVect)
TF-IDF Transformation (TF-IDF)
Using TF-IDF
min and max cut-offs (CutOff)
wihtout stop-words (WithoutStopWords)
without lowercasing (NoLowercase)
To print the results: print(experiment_id, the most appropriate score metric to report))

OUTPUT:
..........................................
Countvectorize results: 
Evaluation Scores:
	 accuracy 0.8609937159127261
	 balanced_accuracy 0.8581063882508497
	 f1_micro 0.8611264299140531
	 f1_macro 0.8588256777227701
	 f1_weighted 0.8606729341948807
Classification Report: 
                           precision    recall  f1-score   support

             alt.atheism       0.75      0.72      0.73       319
           comp.graphics       0.67      0.72      0.69       389
 comp.os.ms-windows.misc       0.70      0.65      0.68       394
comp.sys.ibm.pc.hardware       0.63      0.66      0.65       392
   comp.sys.mac.hardware       0.73      0.79      0.76       385
          comp.windows.x       0.81      0.69      0.74       395
            misc.forsale       0.80      0.88      0.84       390
               rec.autos       0.84      0.83      0.84       396
         rec.motorcycles       0.89      0.93      0.91       398
      rec.sport.baseball       0.85      0.88      0.87       397
        rec.sport.hockey       0.93      0.93      0.93       399
               sci.crypt       0.89      0.90      0.89       396
         sci.electronics       0.67      0.68      0.68       393
                 sci.med       0.82      0.77      0.80       396
               sci.space       0.88      0.88      0.88       394
  soc.religion.christian       0.82      0.91      0.86       398
      talk.politics.guns       0.72      0.84      0.77       364
   talk.politics.mideast       0.92      0.79      0.85       376
      talk.politics.misc       0.69      0.51      0.59       310
      talk.religion.misc       0.61      0.62      0.62       251

                accuracy                           0.79      7532
               macro avg       0.78      0.78      0.78      7532
            weighted avg       0.79      0.79      0.78      7532


..........................................
 
TF-IDF results: 

Evaluation Scores:
	 accuracy 0.9125081624398227
	 balanced_accuracy 0.9091955574481696
	 f1_micro 0.9125081624398227
	 f1_macro 0.9102515585261856
	 f1_weighted 0.9121960164753016
Classification Report: 
                           precision    recall  f1-score   support

             alt.atheism       0.82      0.80      0.81       319
           comp.graphics       0.76      0.80      0.78       389
 comp.os.ms-windows.misc       0.77      0.73      0.75       394
comp.sys.ibm.pc.hardware       0.71      0.76      0.74       392
   comp.sys.mac.hardware       0.84      0.86      0.85       385
          comp.windows.x       0.87      0.76      0.81       395
            misc.forsale       0.83      0.91      0.87       390
               rec.autos       0.92      0.91      0.91       396
         rec.motorcycles       0.95      0.95      0.95       398
      rec.sport.baseball       0.92      0.95      0.93       397
        rec.sport.hockey       0.96      0.98      0.97       399
               sci.crypt       0.93      0.94      0.93       396
         sci.electronics       0.81      0.79      0.80       393
                 sci.med       0.90      0.87      0.88       396
               sci.space       0.90      0.93      0.92       394
  soc.religion.christian       0.84      0.93      0.88       398
      talk.politics.guns       0.75      0.92      0.82       364
   talk.politics.mideast       0.97      0.89      0.93       376
      talk.politics.misc       0.82      0.62      0.71       310
      talk.religion.misc       0.75      0.61      0.68       251

                accuracy                           0.85      7532
               macro avg       0.85      0.85      0.85      7532
            weighted avg       0.85      0.85      0.85      7532


..........................................
 
TF-IDF cutoff results: 

Cutoff used: tokens that appear in less than 10 documents, and tokens that appear in more than 80.0 % of documents
Evaluation Scores:
	 accuracy 0.9010902640268389
	 balanced_accuracy 0.8979776166325829
	 f1_micro 0.9010902640268389
	 f1_macro 0.8987742603939655
	 f1_weighted 0.9007390864524304
Classification Report: 
                           precision    recall  f1-score   support

             alt.atheism       0.81      0.76      0.78       319
           comp.graphics       0.73      0.78      0.75       389
 comp.os.ms-windows.misc       0.74      0.72      0.73       394
comp.sys.ibm.pc.hardware       0.69      0.72      0.70       392
   comp.sys.mac.hardware       0.81      0.83      0.82       385
          comp.windows.x       0.84      0.75      0.79       395
            misc.forsale       0.84      0.90      0.87       390
               rec.autos       0.90      0.88      0.89       396
         rec.motorcycles       0.94      0.94      0.94       398
      rec.sport.baseball       0.92      0.93      0.93       397
        rec.sport.hockey       0.94      0.97      0.96       399
               sci.crypt       0.93      0.92      0.93       396
         sci.electronics       0.74      0.75      0.75       393
                 sci.med       0.88      0.86      0.87       396
               sci.space       0.89      0.93      0.91       394
  soc.religion.christian       0.85      0.93      0.89       398
      talk.politics.guns       0.74      0.90      0.81       364
   talk.politics.mideast       0.96      0.88      0.92       376
      talk.politics.misc       0.80      0.61      0.69       310
      talk.religion.misc       0.71      0.62      0.66       251

                accuracy                           0.84      7532
               macro avg       0.83      0.83      0.83      7532
            weighted avg       0.84      0.84      0.84      7532


..........................................
 
TF-IDF stopwords results: 

Evaluation Scores:
	 accuracy 0.9134375129431401
	 balanced_accuracy 0.910519199747206
	 f1_micro 0.9134375129431401
	 f1_macro 0.9114604744356782
	 f1_weighted 0.9131522197297792
Classification Report: 
                           precision    recall  f1-score   support

             alt.atheism       0.82      0.77      0.80       319
           comp.graphics       0.74      0.81      0.77       389
 comp.os.ms-windows.misc       0.78      0.74      0.76       394
comp.sys.ibm.pc.hardware       0.71      0.76      0.73       392
   comp.sys.mac.hardware       0.84      0.85      0.84       385
          comp.windows.x       0.88      0.77      0.82       395
            misc.forsale       0.82      0.91      0.86       390
               rec.autos       0.92      0.90      0.91       396
         rec.motorcycles       0.95      0.96      0.95       398
      rec.sport.baseball       0.92      0.94      0.93       397
        rec.sport.hockey       0.96      0.97      0.97       399
               sci.crypt       0.94      0.94      0.94       396
         sci.electronics       0.80      0.78      0.79       393
                 sci.med       0.92      0.87      0.89       396
               sci.space       0.90      0.93      0.91       394
  soc.religion.christian       0.85      0.94      0.89       398
      talk.politics.guns       0.75      0.92      0.83       364
   talk.politics.mideast       0.97      0.89      0.93       376
      talk.politics.misc       0.84      0.62      0.71       310
      talk.religion.misc       0.76      0.65      0.70       251

                accuracy                           0.85      7532
               macro avg       0.85      0.85      0.85      7532
            weighted avg       0.85      0.85      0.85      7532


..........................................
 
TF-IDF stpw no lowercasing results: 

Evaluation Scores:
	 accuracy 0.9105170999258881
	 balanced_accuracy 0.907506302399159
	 f1_micro 0.9105170999258881
	 f1_macro 0.9086646072403903
	 f1_weighted 0.910323270620365
Classification Report: 
                           precision    recall  f1-score   support

             alt.atheism       0.81      0.78      0.79       319
           comp.graphics       0.74      0.79      0.76       389
 comp.os.ms-windows.misc       0.76      0.74      0.75       394
comp.sys.ibm.pc.hardware       0.72      0.76      0.74       392
   comp.sys.mac.hardware       0.83      0.84      0.84       385
          comp.windows.x       0.87      0.76      0.81       395
            misc.forsale       0.82      0.91      0.87       390
               rec.autos       0.93      0.91      0.92       396
         rec.motorcycles       0.95      0.95      0.95       398
      rec.sport.baseball       0.91      0.93      0.92       397
        rec.sport.hockey       0.95      0.98      0.97       399
               sci.crypt       0.93      0.94      0.94       396
         sci.electronics       0.81      0.79      0.80       393
                 sci.med       0.89      0.87      0.88       396
               sci.space       0.90      0.93      0.92       394
  soc.religion.christian       0.85      0.93      0.89       398
      talk.politics.guns       0.76      0.92      0.83       364
   talk.politics.mideast       0.97      0.89      0.93       376
      talk.politics.misc       0.84      0.62      0.72       310
      talk.religion.misc       0.71      0.62      0.66       251

                accuracy                           0.85      7532
               macro avg       0.85      0.84      0.84      7532
            weighted avg       0.85      0.85      0.85      7532


Results comparison:

Countvectorize:
						  precision    recall  f1-score 

                accuracy                           0.79      7532
               macro avg       0.78      0.78      0.78      7532
            weighted avg       0.79      0.79      0.78      7532 

TF-IDF:
						  precision    recall  f1-score 

                accuracy                           0.85      7532
               macro avg       0.85      0.85      0.85      7532
            weighted avg       0.85      0.85      0.85      7532 

TF-IDF cutoff:
						  precision    recall  f1-score 

                accuracy                           0.84      7532
               macro avg       0.83      0.83      0.83      7532
            weighted avg       0.84      0.84      0.84      7532 

TF-IDF stopwords:
						  precision    recall  f1-score 

                accuracy                           0.85      7532
               macro avg       0.85      0.85      0.85      7532
            weighted avg       0.85      0.85      0.85      7532 

TF-IDF stpw no lowercasing:
						  precision    recall  f1-score 

                accuracy                           0.85      7532
               macro avg       0.85      0.84      0.84      7532
            weighted avg       0.85      0.85      0.85      7532 

-------------------------------------------------------------------
Comparison simplified

Countvectorize: acc: 0.79
TF-IDF: acc: 0.85
TF-IDF cutoff: acc: 0.84
TF-IDF stopwords: acc: 0.85
TF-IDF stpw no lowercasing: acc: 0.85