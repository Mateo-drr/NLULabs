Exercise (2 points)
Modify the baseline architecture Model IAS:

Add bidirectionality
Add dropout layer
*Dataset to use: ATIS*

OUTPUT:

100%|██████████| 5/5 [03:25<00:00, 41.09s/it]

 Original
F1: S: 0.9206 +- 0.001 I: 0.9206 +- 0.004
Precision: S: 0.9205 +- 0.004 I: 0.9143 +- 0.01
Recall: S: 0.9208 +- 0.003 I: 0.9353 +- 0.001
100%|██████████| 5/5 [03:09<00:00, 37.96s/it]

 Bidirectional
F1: S: 0.9396 +- 0.005 I: 0.9349 +- 0.006
Precision: S: 0.9417 +- 0.005 I: 0.9357 +- 0.008
Recall: S: 0.9375 +- 0.005 I: 0.9456 +- 0.005
100%|██████████| 5/5 [03:07<00:00, 37.40s/it]

 Bidirectional & Dropout
F1: S: 0.9398 +- 0.002 I: 0.9461 +- 0.005
Precision: S: 0.9414 +- 0.004 I: 0.9506 +- 0.003
Recall: S: 0.9383 +- 0.003 I: 0.9532 +- 0.004