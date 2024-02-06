Exercise (2 points)
Modify the baseline architecture Model IAS:

Add bidirectionality
Add dropout layer
*Dataset to use: ATIS*

OUTPUT:

 20%|██        | 1/5 [00:23<01:35, 23.77s/it]Patience reached E: 105
 40%|████      | 2/5 [00:44<01:05, 21.93s/it]Patience reached E: 90
 60%|██████    | 3/5 [01:02<00:40, 20.38s/it]Patience reached E: 85
 80%|████████  | 4/5 [01:23<00:20, 20.50s/it]Patience reached E: 95
100%|██████████| 5/5 [01:47<00:00, 21.53s/it]Patience reached E: 110

 Original
F1: S: 0.9268 +- 0.004 I: 0.9198 +- 0.003
Precision: S: 0.9263 +- 0.005 I: 0.9139 +- 0.008
Recall: S: 0.9273 +- 0.004 I: 0.9359 +- 0.002

 20%|██        | 1/5 [00:33<02:14, 33.59s/it]Patience reached E: 115
 40%|████      | 2/5 [01:08<01:42, 34.13s/it]Patience reached E: 120
 60%|██████    | 3/5 [01:34<01:01, 30.55s/it]Patience reached E: 90
 80%|████████  | 4/5 [01:56<00:27, 27.09s/it]Patience reached E: 75
100%|██████████| 5/5 [02:30<00:00, 30.20s/it]Patience reached E: 120


 Bidirectional
F1: S: 0.9448 +- 0.001 I: 0.9475 +- 0.006
Precision: S: 0.9466 +- 0.002 I: 0.9507 +- 0.006
Recall: S: 0.9429 +- 0.003 I: 0.955 +- 0.005
 20%|██        | 1/5 [00:16<01:04, 16.02s/it]Patience reached E: 55
 40%|████      | 2/5 [00:42<01:05, 21.97s/it]Patience reached E: 90
 60%|██████    | 3/5 [01:01<00:41, 20.54s/it]Patience reached E: 65
 80%|████████  | 4/5 [01:37<00:26, 26.98s/it]Patience reached E: 125
Patience reached E: 70
100%|██████████| 5/5 [02:10<00:00, 26.16s/it]
 Bidirectional & Dropout
F1: S: 0.9456 +- 0.003 I: 0.9575 +- 0.004
Precision: S: 0.9459 +- 0.004 I: 0.9595 +- 0.004
Recall: S: 0.9453 +- 0.004 I: 0.9624 +- 0.004