# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:23:30 2024

@author: Mateo-drr
"""

import matplotlib.pyplot as plt

# Define the metrics and models
models = ['SVM', 'SVM+BERT']
accuracy = [0.8445, 0.885]
precision = [0.83924, 0.88435]
recall = [0.856, 0.889]
f1 = [0.84642, 0.8857]

#plt.figure(dpi=150)
# Plotting the metrics as a line graph
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.suptitle('Performance Metrics Comparison')

# Accuracy
ax.plot(models, accuracy, marker='o', label='Accuracy')

# Precision
ax.plot(models, precision, marker='o', label='Precision')

# Recall
ax.plot(models, recall, marker='o', label='Recall')

# F1 Score
ax.plot(models, f1, marker='o', label='F1-Score')

# Add legend, labels, and grid
ax.legend()
ax.set_ylabel('Metric Value')
#ax.set_xlabel('Models')
ax.grid(True)

# Show the plot
plt.show()


import numpy as np

# Data
metrics = ['Precision', 'Recall', 'F1-score']
categories = ['Joint', 'Aspects', 'Polarity']

precision_means = [0.165550, 0.158788, 0.175466]
precision_std_dev = [0.121862, 0.146407, 0.154877]

recall_means = [0.205892, 0.195872, 0.200202]
recall_std_dev = [0.179403, 0.188033, 0.191395]

f1score_means = [0.167491, 0.167171, 0.174985]
f1score_std_dev = [0.133474, 0.157767, 0.164038]

# Set up figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Bar width
bar_width = 0.2

# Set positions for bars on X-axis
bar_positions = np.arange(len(categories))

# Plot bars for Precision
ax.bar(bar_positions - bar_width, precision_means, bar_width, yerr=precision_std_dev,
       label='Precision', color='b', capsize=5)

# Plot bars for Recall
ax.bar(bar_positions, recall_means, bar_width, yerr=recall_std_dev,
       label='Recall', color='g', capsize=5)

# Plot bars for F1-score
ax.bar(bar_positions + bar_width, f1score_means, bar_width, yerr=f1score_std_dev,
       label='F1-score', color='r', capsize=5)

# Set labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Metrics')
ax.set_title('Metrics for Joint, Aspects, and Polarity')
ax.set_xticks(bar_positions)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Aspects Metrics
aspects_precision = 0.9427
aspects_recall = 0.8779
aspects_f1 = 0.9071
aspects_precision_error = 0.0054
aspects_recall_error = 0.0096
aspects_f1_error = 0.0065

# Polarity Metrics
polarity_precision = 0.5030
polarity_recall = 0.3775
polarity_f1 = 0.4163
polarity_precision_error = 0.0154
polarity_recall_error = 0.0143
polarity_f1_error = 0.0247

# Joint Metrics
joint_precision = 0.7074
joint_recall = 0.6256
joint_f1 = 0.6453
joint_precision_error = 0.0141
joint_recall_error = 0.0144
joint_f1_error = 0.0248

# Categories for the plot
categories = ['Aspects', 'Polarity', 'Joint']

# Values for precision, recall, and F1
precision_values = [aspects_precision, polarity_precision, joint_precision]
recall_values = [aspects_recall, polarity_recall, joint_recall]
f1_values = [aspects_f1, polarity_f1, joint_f1]

# Error values for precision, recall, and F1
precision_errors = [aspects_precision_error, polarity_precision_error, joint_precision_error]
recall_errors = [aspects_recall_error, polarity_recall_error, joint_recall_error]
f1_errors = [aspects_f1_error, polarity_f1_error, joint_f1_error]

# Plotting
fig, ax = plt.subplots(dpi=150)
bar_width = 0.25
index = range(len(categories))

ax.bar(index, precision_values, bar_width, label='Precision', yerr=precision_errors, capsize=5)
ax.bar([i + bar_width for i in index], recall_values, bar_width, label='Recall', yerr=recall_errors, capsize=5)
ax.bar([i + 2 * bar_width for i in index], f1_values, bar_width, label='F1-score', yerr=f1_errors, capsize=5)

ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Metrics with Error Bars')
ax.set_xticks([i + bar_width for i in index])
ax.set_xticklabels(categories)
ax.legend()

plt.show()
