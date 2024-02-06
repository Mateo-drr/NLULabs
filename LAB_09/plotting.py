# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:51:51 2024

@author: Mateo-drr
"""

import matplotlib.pyplot as plt

# Model names and perplexity values
models = ['RNN', 'LSTM', 'LSTM+DO', 'LSTM+DO+AW']
perplexity_values = [212.4671321902594, 208.07634179669785, 206.91529729862333, 199.74195779156824]

# Standard deviations for error bars
std_dev_values = [1.9222371433794985, 3.135549862330028, 1.6579621867593857, 2.3549567359927708]

# Set DPI (dots per inch)
plt.figure(dpi=300)

# Plotting
plt.errorbar(models, perplexity_values, yerr=std_dev_values, marker='o', linestyle='-', color='b', capsize=5)
plt.title('Perplexity Values for Different Models')
plt.xlabel('Models')
plt.ylabel('Perplexity')
plt.grid(True)
plt.show()


# Model names and perplexity values
models = ['WT+DO+AdW', 'WT+VDO+Adw', 'WT+VDO+SGD', 'WT+VDO+NTASGD (init ASGD)', 'WT+VDO+NTASGD (init SGD)']
perplexity_values = [206.0515699277195, 208.57814305585103, 223.30986803973352, 251.6527412195024, 225.61745778792138]

# Standard deviations for error bars
std_dev_values = [3.584362804686838, 3.011482552705973, 6.408733979756244, 6.448456545593591, 6.090788076626313]

# Set DPI (dots per inch)
plt.figure(dpi=100)

# Plotting with rotated x-axis labels
# Additional data for longer training
longer_training_models = ['WT+DO+AdW', 'WT+VDO+Adw', 'WT+VDO+SGD', 'WT+VDO+NTASGD (init ASGD)', 'WT+VDO+NTASGD (init SGD)']
longer_training_perplexity_values = [202.84950097716282, 209.9286848923745, 210.84313845700999, 206.36408510474672, 213.5234331886912]
longer_training_std_dev_values = [1.6400452451995244, 7.128901615806616, 3.4833497110803378, 2.9594777026564296, 4.765333048875472]

# Set DPI (dots per inch)
plt.figure(dpi=100)

# Plotting with rotated x-axis labels for the original data
plt.errorbar(models, perplexity_values, yerr=std_dev_values, marker='o', linestyle='-', color='b', capsize=5, label='Short Training')

# Plotting with rotated x-axis labels for the longer training data
plt.errorbar(longer_training_models, longer_training_perplexity_values, yerr=longer_training_std_dev_values, marker='o', linestyle='-', color='r', capsize=5, label='Longer Training')

plt.title('Perplexity Values for Different Models')
plt.xlabel('Models')
plt.ylabel('Perplexity')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend()  # Display legend to differentiate between short and longer training
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# Updated model names and perplexity values
models = ['WT+DO+AdW', 'WT+VDO+Adw', 'WT+VDO+SGD', 'WT+VDO+NTASGD (init ASGD)', 'WT+VDO+NTASGD (init SGD)']
perplexity_values = [202.84950097716282, 209.9286848923745, 210.84313845700999, 206.36408510474672, 213.5234331886912]

# Updated standard deviations for error bars
std_dev_values = [1.6400452451995244, 7.128901615806616, 3.4833497110803378, 2.9594777026564296, 4.765333048875472]

# Set DPI (dots per inch)
plt.figure(dpi=100)

# Plotting with rotated x-axis labels
plt.errorbar(models, perplexity_values, yerr=std_dev_values, marker='o', linestyle='-', color='b', capsize=5)
plt.title('Perplexity Values for Different Models')
plt.xlabel('Models')
plt.ylabel('Perplexity')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# Updated model names and perplexity values
models = ['WT+DO+AdW', 'WT+VDO+Adw', 'WT+VDO+SGD', 'WT+VDO+NTASGD']
perplexity_values = [192.6000879077835, 189.6482984445361, 193.33935746007955, 191.67009265534406]

# Updated standard deviations for error bars
std_dev_values = [1.5073163441670774, 2.4965727083213096, 1.4278612422564605, 2.0156080400731433]

# Set DPI (dots per inch)
plt.figure(dpi=100)

# Plotting with rotated x-axis labels
plt.errorbar(models, perplexity_values, yerr=std_dev_values, marker='o', linestyle='-', color='b', capsize=5)
plt.title('Perplexity Values for Different Models')
plt.xlabel('Models')
plt.ylabel('Perplexity')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.grid(True)
plt.show()

