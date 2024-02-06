import matplotlib.pyplot as plt
import numpy as np

# Provided data
models = ['Original', 'Bidirectional', 'Bidirectional & Dropout', 'Bert Simple', 'Bert Advanced']

# F1 Scores
f1_slots = [0.9206, 0.9396, 0.9398, 0.9159, 0.9241]
f1_intents = [0.9206, 0.9349, 0.9461, 0.9586, 0.9732]
f1_slots_std = [0.001, 0.005, 0.002, 0.002, 0.0025]
f1_intents_std = [0.004, 0.006, 0.005, 0.0024, 0.0021]

# Precision Scores
precision_slots = [0.9205, 0.9417, 0.9414, 0.9125, 0.9223]
precision_intents = [0.9143, 0.9357, 0.9506, 0.9532, 0.9735]
precision_slots_std = [0.004, 0.005, 0.004, 0.0041, 0.0037]
precision_intents_std = [0.01, 0.008, 0.003, 0.0015, 0.0016]

# Recall Scores
recall_slots = [0.9208, 0.9375, 0.9383, 0.9193, 0.9258]
recall_intents = [0.9353, 0.9456, 0.9532, 0.9664, 0.976]
recall_slots_std = [0.003, 0.005, 0.003, 0.0015, 0.0016]
recall_intents_std = [0.001, 0.005, 0.004, 0.0023, 0.0023]

# Plotting Intents
plt.figure(figsize=(10, 6), dpi=150)
plt.errorbar(models, f1_intents, yerr=f1_intents_std, marker='o', linestyle='-', color='blue', label='F1')
plt.errorbar(models, precision_intents, yerr=precision_intents_std, marker='s', linestyle='--', color='orange', label='Precision')
plt.errorbar(models, recall_intents, yerr=recall_intents_std, marker='^', linestyle=':', color='green', label='Recall')
plt.title('Performance Metrics for Intents')
plt.ylabel('Scores')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Slots
plt.figure(figsize=(10, 6), dpi=150)
plt.errorbar(models, f1_slots, yerr=f1_slots_std, marker='o', linestyle='-', color='blue', label='F1')
plt.errorbar(models, precision_slots, yerr=precision_slots_std, marker='s', linestyle='--', color='orange', label='Precision')
plt.errorbar(models, recall_slots, yerr=recall_slots_std, marker='^', linestyle=':', color='green', label='Recall')
plt.title('Performance Metrics for Slots')
plt.ylabel('Scores')
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# Provided data
models = ['Original', 'Bidirectional', 'Bidirectional & Dropout', 'Bert']

# F1 Scores
f1_slots = [0.9206, 0.9396, 0.9398, 0.9563]
f1_intents = [0.9206, 0.9349, 0.9461, 0.9752]
f1_slots_std = [0.001, 0.005, 0.002, 0.002]
f1_intents_std = [0.004, 0.006, 0.005, 0.0013]

# Precision Scores
precision_slots = [0.9205, 0.9417, 0.9414, 0.9533]
precision_intents = [0.9143, 0.9357, 0.9506, 0.9754]
precision_slots_std = [0.004, 0.005, 0.004, 0.0024]
precision_intents_std = [0.01, 0.008, 0.003, 0.0011]

# Recall Scores
recall_slots = [0.9208, 0.9375, 0.9383, 0.9592]
recall_intents = [0.9353, 0.9456, 0.9532, 0.9781]
recall_slots_std = [0.003, 0.005, 0.003, 0.0016]
recall_intents_std = [0.001, 0.005, 0.004, 0.0015]

# Plotting Intents
plt.figure(figsize=(10, 6), dpi=150)
plt.errorbar(models, f1_intents, yerr=f1_intents_std, marker='o', linestyle='-', color='blue', label='F1')
plt.errorbar(models, precision_intents, yerr=precision_intents_std, marker='s', linestyle='--', color='orange', label='Precision')
plt.errorbar(models, recall_intents, yerr=recall_intents_std, marker='^', linestyle=':', color='green', label='Recall')
plt.title('Performance Metrics for Intents')
plt.ylabel('Scores')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Slots
plt.figure(figsize=(10, 6), dpi=150)
plt.errorbar(models, f1_slots, yerr=f1_slots_std, marker='o', linestyle='-', color='blue', label='F1')
plt.errorbar(models, precision_slots, yerr=precision_slots_std, marker='s', linestyle='--', color='orange', label='Precision')
plt.errorbar(models, recall_slots, yerr=recall_slots_std, marker='^', linestyle=':', color='green', label='Recall')
plt.title('Performance Metrics for Slots')
plt.ylabel('Scores')
plt.legend()
plt.grid(True)
plt.show()
