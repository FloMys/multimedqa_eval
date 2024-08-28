import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Read the CSV file
file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
df = pd.read_csv(file)

def calculate_calibration(y_true, y_pred, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    return prob_true, prob_pred

def plot_calibration(ax, prob_true, prob_pred, label):
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label=label)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')

# Prepare data for GPT-4 and GPT-3.5
gpt4_correct = df['correct_GPT4_zeroshot'].astype(int)
gpt4_prob = df['GPT4_zeroshot_prob']
gpt35_correct = df['correct_GPT35_zeroshot'].astype(int)
gpt35_prob = df['GPT35_zeroshot_prob']

# Calculate calibration curves
prob_true_gpt4, prob_pred_gpt4 = calculate_calibration(gpt4_correct, gpt4_prob)
prob_true_gpt35, prob_pred_gpt35 = calculate_calibration(gpt35_correct, gpt35_prob)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))

plot_calibration(ax, prob_true_gpt4, prob_pred_gpt4, 'GPT-4')
plot_calibration(ax, prob_true_gpt35, prob_pred_gpt35, 'GPT-3.5')

ax.set_title('Calibration plot')
plt.tight_layout()
plt.show()

# Print average probabilities and accuracy for each model
print(f"GPT-4 average probability: {gpt4_prob.mean():.3f}")
print(f"GPT-4 accuracy: {gpt4_correct.mean():.3f}")
print(f"GPT-3.5 average probability: {gpt35_prob.mean():.3f}")
print(f"GPT-3.5 accuracy: {gpt35_correct.mean():.3f}")