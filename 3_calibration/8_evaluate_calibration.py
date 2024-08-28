import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from collections import Counter

# Read the CSV file
file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
df = pd.read_csv(file)

# Extract relevant columns
calibration = df['Answer Calibration GPT4o 0-shot']
status = df['Answer Status GPT4o 0shot']

# Function to clean and validate calibration values
def clean_calibration(value):
    if isinstance(value, str):
        value = value.replace('%', '').strip()
        try:
            float_value = float(value)
            int_value = int(round(float_value))
            if 0 <= int_value <= 100:
                return int_value
        except ValueError:
            pass
    elif isinstance(value, (int, float)):
        if 0 <= value <= 100:
            return int(round(value))
    return None

# Apply cleaning function and remove invalid values
calibration = calibration.apply(clean_calibration)
calibration = calibration.dropna()

# Convert status to binary (1 for correct, 0 for incorrect)
y_true = (status == 'correct').astype(int)

# Ensure calibration and y_true have the same index
calibration, y_true = calibration.align(y_true, join='inner')

# Convert to numpy arrays
calibration_prob = calibration.to_numpy() / 100  # Convert to probability scale
y_true = y_true.to_numpy()

# Calculate calibration curves (uniform and quantile)
prob_true_uniform, prob_pred_uniform = calibration_curve(y_true, calibration_prob, n_bins=10, strategy='uniform')
prob_true_quantile, prob_pred_quantile = calibration_curve(y_true, calibration_prob, n_bins=10, strategy='quantile')

# Create the calibration curve plot
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.plot(prob_pred_uniform, prob_true_uniform, marker='o', linewidth=2, color='red', label='Uniform bins')
plt.plot(prob_pred_quantile, prob_true_quantile, marker='s', linewidth=2, color='blue', label='Quantile bins')

# Customize the plot
plt.xlabel('Mean predicted probability', fontsize=12)
plt.ylabel('Fraction of correct answers', fontsize=12)
plt.title('Calibration Curves for GPT4o on the MultiMedQA Benchmark', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

# Set axis limits and ticks
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.yticks(np.arange(0, 1.1, 0.2))

# Add a light blue background
plt.gca().set_facecolor('#F0F8FF')

# Add text for additional information
plt.text(0.05, 0.95, f'Total samples: {len(y_true)}', transform=plt.gca().transAxes, fontsize=10)
plt.text(0.05, 0.90, f'Accuracy: {np.mean(y_true):.2f}', transform=plt.gca().transAxes, fontsize=10)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('calibration_curves_gpt4o_multimedqa.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a bar chart for the distribution of predicted percentages
plt.figure(figsize=(12, 6))

# Round calibration to nearest 5%
rounded_calibration = (calibration // 5) * 5

# Count occurrences of each rounded percentage
percentage_counts = Counter(rounded_calibration)

# Prepare data for plotting
percentages = list(range(0, 105, 5))
counts = [percentage_counts.get(p, 0) for p in percentages]

# Create the bar chart
plt.bar(percentages, counts, width=4, align='center', alpha=0.7)

# Customize the plot
plt.xlabel('Predicted Percentage', fontsize=12)
plt.ylabel('Number of Predictions', fontsize=12)
plt.title('Distribution of Predicted Percentages for GPT4o on MultiMedQA (Rounded to 5%)', fontsize=14)
plt.xticks(percentages[::2])  # Show every other tick to avoid crowding
plt.grid(axis='y', linestyle=':', alpha=0.7)

# Add value labels on top of each bar
for i, count in enumerate(counts):
    if count > 0:
        plt.text(percentages[i], count, str(count), ha='center', va='bottom')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('predicted_percentage_distribution_gpt4o_multimedqa.png', dpi=300, bbox_inches='tight')
plt.close()

# Print statistics
print(f"Total samples: {len(y_true)}")
print(f"Accuracy: {np.mean(y_true):.4f}")
print(f"Mean calibration: {np.mean(calibration_prob):.4f}")
print(f"Correlation between predicted and actual: {np.corrcoef(calibration_prob, y_true)[0, 1]:.2f}")

# Calculate ECE (Expected Calibration Error) for both strategies
ece_uniform = np.mean(np.abs(prob_true_uniform - prob_pred_uniform))
ece_quantile = np.mean(np.abs(prob_true_quantile - prob_pred_quantile))
print(f"Expected Calibration Error (Uniform): {ece_uniform:.4f}")
print(f"Expected Calibration Error (Quantile): {ece_quantile:.4f}")


# Calculate calibration curves (uniform and quantile)
prob_true_uniform, prob_pred_uniform = calibration_curve(y_true, calibration_prob, n_bins=10, strategy='uniform')
prob_true_quantile, prob_pred_quantile = calibration_curve(y_true, calibration_prob, n_bins=10, strategy='quantile')

# [Rest of the previous code remains the same]

# Add comparison printing for uniform bins
print("\nComparison for Uniform Bins:")
for pred, true in zip(prob_pred_uniform, prob_true_uniform):
    print(f"Predicted: {pred:.2f}, Actual: {true:.2f}")

# Add comparison printing for quantile bins
print("\nComparison for Quantile Bins:")
for pred, true in zip(prob_pred_quantile, prob_true_quantile):
    print(f"Predicted: {pred:.2f}, Actual: {true:.2f}")

# Calculate and print the differences
print("\nDifferences (Predicted - Actual):")
print("Uniform Bins:", np.round(prob_pred_uniform - prob_true_uniform, 2))
print("Quantile Bins:", np.round(prob_pred_quantile - prob_true_quantile, 2))

# Find the maximum overconfidence and underconfidence
max_overconfidence_uniform = max(prob_pred_uniform - prob_true_uniform)
max_underconfidence_uniform = min(prob_pred_uniform - prob_true_uniform)
max_overconfidence_quantile = max(prob_pred_quantile - prob_true_quantile)
max_underconfidence_quantile = min(prob_pred_quantile - prob_true_quantile)

print(f"\nUniform Bins - Max Overconfidence: {max_overconfidence_uniform:.2f}, Max Underconfidence: {max_underconfidence_uniform:.2f}")
print(f"Quantile Bins - Max Overconfidence: {max_overconfidence_quantile:.2f}, Max Underconfidence: {max_underconfidence_quantile:.2f}")

# Find the bin with the largest discrepancy
largest_discrepancy_uniform = max(abs(prob_pred_uniform - prob_true_uniform))
largest_discrepancy_quantile = max(abs(prob_pred_quantile - prob_true_quantile))

print(f"\nLargest Discrepancy - Uniform Bins: {largest_discrepancy_uniform:.2f}")
print(f"Largest Discrepancy - Quantile Bins: {largest_discrepancy_quantile:.2f}")
