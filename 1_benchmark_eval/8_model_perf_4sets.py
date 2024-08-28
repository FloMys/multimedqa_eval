import json
import matplotlib.pyplot as plt
import numpy as np

# Read the evaluation results from the specified file
eval_results_file = "/Users/fmys/Documents/Data/master_thesis/evaluation_results.json"

with open(eval_results_file, 'r') as file:
    results = json.load(file)

# Data
datasets = [
    "MedQA", "PubMedQA", "MedMCQA Dev", "MMLU (Clinical Topics Avg)"
]

def get_best_accuracy(model_base, dataset):
    zero_shot = results['dataset_results'][dataset][f"{model_base} 0-shot"]['accuracy']
    five_shot = results['dataset_results'][dataset][f"{model_base} 5-shot"]['accuracy']
    if zero_shot > five_shot:
        print(f"{model_base} on {dataset}: 0-shot ({zero_shot:.2f})")
        return zero_shot
    else:
        print(f"{model_base} on {dataset}: 5-shot ({five_shot:.2f})")
        return five_shot

# Calculate average for MMLU clinical topics
mmlu_topics = ["clinical knowledge", "medical genetics", "anatomy", "professional medicine", "college biology", "college medicine"]

def get_mmlu_avg(model_base):
    accuracies = []
    for topic in mmlu_topics:
        dataset = f"MMLU - test - {topic}"
        accuracies.append(get_best_accuracy(model_base, dataset))
    return np.mean(accuracies)

models = ["GPT4o", "GPT4 base", "GPT4turbo", "GPT3.5TurboFinetuned", "GPT3.5Turbo"]
# models = ["GPT-4o", "GPT-4-base", "GPT-4 Turbo", "GPT-3.5 Turbo Finetuned", "GPT-3.5 Turbo"]
data = {model: [] for model in models}

for model in models:
    if model == "GPT4 base":
        # Manually add GPT4 base data, using the correct 86.10 for MedQA US 4 options
        data[model] = [86.10, 80.40, 73.66, 90.46]
    else:
        data[model] = [
            get_best_accuracy(model, "MedQA - US - 4 options - test"),
            get_best_accuracy(model, "PubMedQA - PQA-L with ground truth labels"),
            get_best_accuracy(model, "MedMCQA - dev"),
            get_mmlu_avg(model)
        ]
    print("\n\n")

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 9))

# Set the width of each bar and the positions of the bars
width = 0.18
x = np.arange(len(datasets))

# Colors for each model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create the bars
for i, model in enumerate(models):
    ax.bar(x + (i-2)*width, data[model], width, label=model, color=colors[i])

# Customize the plot
ax.set_ylabel('Accuracy (%)')
ax.set_title('Best Model Performance Across the MultiMedQA Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')

# Place the legend inside the plot area, in the upper right corner
ax.legend(loc='best')

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of each bar
for i, model in enumerate(models):
    for j, v in enumerate(data[model]):
        ax.text(j + (i-2)*width, v, f'{v:.1f}', ha='center', va='bottom')

# Adjust the layout
plt.tight_layout()

# Save the figure as SVG and PNG
plt.savefig('multimedqa_performance_comparison.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('multimedqa_performance_comparison.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved as 'multimedqa_performance_comparison_taller.svg' and 'multimedqa_performance_comparison_taller.png'")
