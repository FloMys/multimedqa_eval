import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['MedQA\nUS (4-option)', 'PubMedQA\nReasoning Required', 'MedMCQA\nDev', 'MMLU\n(Average of 6 subsets)']
gpt4o = [87.27, 74.00, 75.78, 91.82]  # GPT-4o data
gpt4_base = [86.10, 80.40, 73.66, 90.46]  # GPT-4-base data

# Set up the bar chart
x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# Create the bars (order switched)
rects1 = ax.bar(x - width/2, gpt4o, width, label='GPT-4o', color='#008000')  # Green color, now on the left
rects2 = ax.bar(x + width/2, gpt4_base, width, label='GPT-4-base', color='#FFA500')  # Orange color, now on the right

# Customize the chart
ax.set_ylabel('Performance')
ax.set_title('Performance of GPT-4o vs GPT-4-base on MultiMedQA Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Adjust layout and display
fig.tight_layout()

# Save as SVG
plt.savefig('sota_barchart.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()