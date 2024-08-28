import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['GPT-Neo', 'PubMedBERT', 'BioLinkBERT', 'DRAGON', 'BioMedLM',
          'GPT-3.5', 'Med-PaLM', 'GPT-4 base', 'Med-PaLM 2', 'GPT-4o']
accuracies = [33.3, 38.1, 45.1, 47.5, 50.3, 60.3, 67.2, 86.1, 86.5, 87.27]
dates = ['Dec 20', 'Sep 21', 'Mar 22', 'Oct 22', 'Dec 22', 'Dec 22', 'Dec 22', 'Apr 23', 'May 23', 'May 24']

# Create evenly spaced x-positions
x_positions = list(range(len(models)))

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Plot lines
ax.plot(x_positions, accuracies, color='skyblue', linewidth=2, zorder=1)

# Plot data points
ax.scatter(x_positions, accuracies, s=100, color='skyblue', edgecolor='white', linewidth=2, zorder=2)

# Customize the plot
ax.set_ylim(30, 90)
ax.set_ylabel('MedQA (USMLE-Style) Accuracy (%)', fontsize=14)
ax.set_title('MedQA (USMLE) Accuracy Progression', fontsize=18, fontweight='bold', pad=20)

# Style the axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Add a light grid
ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

# Set x-ticks and labels
ax.set_xticks(x_positions)
ax.set_xticklabels(dates, rotation=45, ha='right')

# Add value labels
for i, accuracy in enumerate(accuracies):
    ax.annotate(f'{accuracy}', (x_positions[i], accuracy), xytext=(0, 7),
                textcoords='offset points', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add model names
for i, model in enumerate(models):
    ax.annotate(model, (x_positions[i], accuracies[i]), xytext=(0, -15),
                textcoords='offset points', ha='center', va='top', fontsize=12,
                fontweight='bold' if accuracies[i] > 80 else 'normal')

# Adjust layout
plt.tight_layout()

# Save as SVG
plt.savefig('medqa_accuracy_progression.svg', format='svg', dpi=300, bbox_inches='tight')

# Save as PNG
plt.savefig('medqa_accuracy_progression.png', format='png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()