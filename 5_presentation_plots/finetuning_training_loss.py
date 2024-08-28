import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# File path
loss_values_xlsx = "/Users/fmys/Documents/Data/master_thesis/finetuning_data/finetuning_loss.xlsx"

# Read the Excel file
data = pd.read_excel(loss_values_xlsx)

# Reverse the order of rows to have the earliest step first
data = data.iloc[::-1].reset_index(drop=True)

# Create a new 'Step' column starting from 1 to 1573
data['Adjusted Step'] = range(1, len(data) + 1)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(data['Adjusted Step'], data['Training loss'], label='Training Loss')

# Set labels and title
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss vs. Steps')

# Add legend
plt.legend()

# Set x-axis limits and ticks
plt.xlim(1, 1573)
x_ticks = np.linspace(1, 1573, 7, dtype=int)  # 7 evenly spaced ticks
plt.xticks(x_ticks)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Customize y-axis
plt.ylim(bottom=0)  # Start y-axis from 0
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# Save the plot
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()