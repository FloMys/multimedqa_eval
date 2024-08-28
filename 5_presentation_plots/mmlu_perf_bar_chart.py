import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame from the data
data = {
    'Model': [
        'GPT-4o-2024-05-13', 'Claude 3.5 Sonnet', 'Gemini-1.5-Pro-API-0514',
        'Claude 3 Opus', 'GPT-4-0314', 'Llama-3-70b-Instruct',
        'Reka-Core-20240501', 'Qwen2-72B-Instruct', 'Gemini-1.5-Flash-API-0514',
        'Claude 3 Sonnet', 'Mistral Large-2402', 'Yi-1.5-34B-Chat',
        'Qwen1.5-110B-Chat', 'Mixtral-8x22b-Instruct-v0.1', 'Claude-2.0',
        'Qwen1.5-72B-Chat', 'Mistral Medium', 'Reka-Flash-21B', 'Claude-1',
        'Llama-3-8b-Instruct', 'Claude 3 Haiku'
    ],
    'MMLU': [
        88.7, 88.7, 85.9, 86.8, 86.4, 82.0, 83.2, 84.2, 78.9, 79.0, 81.2,
        76.8, 80.4, 77.8, 78.5, 77.5, 75.3, 73.5, 77.0, 68.4, 75.2
    ]
}

df = pd.DataFrame(data)

# Sort the DataFrame by MMLU score in ascending order
df_sorted = df.sort_values('MMLU', ascending=True)

# Create the plot
plt.figure(figsize=(12, 8))
plt.bar(df_sorted['Model'], df_sorted['MMLU'])

# Customize the plot
plt.title('LLM Performance Sorted by MMLU Score', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('MMLU Score', fontsize=12)
plt.xticks(rotation=90, ha='right')
plt.ylim(50, 100)  # Set y-axis limit from 50 to 100

# Add value labels on top of each bar
for i, v in enumerate(df_sorted['MMLU']):
    plt.text(i, v, f'{v:.1f}', ha='center', va='bottom')

plt.tight_layout()
# Save as SVG
plt.savefig('mmlu_perf.svg', format='svg', dpi=300, bbox_inches='tight')

plt.show()