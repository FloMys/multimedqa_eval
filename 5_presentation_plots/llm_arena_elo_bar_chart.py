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
    ],
    'Arena Score': [
        1287, 1271, 1263, 1247, 1186, 1207, 1199, 1185, 1228, 1200, 1157,
        1158, 1161, 1147, 1131, 1147, 1147, 1147, 1148, 1152, 1179
    ]
}

df = pd.DataFrame(data)

# Sort the DataFrame by Arena Score in ascending order
df_sorted = df.sort_values('Arena Score', ascending=True)

# Create the plot
plt.figure(figsize=(12, 8))
plt.bar(df_sorted['Model'], df_sorted['Arena Score'])

# Customize the plot
plt.title('LLM Performance Sorted by Arena Score (ELO)', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Arena Score (ELO)', fontsize=12)
plt.xticks(rotation=90, ha='right')
plt.ylim(1100, 1300)  # Set y-axis limit based on the score range

# Add value labels on top of each bar
for i, v in enumerate(df_sorted['Arena Score']):
    plt.text(i, v, f'{v}', ha='center', va='bottom')

plt.tight_layout()
# Save as SVG
plt.savefig('llm_arena_elo.svg', format='svg', dpi=300, bbox_inches='tight')

plt.show()