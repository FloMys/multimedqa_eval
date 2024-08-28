import pandas as pd
import math

# File path
file_path = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Function to split a string into two halves
def split_string(text):
    words = text.split()
    mid = math.ceil(len(words) / 2)
    return ' '.join(words[:mid]), ' '.join(words[mid:])

# Apply the split_string function to the 'Question' column
df[['Memorization q1', 'Memorization q2']] = df['Question'].apply(lambda x: pd.Series(split_string(x)))

# Function to create the memorization prompt
def create_memorization_prompt(dataset, q1):
    return f"""In the following you will see the first half (Q1) of a question from the dataset {dataset}.
---
Q1: {q1}
---

Please complete the second half of the question correctly word for word.
It is extremely important that you only answer with the second half of the question and nothing else."""

# Apply the create_memorization_prompt function to create the 'Memorization Prompt' column
df['Memorization Prompt'] = df.apply(lambda row: create_memorization_prompt(row['Dataset'], row['Memorization q1']), axis=1)

# Ensure all column names are present
all_columns = [
    "Question", "Correct Answer", "Correct Answer ID",
    "Answer GPT4o 0-shot", "Answer GPT4turbo 0-shot",
    "Answer GPT3.5Turbo 0-shot", "Answer GPT3.5TurboFinetuned 0-shot",
    "Answer GPT4o 5-shot", "Answer GPT4turbo 5-shot",
    "Answer GPT3.5Turbo 5-shot", "Answer GPT3.5TurboFinetuned 5-shot",
    "Answer Options", "Dataset", "Unique ID",
    "0-shot Prompt", "5-shot Prompt",
    "Calibration Prompt 0-shot GPT4o",
    "Answer Calibration GPT4o 0-shot",
    "Answer Status GPT4o 0shot",
    "Memorization q1", "Memorization q2",
    "Memorization Prompt"
]

# Add any missing columns with empty values
for col in all_columns:
    if col not in df.columns:
        df[col] = ''

# Reorder columns to match the desired order
df = df[all_columns]

# Save the modified DataFrame back to the same CSV file
df.to_csv(file_path, index=False)

print("CSV file has been updated with 'Memorization q1', 'Memorization q2', and 'Memorization Prompt' columns.")