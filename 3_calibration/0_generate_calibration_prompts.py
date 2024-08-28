import csv
import os

# File path
file_path = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"

# Temporary file path
temp_file_path = file_path + '.temp'

# Evaluation prompt template
evaluation_prompt = """
In an evaluation you were given the following task and then you gave the answer below.
Task:
---
{0-shot_prompt}
---
Generated answer:
{answer_gpt4o_0-shot}


Please provide your confidence level (as a percentage) that your answer is correct.
To do that it is extremely important that you only answer with a number between 0 and 100.
"""

# Read the input CSV and write to a temporary file with the added prompt
with open(file_path, 'r', newline='', encoding='utf-8') as infile, \
        open(temp_file_path, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['Calibration Prompt 0-shot GPT4o']

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # Create the evaluation prompt for this row
        row['Calibration Prompt 0-shot GPT4o'] = evaluation_prompt.format(
            **{'0-shot_prompt': row['0-shot Prompt'],
               'answer_gpt4o_0-shot': row['Answer GPT4o 0-shot']}
        )

        # Write the row with the added prompt
        writer.writerow(row)

# Replace the original file with the updated one
os.replace(temp_file_path, file_path)

print(f"Processing complete. File updated: {file_path}")