import csv
import json
import tiktoken
import os

# File paths
input_csv_path = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
output_jsonl_path = "batches_to_upload/MultiMedQA_memorization_GPT4o.jsonl"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

# Initialize the tokenizer for GPT-4o (o200k_base)
enc = tiktoken.encoding_for_model("gpt-4o")

# Create a list to store the JSONL data
jsonl_data = []

# Read the CSV file and process each row
with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
        # Calculate the number of tokens in "Memorization q2"
        q2_tokens = len(enc.encode(row['Memorization q2']))

        # Set max_tokens to the number of tokens in "Memorization q2" (second half of the question)
        max_tokens = q2_tokens

        # Create the JSONL entry
        entry = {
            "custom_id": row['Unique ID'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-2024-05-13",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that completes the second half of a question from memory."
                    },
                    {
                        "role": "user",
                        "content": row['Memorization Prompt']
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 42
            }
        }

        jsonl_data.append(entry)

# Write the JSONL data to the output file
with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    for entry in jsonl_data:
        json.dump(entry, jsonl_file)
        jsonl_file.write('\n')

print(f"Generated {len(jsonl_data)} entries in {output_jsonl_path}")