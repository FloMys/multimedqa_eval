import csv
import json

# File paths
input_csv_path = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
output_jsonl_path = "batches_to_upload/MultiMedQA_calibration_0shot_GPT4o.jsonl"

# Create a list to store the JSONL data
jsonl_data = []

# Read the CSV file and process each row
with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
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
                        "content": "You are a helpful assistant that gives its confidence level for the answer to a multiple choice question about medical knowledge as a percentage."
                    },
                    {
                        "role": "user",
                        "content": row['Calibration Prompt 0-shot GPT4o']
                    }
                ],
                "max_tokens": 1000
            }
        }

        jsonl_data.append(entry)

# Write the JSONL data to the output file
with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    for entry in jsonl_data:
        json.dump(entry, jsonl_file)
        jsonl_file.write('\n')

print(f"Generated {len(jsonl_data)} entries in {output_jsonl_path}")
