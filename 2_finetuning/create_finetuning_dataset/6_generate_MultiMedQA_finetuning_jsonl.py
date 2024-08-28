import csv
import json
import os

path = "/Users/fmys/Documents/Data/master_thesis/finetuning_data"
files = ["MultiMedQA_finetuning.csv"]
column_names = ["Question", "Correct Answer", "Correct Answer ID", "Answer Options", "Dataset", "Unique ID", "0-shot Prompt"]
output_file = "MultiMedQA_finetuning.jsonl"

def csv_to_jsonl(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile, \
         open(output_file, 'w', encoding='utf-8') as jsonlfile:
        reader = csv.DictReader(csvfile, fieldnames=column_names)
        next(reader)  # Skip the header row

        for row in reader:
            jsonl_entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers multiple choice questions about medical knowledge."
                    },
                    {
                        "role": "user",
                        "content": row["0-shot Prompt"]
                    },
                    {
                        "role": "assistant",
                        "content": row["Correct Answer ID"]
                    }
                ]
            }
            json.dump(jsonl_entry, jsonlfile)
            jsonlfile.write('\n')

for file in files:
    input_file = os.path.join(path, file)
    output_file = os.path.join(path, output_file)
    csv_to_jsonl(input_file, output_file)

print(f"Conversion complete. Output file: {output_file}")