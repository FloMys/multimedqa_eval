import csv
import json

MODEL_NAMES = [
    ("gpt-4-turbo-2024-04-09", "GPT4turbo"),
    ("gpt-4o-2024-05-13", "GPT4o")
]

# TODO: add file path to "batches_to_upload" dir (save the JSONL files in this directory)

path = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
output_files = [
    "MultiMedQA_0shot_GPT4turbo.jsonl",
    "MultiMedQA_5shot_GPT4turbo.jsonl",
    "MultiMedQA_0shot_GPT4o.jsonl",
    "MultiMedQA_5shot_GPT4o.jsonl"
]

sample_counts = [0, 0, 0, 0]

with open(path, 'r') as csv_file, \
     open(output_files[0], 'w') as jsonl_file_0shot_turbo, \
     open(output_files[1], 'w') as jsonl_file_5shot_turbo, \
     open(output_files[2], 'w') as jsonl_file_0shot_o, \
     open(output_files[3], 'w') as jsonl_file_5shot_o:

    csv_reader = csv.DictReader(csv_file)
    jsonl_files = [jsonl_file_0shot_turbo, jsonl_file_5shot_turbo, jsonl_file_0shot_o, jsonl_file_5shot_o]

    for row in csv_reader:
        for model_index, (model, model_name) in enumerate(MODEL_NAMES):
            for prompt_index, prompt_type in enumerate(['0-shot Prompt', '5-shot Prompt']):
                jsonl_data = {
                    "custom_id": row['Unique ID'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that answers multiple choice questions about medical knowledge."
                            },
                            {
                                "role": "user",
                                "content": row[prompt_type]
                            }
                        ],
                        "max_tokens": 1000
                    }
                }

                file_index = model_index * 2 + prompt_index
                jsonl_files[file_index].write(json.dumps(jsonl_data) + "\n")
                sample_counts[file_index] += 1

for i, file_name in enumerate(output_files):
    print(f"Number of samples in {file_name}: {sample_counts[i]}")