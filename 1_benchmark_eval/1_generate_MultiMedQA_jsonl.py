import csv
import json
import os

MODEL_NAMES = [
    ("gpt-4-turbo-2024-04-09", "GPT4turbo"),
    ("gpt-4o-2024-05-13", "GPT4o"),
    ("gpt-3.5-turbo-0125", "GPT3.5turbo"),
    ("ft:gpt-3.5-turbo-0125:patentplus-io:multimedqa-ft:9jBQiyLM", "GPT3.5turbo_finetuned")
]

path = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"

output_files = [
    f"MultiMedQA_{shot}shot_{model[1]}.jsonl"
    for model in MODEL_NAMES
    for shot in ["0", "5"]
]

sample_counts = {file: 0 for file in output_files}

with open(path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

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

                file_name = output_files[model_index * 2 + prompt_index]

                with open(file_name, 'a') as jsonl_file:
                    jsonl_file.write(json.dumps(jsonl_data) + "\n")

                sample_counts[file_name] += 1

for file_name, count in sample_counts.items():
    print(f"Number of samples in {file_name}: {count}")