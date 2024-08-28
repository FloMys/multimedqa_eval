import csv
import json
import os

# File paths
csv_file_path = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
batch_outputs_dir = "downloaded_batch_outputs"

# Column names
column_names = [
    "Question",
    "Correct Answer",
    "Correct Answer ID",
    "Answer GPT4o 0-shot",
    "Answer GPT4turbo 0-shot",
    "Answer GPT3.5Turbo 0-shot",
    "Answer GPT3.5TurboFinetuned 0-shot",
    "Answer GPT4o 5-shot",
    "Answer GPT4turbo 5-shot",
    "Answer GPT3.5Turbo 5-shot",
    "Answer GPT3.5TurboFinetuned 5-shot",
    "Answer Options",
    "Dataset",
    "Unique ID",
    "0-shot Prompt",
    "5-shot Prompt",
    "Calibration Prompt 0-shot GPT4o",
    "Answer Calibration GPT4o 0-shot",
    "Answer Status GPT4o 0shot",
    "Memorization q1",
    "Memorization q2",
    "Memorization Prompt",
    "Memorization Answer GPT4o"
]


# Function to read JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Read all JSONL files
jsonl_files = {
    "memorization_GPT4o": "batch_info_MultiMedQA_memorization_GPT4o_output.jsonl"
}

answer_dicts = {}
for key, filename in jsonl_files.items():
    file_path = os.path.join(batch_outputs_dir, filename)
    data = read_jsonl(file_path)
    answer_dicts[key] = {item["custom_id"]: item["response"]["body"]["choices"][0]["message"]["content"] for item in data}

# Read the CSV file
with open(csv_file_path, "r") as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)

# Update the answer columns based on the custom_id mapping
for row in rows:
    unique_id = row["Unique ID"]
    row["Memorization Answer GPT4o"] = answer_dicts["memorization_GPT4o"].get(unique_id, "")

# Write the updated rows back to the CSV file
with open(csv_file_path, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=column_names)
    writer.writeheader()
    writer.writerows(rows)

print(f"Answers saved to {csv_file_path}.")
