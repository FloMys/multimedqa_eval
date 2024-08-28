import os
import json
import csv
from collections import Counter

path = "/Users/fmys/Documents/Data/master_thesis/1_MedQA/data_clean/questions/"
files = ['US/4_options/phrases_no_exclude_dev.jsonl', 'US/4_options/phrases_no_exclude_train.jsonl']

output_file = '1_MedQA_US4_dev_train_finetuning.csv'

# Initialize a counter to count the occurrences of each dataset
dataset_counter = Counter()

# Write the header row to the output CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Question', 'Correct Answer', 'Correct Answer ID', 'Answer Options', 'Dataset']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over each file
    for file in files:
        dataset_name = "US - 4 options"

        print(f"Processing {dataset_name} test set...")
        file_path = os.path.join(path, file)

        # Open the JSONL file and process each line
        with open(file_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)

                # Extract the relevant fields
                question = data['question']
                correct_answer = data['answer']
                correct_answer_id = data.get('answer_idx', '')
                answer_options = json.dumps(data.get('options', ''))

                # Write the data to the output CSV file
                writer.writerow({
                    'Question': question,
                    'Correct Answer': correct_answer,
                    'Correct Answer ID': correct_answer_id,
                    'Answer Options': answer_options,
                    'Dataset': f"MedQA - {dataset_name} - test"
                })

                # Increment the counter for the corresponding dataset
                dataset_counter[f"MedQA - {dataset_name} - test"] += 1

print(f"Combined data saved to {output_file}")

# Print the count of each dataset
print("\nDataset counts:")
for dataset, count in dataset_counter.items():
    print(f"{dataset}: {count}")