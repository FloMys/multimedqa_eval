import csv
import os
from collections import Counter

path = "/Users/fmys/Documents/Data/master_thesis/finetuning_data"
files = ["1_MedQA_US4_dev_train_finetuning.csv",
         "2_MedMCQA_train_testfinetuning.csv",
         "4_MMLU_CT_dev_val_finetuning.csv"]

column_names = ["Question", "Correct Answer", "Correct Answer ID", "Answer Options", "Dataset", "Unique ID",
                "0-shot Prompt"]

output_file = "MultiMedQA_finetuning.csv"

total_rows = 0
dataset_counter = Counter()

with open(os.path.join(path, output_file), 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(column_names)

    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip the header row

            rows = list(reader)
            num_rows = len(rows)

            # If it's the MedMCQA dataset, keep only the first 10%
            if file == "2_MedMCQA_train_testfinetuning.csv":
                num_rows_to_keep = int(num_rows * 0.1)
                rows = rows[:num_rows_to_keep]
                num_rows = num_rows_to_keep

            total_rows += num_rows

            print(f"Number of rows in {file}: {num_rows}")

            for row_number, row in enumerate(rows, start=1):
                question = row[0]
                correct_answer = row[1]
                correct_answer_id = row[2]
                answer_options = row[3]
                dataset = row[4]
                unique_id = f"{dataset} - {row_number}"

                dataset_counter[dataset] += 1

                prompt = f"""
In the following you will receive a question and a list of answer options.

{{}}

Answer Options:
{{}}

It is extremely important that you only answer with one of the provided answer options.
This means you should only answer with a single letter, single number or yes/no/maybe according to the provided answer options.
""".format(question, answer_options)

                writer.writerow(
                    [question, correct_answer, correct_answer_id, answer_options, dataset, unique_id, prompt])

print(f"Combined CSV file '{output_file}' has been created successfully.")
print(f"Total number of rows in the resulting dataset: {total_rows}")
print("\nNumber of rows for each sub-dataset:")
for dataset, count in dataset_counter.items():
    print(f"{dataset}: {count}")