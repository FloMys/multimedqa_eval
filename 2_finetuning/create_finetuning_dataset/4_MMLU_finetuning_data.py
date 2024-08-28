import os
import csv

path = "/Users/fmys/Documents/Data/master_thesis/4_mmlu/data/"
dirs = ['dev', 'val']
clinical_categories = ["clinical knowledge", "medical genetics", "anatomy", "professional medicine", "college biology",
                       "college medicine"]

output_file = "4_MMLU_CT_dev_val_finetuning.csv"

with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Question", "Correct Answer", "Correct Answer ID", "Answer Options", "Dataset"])

    for dir in dirs:
        for category in clinical_categories:
            file_path = os.path.join(path, dir, f"{category.replace(' ', '_')}_{dir}.csv")

            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    print(f"Processing file: {file_path}")

                    for row in reader:
                        question = row[0]
                        correct_answer = row[-1]
                        answer_options = row[1:-1]

                        # Add IDs to answer options
                        answer_options_with_ids = [f"{chr(65 + i)}. {option}" for i, option in
                                                   enumerate(answer_options)]

                        writer.writerow([
                            question,
                            correct_answer,
                            correct_answer,
                            "|".join(answer_options_with_ids),
                            f"MMLU - {dir} - {category}"
                        ])

print("Data merged successfully!")