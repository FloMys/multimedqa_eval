"""
Evaluation for the 4 different benchmark settings.
"""
import csv
from collections import Counter, defaultdict
import tempfile
import os

def clean_text(text):
    text = text.strip()
    text = text.replace("\n", "")
    text = text.replace("\"", "")
    text = text.replace("'", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    return text

file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"

column_names = [
    "Question", "Correct Answer", "Correct Answer ID",
    "Answer GPT4o 0-shot", "Answer GPT4turbo 0-shot",
    "Answer GPT4o 5-shot", "Answer GPT4turbo 5-shot",
    "Answer Options", "Dataset", "Unique ID", "0-shot Prompt",
    "5-shot Prompt", "Calibration Prompt 0-shot GPT4o",
    "Answer Calibration GPT4o 0-shot", "Answer Status GPT4o 0shot"
]

settings = ["GPT4o 0-shot"]
correct_counts = {setting: 0 for setting in settings}
total_count = 0
dataset_tags = []
dataset_correct_counts = {setting: defaultdict(int) for setting in settings}
dataset_total_counts = defaultdict(int)
wrong_answers = {setting: [] for setting in settings}

# Create a temporary file
temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')

with open(file, 'r') as csv_file, temp_file:
    csv_reader = csv.DictReader(csv_file)
    csv_writer = csv.DictWriter(temp_file, fieldnames=column_names)

    csv_writer.writeheader()

    for row in csv_reader:
        correct_answer = row["Correct Answer ID"]
        dataset_tag = row["Dataset"]
        dataset_tags.append(dataset_tag)

        for setting in settings:
            model_answer = clean_text(row[f"Answer {setting}"])

            if correct_answer == model_answer[0] or correct_answer == model_answer.lower():
                correct_counts[setting] += 1
                dataset_correct_counts[setting][dataset_tag] += 1
                row["Answer Status GPT4o 0shot"] = "correct"
            else:
                wrong_answers[setting].append(model_answer)
                row["Answer Status GPT4o 0shot"] = "wrong"

        total_count += 1
        dataset_total_counts[dataset_tag] += 1

        # Write the updated row to the temporary file
        csv_writer.writerow(row)

# Close the temp file
temp_file.close()

# Replace the original file with the temporary file
os.replace(temp_file.name, file)

# Print overall results
print("\nOverall Results:")
for setting in settings:
    accuracy = correct_counts[setting] / total_count * 100
    print(f"{setting} - Accuracy: {accuracy:.2f}%")
    print(f"{setting} - Correct Answers: {correct_counts[setting]}/{total_count}")
    print()

# Print results for each dataset
dataset_tag_counts = Counter(dataset_tags)
print(dataset_tag_counts)
for tag, count in dataset_tag_counts.items():
    print(f"\nResults for {tag}:")
    for setting in settings:
        tag_accuracy = dataset_correct_counts[setting][tag] / dataset_total_counts[tag] * 100
        print(f"{setting} - Accuracy: {tag_accuracy:.2f}%")
    print()

print(f"The original CSV file has been updated: {file}")