import csv
import json
from collections import Counter, defaultdict


def clean_text(text):
    text = text.strip()
    text = text.replace("\n", "")
    text = text.replace("\"", "")
    text = text.replace("'", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    return text


file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"

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
    "Answer Status GPT4o 0shot"
]

settings = [
    "GPT4o 0-shot",
    "GPT4turbo 0-shot",
    "GPT3.5Turbo 0-shot",
    "GPT3.5TurboFinetuned 0-shot",
    "GPT4o 5-shot",
    "GPT4turbo 5-shot",
    "GPT3.5Turbo 5-shot",
    "GPT3.5TurboFinetuned 5-shot"
]

correct_counts = {setting: 0 for setting in settings}
total_count = 0
dataset_tags = []
dataset_correct_counts = {setting: defaultdict(int) for setting in settings}
dataset_total_counts = defaultdict(int)
wrong_answers = {setting: [] for setting in settings}

with open(file, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
        correct_answer = row["Correct Answer ID"]
        dataset_tag = row["Dataset"]
        dataset_tags.append(dataset_tag)

        for setting in settings:
            model_answer = clean_text(row[f"Answer {setting}"])

            if not model_answer:
                print(f"Warning: Empty answer for {setting} on question {row['Question'][:50]}...")

            if correct_answer == model_answer or (
                    model_answer and correct_answer == model_answer[0]) or correct_answer == model_answer.lower():
                correct_counts[setting] += 1
                dataset_correct_counts[setting][dataset_tag] += 1
            else:
                wrong_answers[setting].append(model_answer)

        total_count += 1
        dataset_total_counts[dataset_tag] += 1

# Prepare results dictionary
results = {
    "overall_results": {},
    "dataset_results": {},
    "dataset_counts": dict(Counter(dataset_tags))
}

# Overall results
for setting in settings:
    accuracy = correct_counts[setting] / total_count * 100
    results["overall_results"][setting] = {
        "accuracy": round(accuracy, 2),
        "correct_answers": correct_counts[setting],
        "total_questions": total_count
    }

# Dataset results
for tag in set(dataset_tags):
    results["dataset_results"][tag] = {}
    for setting in settings:
        tag_accuracy = dataset_correct_counts[setting][tag] / dataset_total_counts[tag] * 100
        results["dataset_results"][tag][setting] = {
            "accuracy": round(tag_accuracy, 2),
            "correct_answers": dataset_correct_counts[setting][tag],
            "total_questions": dataset_total_counts[tag]
        }

# Save results to JSON file
output_file = "/Users/fmys/Documents/Data/master_thesis/evaluation_results.json"
with open(output_file, 'w') as json_file:
    json.dump(results, json_file, indent=2)

print(f"Results have been saved to {output_file}")

# Print overall results (optional)
print("\nOverall Results:")
for setting, data in results["overall_results"].items():
    print(f"{setting} - Accuracy: {data['accuracy']}%")
    print(f"{setting} - Correct Answers: {data['correct_answers']}/{data['total_questions']}")
    print()

# Print results for each dataset (optional)
for tag, data in results["dataset_results"].items():
    print(f"\nResults for {tag}:")
    for setting, setting_data in data.items():
        print(f"{setting} - Accuracy: {setting_data['accuracy']}%")
    print()