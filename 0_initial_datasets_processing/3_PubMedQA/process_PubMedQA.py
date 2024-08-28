import json
import csv

# Paths to the input JSON files
ori_pqal_path = "/Users/fmys/Documents/Data/master_thesis/3_pubmedqa/data/ori_pqal.json"
ground_truth_path = "/Users/fmys/Documents/Data/master_thesis/3_pubmedqa/data/test_ground_truth.json"

# Load the JSON data
with open(ori_pqal_path, "r") as f:
    ori_pqal_data = json.load(f)

with open(ground_truth_path, "r") as f:
    ground_truth_data = json.load(f)
    print(len(ground_truth_data))

# Prepare the CSV data
csv_data = []

for pmid, data in ori_pqal_data.items():
    if pmid in ground_truth_data:
        question = data["QUESTION"]
        contexts = " ".join(data["CONTEXTS"])
        correct_answer = ground_truth_data[pmid]

        csv_row = {
            "Question": f"Question: {question} \n\n Context: {contexts}",
            "Correct Answer": correct_answer,
            "Correct Answer ID": correct_answer,
            "Answer Options": "yes / no / maybe",
            "Dataset": "PubMedQA - PQA-L with ground truth labels",
            "Original ID": pmid
        }

        csv_data.append(csv_row)

# Write the CSV data to a file
fieldnames = ["Question", "Correct Answer", "Correct Answer ID", "Answer Options", "Dataset", "Original ID"]

with open("3_PubMedQA_only_PQALwithGroundTruthSamples.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Merged data saved to 3_PubMedQA_only_PQALwithGroundTruthSamples.csv")
