import json
import csv

path = "/Users/fmys/Documents/Data/master_thesis/2_medmcqa/data/"
files = ['train.json']


data = []

for file in files:
    with open(path + file, 'r') as f:
        print(f"Processing {file}...")
        for line in f:
            item = json.loads(line)
            question = item['question']
            correct_answer = item['exp'] if item['exp'] else ''
            correct_answer_id = item['cop']
            answer_options = {
                '1': item['opa'],
                '2': item['opb'],
                '3': item['opc'],
                '4': item['opd']
            }
            dataset = "MedMCQA - " + str(file.split('.')[0])
            choice_type = item['choice_type']

            data.append([question, correct_answer, correct_answer_id, answer_options, dataset, choice_type])

with open('2_MedMCQA_train_testfinetuning.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Question', 'Correct Answer', 'Correct Answer ID', 'Answer Options', 'Dataset', 'Choice Type'])
    writer.writerows(data)

print("CSV file '2_MedMCQA_testfinetuning.csv' created successfully.")