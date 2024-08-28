import json
# I don't know why there is a multi-choice tag when there was always only one correct option provided

path = "/Users/fmys/Documents/Data/master_thesis/medmcqa/data/"
files = ['dev.json', 'train.json']

for file in files:
    with open(path + file, 'r') as f:
        for line in f:
            question = json.loads(line)

            if len(str(question['cop'])) > 1:
                print(f"Question: {question['question']}")
                print(f"Correct options: {question['cop']}")
                # print(f"Option 1: {question['opa']}")
                # print(f"Option 2: {question['opb']}")
                # print(f"Option 3: {question['opc']}")
                # print(f"Option 4: {question['opd']}")
                print()

            # if question['choice_type'] == 'multi':
            #     print(f"Question: {question['question']}")
            #     print(f"Correct options: {question['cop']}")
                # print(f"Option 1: {question['opa']}")
                # print(f"Option 2: {question['opb']}")
                # print(f"Option 3: {question['opc']}")
                # print(f"Option 4: {question['opd']}")
            #     print()