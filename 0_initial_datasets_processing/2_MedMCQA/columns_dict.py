"""
https://github.com/medmcqa/medmcqa?tab=readme-ov-file

Original Data Fields
id : a string question identifier for each example
question : question text (a string)
opa : Option A
opb : Option B
opc : Option C
opd : Option D
cop : Correct option (Answer of the question) (always only one correct option provided!!!)
choice_type : Question is single-choice or multi-choice
exp : Expert's explanation of the answer
subject_name : Medical Subject name of the particular question
topic_name : Medical topic name from the particular subject


Transformed into:
"Question": question to be answered.
"Correct Answers": Correct answer to the question (here this is the expert's explanation of the answer)
"Correct Answer IDs": Only relevant if answers are multiple-choice otherwise leave empty. Contains the ID (letter or number) of the correct answer option.
"Answer Options": Only relevant if answers are multiple-choice otherwise leave empty. Contains all the presented answer options and their corresponding ID.
"Dataset": Name of the dataset this question is from (MedMCQA - train, MedMCQA - dev, MedMCQA - test)
"Choice Type": Question is single-choice or multi-choice

"""