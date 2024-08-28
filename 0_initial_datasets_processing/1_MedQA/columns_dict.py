"""
Original Data Fields

question (string): The medical question being asked. Includes background information about the patient.
options (dict): A dictionary containing the multiple choice answer options, with keys "A", "B", "C", "D" mapping to the text of each answer choice.
answer_idx (string): The letter corresponding to the correct answer in the options dict (e.g. "A", "B", "C", "D"). Only one correct answer / best answer per question!
answer (string): The correct answer to the medical question. Only one correct answer / best answer per question!
meta_info (string): Metadata about the question, such as what exam it is from (e.g. "step1", "step2&3").
metamap_phrases (list of strings): A list of medical phrases and concepts extracted from the question text using the MetaMap tool. Helps identify key entities.


Transformed into:
"Question": question to be answered.
"Correct Answer": Correct answer to the question.
"Correct Answer ID": Only relevant if answers are multiple-choice otherwise leave empty. Contains the ID (letter or number) of the correct answer option.
"Answer Options": Only relevant if answers are multiple-choice otherwise leave empty. Contains all the presented answer options including their IDs.
"Dataset": Name of the dataset this question is from (MedQA - Mainland - test, MedQA - Taiwan - test, MedQA - US - test, MedQA - US - 4 options - test)
"""

