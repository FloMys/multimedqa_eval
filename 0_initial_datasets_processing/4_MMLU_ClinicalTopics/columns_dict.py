"""
Based on the information provided in the paper, the medical topics from the MMLU benchmark that were used are:

- Clinical Knowledge
- Medical Genetics
- Anatomy
- Professional Medicine
- College Biology
- College Medicine


Original Data Fields:
1. Question (string): The question or prompt presented to the user.
2. Option_A (string): The first answer option for the given question.
3. Option_B (string): The second answer option for the given question.
4. Option_C (string): The third answer option for the given question.
5. Option_D (string): The fourth answer option for the given question.
6. Correct_Answer (string): The letter (A, B, C, or D) corresponding to the correct answer option for the given question.

Note: The data is provided in a CSV format without a header row. Each row represents a single question with its answer options and the correct answer.

Transformed into (all datafields are strings):
"Question": question to be answered
"Correct Answers": Correct answer to the question.
"Correct Answer IDs": Only relevant if answers are multiple-choice otherwise leave empty.
"Answer Options": Only relevant if answers are multiple-choice otherwise leave empty.
"Dataset": Name of the dataset this question is from ("MMLU - folder name - category_name")

"""