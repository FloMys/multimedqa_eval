import xml.etree.ElementTree as ET

path = "/Users/fmys/Documents/Data/master_thesis/LiveQA_MedicalTask_TREC2017/TrainingDatasets/"
file1 = "TREC-2017-7_LiveQA-Medical-Train-1.xml"
file2 = "TREC-2017-7_LiveQA-Medical-Train-2.xml"


# Parse the XML file
tree = ET.parse(path + file1)
root = tree.getroot()

# Open a text file to write the output
with open('qa_pairs.txt', 'w') as f:
    # Iterate over each NLM-QUESTION element
    for question in root.findall('NLM-QUESTION'):
        # Get the question text
        message = question.find('MESSAGE').text
        f.write(f"Question: {message}\n")
        print(f"Question: {message}\n")

        # Iterate over each SUB-QUESTION element
        for sub_question in question.findall('SUB-QUESTIONS/SUB-QUESTION'):
            # Get the focus and type of the sub-question
            focus = sub_question.find('ANNOTATIONS/FOCUS').text
            question_type = sub_question.find('ANNOTATIONS/TYPE').text
            f.write(f"Focus: {focus}, Type: {question_type}\n")
            print(f"Focus: {focus}, Type: {question_type}\n")

            # Iterate over each ANSWER element
            for answer in sub_question.findall('ANSWERS/ANSWER'):
                # Get the answer text
                answer_text = answer.text
                f.write(f"Answer: {answer_text}\n")
                print(f"Answer: {answer_text}\n")

            f.write("\n")
            print("\n")

        f.write("---\n")
        print("---\n")

print("Question-answer pairs extracted successfully.")